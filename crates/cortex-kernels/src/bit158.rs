//! 1.58-bit ternary linear layer — Tile16x4 / CacheLine16x16 packing.
//!
//! Layout (Phase 1 spec):
//! * [`Tile16x4`]: `#[repr(C, align(16))]`, 16 bytes = 16 output rows
//!   x 4 input cols = 64 ternary weights, 2 bits each.
//!   Encoding: `00 -> 0`, `01 -> +1`, `11 -> -1`; `10` is reserved
//!   and decodes as `0`.
//!   Packing order inside a tile (shared with `tools/train_seed_qat.py`):
//!   `idx = (out % 16) * 4 + (in % 4)`, `byte = idx / 4`,
//!   `shift = (idx % 4) * 2`.
//! * [`CacheLine16x16`]: `#[repr(C, align(64))]`, 64 bytes wrapping
//!   4 consecutive [`Tile16x4`]s. The weight matrix is stored as a
//!   row-major sequence of tiles, grouped 4-per-cache-line.
//!
//! Forward path: per-token symmetric INT8 activation quantisation
//! (`[-127, +127]`, `-128` is never produced, so negation cannot
//! overflow), integer-only inner loop, and a single dequantisation
//! per row at the end: `y[r] = acc[r] * act_scale * row_scale[r]`.
//!
//! Hardware dispatch (`forward_gemv`):
//! * x86_64 + AVX2: the row is unpacked in 32-weight blocks where
//!   `_mm256_shuffle_epi8` performs the code -> sign table lookup
//!   (`[0, 1, 0, -1]`), and the dot product accumulates in `i16`
//!   for 128 columns before widening to `i32` (128 * 127 = 16256
//!   fits in `i16`, so this cannot overflow).
//! * aarch64 + NEON: `vqtbl1q_u8` does the same table lookup, with
//!   `vaddl_s8` / `vaddl_high_s8` widening to `i16`.
//! * Otherwise (and as the golden reference for tests): scalar packed
//!   decode. `forward_gemv_kernel(..., GemvKernel::Scalar)` forces it.

/// One 16-byte tile: 16 output rows x 4 input cols, 2 bits per weight.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tile16x4(pub [u8; 16]);

/// One 64-byte cache line: 4 consecutive [`Tile16x4`]s.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheLine16x16(pub [Tile16x4; 4]);

/// Which GEMV kernel to use. `Auto` picks AVX2/NEON when available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemvKernel {
    Auto,
    Scalar,
    Avx2,
    Neon,
}

/// A ternary weight matrix `W: [rows, cols]` (rows = out channels,
/// cols = in features), stored as Tile16x4 packed bytes grouped into
/// cache lines, plus one per-row FP32 scale `gamma`.
#[derive(Clone, Debug)]
pub struct BitLinearWeights {
    pub rows: usize,
    pub cols: usize,
    /// Row scales, one per output channel.
    pub scales: Vec<f32>,
    cache_lines: Vec<CacheLine16x16>,
    in_blocks: usize,
    out_blocks: usize,
    num_tiles: usize,
}

impl BitLinearWeights {
    /// Build from a row-major ternary matrix (`-1 / 0 / +1`).
    ///
    /// Cross-crate API relied on by `cortex-hippocampus`,
    /// `cortex-ssm` and the `.ctxb` loader.
    pub fn from_ternary_matrix(w: &[i8], rows: usize, cols: usize, scales: &[f32]) -> Self {
        assert_eq!(w.len(), rows * cols, "ternary matrix size mismatch");
        assert_eq!(scales.len(), rows, "one scale per output row required");
        assert!(rows > 0 && cols > 0);
        let out_blocks = rows.div_ceil(16);
        let in_blocks = cols.div_ceil(4);
        let num_tiles = out_blocks * in_blocks;
        let num_lines = num_tiles.div_ceil(4);
        let mut cache_lines = vec![CacheLine16x16::default(); num_lines];
        for r in 0..rows {
            for c in 0..cols {
                let v = w[r * cols + c];
                debug_assert!((-1..=1).contains(&v), "ternary weight must be -1/0/+1");
                let code: u8 = match v {
                    1 => 0b01,
                    -1 => 0b11,
                    _ => 0b00,
                };
                let tile = (r / 16) * in_blocks + (c / 4);
                let idx = (r % 16) * 4 + (c % 4);
                let byte = idx / 4;
                let shift = (idx % 4) * 2;
                cache_lines[tile / 4].0[tile % 4].0[byte] |= code << shift;
            }
        }
        Self {
            rows,
            cols,
            scales: scales.to_vec(),
            cache_lines,
            in_blocks,
            out_blocks,
            num_tiles,
        }
    }

    /// Number of 16-byte tiles.
    pub fn num_tiles(&self) -> usize {
        self.num_tiles
    }

    /// Number of 16-row output blocks / 4-column input blocks.
    pub fn out_blocks(&self) -> usize {
        self.out_blocks
    }
    pub fn in_blocks(&self) -> usize {
        self.in_blocks
    }

    /// Packed size in bytes (16 bytes per tile, cache-line padded).
    pub fn packed_bytes(&self) -> usize {
        self.cache_lines.len() * 64
    }

    /// Raw tile bytes, in tile order (for the `.ctxb` writer / tests).
    pub fn tile_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.num_tiles * 16);
        for t in 0..self.num_tiles {
            out.extend_from_slice(&self.tile_at(t));
        }
        out
    }

    /// Rebuild from raw tile bytes produced by [`BitLinearWeights::tile_bytes`].
    pub fn from_tile_bytes(tiles: &[u8], rows: usize, cols: usize, scales: &[f32]) -> Self {
        let out_blocks = rows.div_ceil(16);
        let in_blocks = cols.div_ceil(4);
        let num_tiles = out_blocks * in_blocks;
        assert_eq!(tiles.len(), num_tiles * 16, "tile byte length mismatch");
        assert_eq!(scales.len(), rows);
        let num_lines = num_tiles.div_ceil(4);
        let mut cache_lines = vec![CacheLine16x16::default(); num_lines];
        for t in 0..num_tiles {
            cache_lines[t / 4].0[t % 4]
                .0
                .copy_from_slice(&tiles[t * 16..(t + 1) * 16]);
        }
        Self {
            rows,
            cols,
            scales: scales.to_vec(),
            cache_lines,
            in_blocks,
            out_blocks,
            num_tiles,
        }
    }

    #[inline]
    fn tile_at(&self, tile: usize) -> [u8; 16] {
        self.cache_lines[tile / 4].0[tile % 4].0
    }

    /// Decode one ternary weight.
    pub fn get(&self, r: usize, c: usize) -> i8 {
        assert!(r < self.rows && c < self.cols);
        let tile = (r / 16) * self.in_blocks + (c / 4);
        let idx = (r % 16) * 4 + (c % 4);
        let code = (self.tile_at(tile)[idx / 4] >> ((idx % 4) * 2)) & 0b11;
        match code {
            0b01 => 1,
            0b11 => -1,
            _ => 0, // 00 and reserved 10 both decode as 0
        }
    }

    /// Unpack a whole row into `out` (length `cols`).
    ///
    /// Iterates tile-by-tile (no per-element division: `get` recomputes
    /// the tile index with `/` and `%` per element, which made the
    /// AVX2 GEMV path ~10x slower than the dot product itself).
    pub fn unpack_row(&self, r: usize, out: &mut [i8]) {
        assert_eq!(out.len(), self.cols);
        let out_block = r / 16;
        let row_in_tile = r % 16;
        for in_block in 0..self.in_blocks {
            let tile = self.tile_at(out_block * self.in_blocks + in_block);
            for j in 0..4 {
                let c = in_block * 4 + j;
                if c >= self.cols {
                    break;
                }
                let idx = row_in_tile * 4 + j;
                let code = (tile[idx / 4] >> ((idx % 4) * 2)) & 0b11;
                out[c] = match code {
                    0b01 => 1,
                    0b11 => -1,
                    _ => 0,
                };
            }
        }
    }

    /// Quantise an f32 activation vector to INT8 (symmetric, per-vector).
    /// Returns `(quantised, scale)`; `scale == 0` iff the input is all zero.
    /// The quantised values are always in `[-127, 127]` — never `-128`.
    pub fn quantize_activation(x: &[f32]) -> (Vec<i8>, f32) {
        let max_abs = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if max_abs == 0.0 {
            return (vec![0i8; x.len()], 0.0);
        }
        let scale = max_abs / 127.0;
        let q: Vec<i8> = x
            .iter()
            .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        (q, scale)
    }

    /// `y = W @ x`, `x: f32[cols]`, `y: f32[rows]`, auto hardware dispatch.
    pub fn forward_gemv(&self, x: &[f32], out: &mut [f32]) {
        self.forward_gemv_kernel(x, out, GemvKernel::Auto);
    }

    /// `y = W @ x` with an explicit kernel (tests use `Scalar` as the
    /// golden reference and `Avx2`/`Neon` for bit-exactness checks).
    pub fn forward_gemv_kernel(&self, x: &[f32], out: &mut [f32], kernel: GemvKernel) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        let (qx, x_scale) = Self::quantize_activation(x);
        if x_scale == 0.0 {
            out.fill(0.0);
            return;
        }
        self.forward_gemv_i8_kernel(&qx, x_scale, out, kernel);
    }

    /// `y = W @ x_i8 * x_scale`, with a pre-quantised INT8 activation.
    pub fn forward_gemv_i8(&self, qx: &[i8], x_scale: f32, out: &mut [f32]) {
        self.forward_gemv_i8_kernel(qx, x_scale, out, GemvKernel::Auto);
    }

    pub fn forward_gemv_i8_kernel(
        &self,
        qx: &[i8],
        x_scale: f32,
        out: &mut [f32],
        kernel: GemvKernel,
    ) {
        assert_eq!(qx.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        let use_avx2 = matches!(kernel, GemvKernel::Avx2)
            || (matches!(kernel, GemvKernel::Auto) && qx.len() >= 32 && avx2_available());
        // NEON is only reachable on aarch64; on x86_64 an explicit
        // Neon request falls back to scalar (documented, not silent).
        #[cfg(target_arch = "aarch64")]
        let use_neon = matches!(kernel, GemvKernel::Neon)
            || (matches!(kernel, GemvKernel::Auto)
                && std::arch::is_aarch64_feature_detected!("neon"));
        #[cfg(not(target_arch = "aarch64"))]
        let use_neon = false;

        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                let mut w_row = vec![0i8; self.cols];
                for r in 0..self.rows {
                    self.unpack_row(r, &mut w_row);
                    let acc = unsafe { dot_i8_avx2(&w_row, qx) };
                    out[r] = acc as f32 * x_scale * self.scales[r];
                }
                return;
            }
        }
        if use_neon {
            #[cfg(target_arch = "aarch64")]
            {
                let mut w_row = vec![0i8; self.cols];
                for r in 0..self.rows {
                    self.unpack_row(r, &mut w_row);
                    let acc = unsafe { dot_i8_neon(&w_row, qx) };
                    out[r] = acc as f32 * x_scale * self.scales[r];
                }
                return;
            }
        }
        // Scalar golden reference: decode the packed tiles directly.
        for r in 0..self.rows {
            let mut acc: i32 = 0;
            let out_block = r / 16;
            let row_in_tile = r % 16;
            for in_block in 0..self.in_blocks {
                let tile = self.tile_at(out_block * self.in_blocks + in_block);
                for j in 0..4 {
                    let c = in_block * 4 + j;
                    if c >= self.cols {
                        break;
                    }
                    let idx = row_in_tile * 4 + j;
                    let code = (tile[idx / 4] >> ((idx % 4) * 2)) & 0b11;
                    let w: i32 = match code {
                        0b01 => 1,
                        0b11 => -1,
                        _ => 0,
                    };
                    acc += w * qx[c] as i32;
                }
            }
            out[r] = acc as f32 * x_scale * self.scales[r];
        }
    }
}

/// Runtime AVX2 detection. The `is_x86_feature_detected!` macro may
/// only be expanded on x86/x86_64, so it lives behind a `cfg` gate here
/// (a `cfg!(...) && macro` expression would still expand — and fail —
/// on aarch64).
#[inline]
fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX2 integer dot product of two i8 slices.
///
/// The code -> sign table lookup is done with `_mm256_shuffle_epi8`
/// (`table = [0, 1, 0, -1]` as bytes `0, 1, 0, 0xFF`): each weight is
/// first expanded to its 2-bit code byte, then shuffled to its sign.
/// The dot product itself uses `_mm256_sign_epi8` (ternary weight
/// `-1/0/+1` selects negate / zero / keep) and accumulates in `i16`
/// for 128 columns before widening to `i32`, exactly as specified.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(w: &[i8], x: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    debug_assert_eq!(w.len(), x.len());
    let n = w.len();
    // Sign lookup table for shuffle: code 0->0, 1->+1, 2->0, 3->-1.
    let table = _mm256_setr_epi8(
        0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1,
        0, 1, 0, -1,
    );
    let ones16 = _mm256_set1_epi16(1);
    let mut acc32 = _mm256_setzero_si256();
    let mut acc16 = _mm256_setzero_si256();
    let mut since_widen = 0usize; // columns accumulated in i16
    let mut i = 0;
    while i + 32 <= n {
        // Expand the 32 ternary codes in w[i..i+32] to sign bytes via
        // the shuffle table. w holds -1/0/+1, so code = w & 3
        // ( -1 as u8 is 0xFF, 0xFF & 3 = 3 -> -1; 1 & 3 = 1; 0 -> 0 ).
        let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
        let codes = _mm256_and_si256(wv, _mm256_set1_epi8(3));
        let signs = _mm256_shuffle_epi8(table, codes);
        let xv = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
        // prod = xv * signs, via sign() (signs are already -1/0/+1, so
        // sign(xv, signs) == xv * signs).
        let prod = _mm256_sign_epi8(xv, signs);
        let lo = _mm256_castsi256_si128(prod);
        let hi = _mm256_extracti128_si256(prod, 1);
        let lo16 = _mm256_cvtepi8_epi16(lo);
        let hi16 = _mm256_cvtepi8_epi16(hi);
        acc16 = _mm256_add_epi16(acc16, lo16);
        acc16 = _mm256_add_epi16(acc16, hi16);
        since_widen += 32;
        if since_widen >= 128 {
            // Widen i16 -> i32 every 128 columns.
            acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(acc16, ones16));
            acc16 = _mm256_setzero_si256();
            since_widen = 0;
        }
        i += 32;
    }
    if since_widen > 0 {
        acc32 = _mm256_add_epi32(acc32, _mm256_madd_epi16(acc16, ones16));
    }
    let mut buf = [0i32; 8];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc32);
    let mut total: i32 = buf.iter().sum();
    while i < n {
        total += w[i] as i32 * x[i] as i32;
        i += 1;
    }
    total
}

/// NEON integer dot product (aarch64 only).
///
/// `vqtbl1q_u8` performs the code -> sign table lookup, then the
/// product is widened with `vaddl_s8` / `vaddl_high_s8` to `i16`
/// (two columns per step) before widening to `i32`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_neon(w: &[i8], x: &[i8]) -> i32 {
    use core::arch::aarch64::*;
    debug_assert_eq!(w.len(), x.len());
    let n = w.len();
    // Table: code 0->0, 1->1, 2->0, 3->0xFF (-1), repeated.
    let mut table = vdupq_n_u8(0);
    table = vsetq_lane_u8::<1>(1, table);
    table = vsetq_lane_u8::<3>(0xFF, table);
    let mut acc = vdupq_n_s32(0);
    let mut i = 0;
    while i + 16 <= n {
        let wv = vld1q_s8(w.as_ptr().add(i));
        let codes = vandq_s8(wv, vdupq_n_s8(3));
        let signs_u = vqtbl1q_u8(table, vreinterpretq_u8_s8(codes));
        let signs = vreinterpretq_s8_u8(signs_u);
        let xv = vld1q_s8(x.as_ptr().add(i));
        // prod = xv * signs: use sign semantics via vmul (signs are -1/0/1).
        let prod = vmulq_s8(xv, signs);
        // Widen i8 -> i16 (low and high halves), then -> i32.
        let prod_lo = vget_low_s8(prod);
        let prod_hi = vget_high_s8(prod);
        let lo16 = vaddl_s8(prod_lo, vdup_n_s8(0));
        let hi16 = vaddl_s8(prod_hi, vdup_n_s8(0));
        // Split each i16x8 into low/high i16x4 and widen to i32.
        let lo16_lo = vget_low_s16(lo16);
        let lo16_hi = vget_high_s16(lo16);
        let hi16_lo = vget_low_s16(hi16);
        let hi16_hi = vget_high_s16(hi16);
        acc = vaddq_s32(acc, vaddl_s16(lo16_lo, vdup_n_s16(0)));
        acc = vaddq_s32(acc, vaddl_s16(lo16_hi, vdup_n_s16(0)));
        acc = vaddq_s32(acc, vaddl_s16(hi16_lo, vdup_n_s16(0)));
        acc = vaddq_s32(acc, vaddl_s16(hi16_hi, vdup_n_s16(0)));
        i += 16;
    }
    let mut total: i32 = vaddvq_s32(acc);
    while i < n {
        total += w[i] as i32 * x[i] as i32;
        i += 1;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta_net::xorshift64star;

    fn rand_ternary(rows: usize, cols: usize, seed: u64) -> Vec<i8> {
        let mut st = seed ^ 0x9E37_79B9_7F4A_7C15;
        (0..rows * cols)
            .map(|_| (xorshift64star(&mut st) % 3) as i8 - 1)
            .collect()
    }

    fn rand_f32(n: usize, seed: u64) -> Vec<f32> {
        let mut st = seed ^ 0x1234_5678_9ABC_DEF0;
        (0..n)
            .map(|_| {
                let u = (xorshift64star(&mut st) >> 11) as f64 / (1u64 << 53) as f64;
                (u as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_simd_bitexact() {
        // DoD 1: non-16-aligned dims 37 x 523, AVX2/NEON vs scalar < 1e-5.
        let (rows, cols) = (37usize, 523usize);
        let w = rand_ternary(rows, cols, 7);
        let scales = vec![0.13f32; rows];
        let m = BitLinearWeights::from_ternary_matrix(&w, rows, cols, &scales);
        let x = rand_f32(cols, 8);
        let mut out_scalar = vec![0.0f32; rows];
        let mut out_auto = vec![0.0f32; rows];
        m.forward_gemv_kernel(&x, &mut out_scalar, GemvKernel::Scalar);
        m.forward_gemv_kernel(&x, &mut out_auto, GemvKernel::Auto);
        let max_err = out_scalar
            .iter()
            .zip(out_auto.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "SIMD vs scalar max err {max_err}");
        // Tile round-trip on the same non-aligned shape.
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(m.get(r, c), w[r * cols + c]);
            }
        }
        let tiles = m.tile_bytes();
        let m2 = BitLinearWeights::from_tile_bytes(&tiles, rows, cols, &scales);
        assert_eq!(m2.tile_bytes(), tiles);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let (rows, cols) = (20, 10);
        let w: Vec<i8> = (0..rows * cols)
            .map(|i| (((i * 7 + 3) % 5) as i8 % 3) - 1)
            .collect();
        let scales = vec![0.5; rows];
        let m = BitLinearWeights::from_ternary_matrix(&w, rows, cols, &scales);
        assert_eq!(m.num_tiles(), 2 * 3);
        assert_eq!(m.packed_bytes(), m.num_tiles().div_ceil(4) * 64);
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(m.get(r, c), w[r * cols + c]);
            }
        }
    }

    #[test]
    fn gemv_matches_reference() {
        let (rows, cols) = (20, 10);
        let w: Vec<i8> = (0..rows * cols)
            .map(|i| (((i * 7 + 3) % 5) as i8 % 3) - 1)
            .collect();
        let scales = vec![0.25; rows];
        let m = BitLinearWeights::from_ternary_matrix(&w, rows, cols, &scales);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 - 4.0) * 2.0).collect();
        let (qx, xs) = BitLinearWeights::quantize_activation(&x);
        assert!(qx.iter().all(|&v| (v as i32) >= -127 && (v as i32) <= 127));
        let mut out = vec![0.0; rows];
        m.forward_gemv(&x, &mut out);
        for r in 0..rows {
            let expected: i32 = (0..cols)
                .map(|c| w[r * cols + c] as i32 * qx[c] as i32)
                .sum();
            let expected_f = expected as f32 * xs * scales[r];
            assert!((out[r] - expected_f).abs() < 1e-4, "row {r}");
        }
    }

    #[test]
    fn gemv_zero_input_is_zero() {
        let w = rand_ternary(8, 8, 1);
        let m = BitLinearWeights::from_ternary_matrix(&w, 8, 8, &[1.0; 8]);
        let mut out = vec![9.0; 8];
        m.forward_gemv(&[0.0; 8], &mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_dot_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let w: Vec<i8> = (0..256).map(|i| (i % 3) as i8 - 1).collect();
        let x: Vec<i8> = (0..256).map(|i| ((i * 5) % 17) as i8 - 8).collect();
        let expected: i32 = w
            .iter()
            .zip(x.iter())
            .map(|(&a, &b)| a as i32 * b as i32)
            .sum();
        let got = unsafe { dot_i8_avx2(&w, &x) };
        assert_eq!(got, expected);
    }
}
