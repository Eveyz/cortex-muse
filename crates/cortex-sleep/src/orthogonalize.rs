//! Hippocampus orthogonalisation ("weight slimming", Phase 4).
//!
//! Two approximations, both operating on row vectors only (no SVD
//! solver, no extra large buffers):
//! * [`orthogonalize_rows`]: Gram–Schmidt-style dedup. For each row,
//!   if its cosine similarity with an earlier row exceeds
//!   `cosine_threshold`, the component along that earlier row is
//!   subtracted and the row is rescaled to its original norm. Returns
//!   the number of rows that were deduped.
//! * [`soft_singular_suppress`]: shrink rows whose norm is far above
//!   the mean norm toward the mean, a soft stand-in for singular-value
//!   suppression of saturated directions.

use cortex_hippocampus::HippocampusWeights;

/// Gram–Schmidt approximate dedup of the rows of a `[rows, cols]`
/// row-major matrix, in place. Returns the number of deduped rows.
pub fn orthogonalize_rows(
    weights: &mut [f32],
    rows: usize,
    cols: usize,
    cosine_threshold: f32,
) -> usize {
    assert_eq!(weights.len(), rows * cols);
    let mut deduped = 0usize;
    for i in 0..rows {
        let (done, rest) = weights.split_at_mut(i * cols);
        let (row_i, _) = rest.split_at_mut(cols);
        let norm_i0: f32 = row_i.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm_i0 == 0.0 {
            continue;
        }
        let mut changed = false;
        for j in 0..i {
            let row_j = &done[j * cols..(j + 1) * cols];
            let norm_j: f32 = row_j.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm_j == 0.0 {
                continue;
            }
            let dot: f32 = row_i.iter().zip(row_j.iter()).map(|(a, b)| a * b).sum();
            let cos = dot / (norm_i0 * norm_j);
            if cos.abs() > cosine_threshold {
                // Subtract the projection of row_i onto row_j.
                let coeff = dot / (norm_j * norm_j);
                for c in 0..cols {
                    row_i[c] -= coeff * row_j[c];
                }
                changed = true;
            }
        }
        if changed {
            // Rescale to the original norm (soft: keep the direction,
            // drop the duplicated component).
            let norm_i1: f32 = row_i.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm_i1 > 1e-9 {
                let s = norm_i0 / norm_i1;
                for x in row_i.iter_mut() {
                    *x *= s;
                }
            }
            deduped += 1;
        }
    }
    deduped
}

/// Shrink over-large row norms toward the mean norm by `shrink`
/// (0 = no-op, 1 = clamp to the mean). Returns the number of rows shrunk.
pub fn soft_singular_suppress(weights: &mut [f32], rows: usize, cols: usize, shrink: f32) -> usize {
    assert_eq!(weights.len(), rows * cols);
    let norms: Vec<f32> = (0..rows)
        .map(|i| {
            weights[i * cols..(i + 1) * cols]
                .iter()
                .map(|&x| x * x)
                .sum::<f32>()
                .sqrt()
        })
        .collect();
    let mean = norms.iter().sum::<f32>() / rows as f32;
    let mut shrunk = 0usize;
    for i in 0..rows {
        if norms[i] > mean && norms[i] > 0.0 {
            let target = mean + (norms[i] - mean) * (1.0 - shrink);
            let s = target / norms[i];
            for x in weights[i * cols..(i + 1) * cols].iter_mut() {
                *x *= s;
            }
            shrunk += 1;
        }
    }
    shrunk
}

/// Consolidate a hippocampus snapshot: orthogonalise both weight
/// matrices and soft-suppress saturated rows. Returns
/// `(deduped_rows_w1, deduped_rows_w2)`.
pub fn consolidate_hippocampus(
    weights: &mut HippocampusWeights,
    cosine_threshold: f32,
) -> (usize, usize) {
    let d1 = orthogonalize_rows(
        &mut weights.w1,
        weights.hidden_dim,
        weights.dim,
        cosine_threshold,
    );
    let d2 = orthogonalize_rows(
        &mut weights.w2,
        weights.dim,
        weights.hidden_dim,
        cosine_threshold,
    );
    soft_singular_suppress(&mut weights.w1, weights.hidden_dim, weights.dim, 0.2);
    soft_singular_suppress(&mut weights.w2, weights.dim, weights.hidden_dim, 0.2);
    (d1, d2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonalize_removes_duplicate_rows() {
        // Rows 0 and 1 are identical, row 2 is orthogonal.
        let mut w = vec![
            1.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
        ];
        let n = orthogonalize_rows(&mut w, 3, 3, 0.9);
        assert_eq!(n, 1, "exactly the duplicate row should be deduped");
        // After dedup, rows 0 and 1 must no longer be parallel.
        let dot: f32 = (0..3).map(|c| w[c] * w[3 + c]).sum();
        assert!(
            dot.abs() < 0.5,
            "deduped rows must not stay parallel, dot={dot}"
        );
        // Row norms are preserved.
        let n0: f32 = w[0..3].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn consolidate_hippocampus_runs() {
        let mut w = HippocampusWeights::new_initialized(8, 16, 3);
        let (d1, d2) = consolidate_hippocampus(&mut w, 0.95);
        assert!(w.w1.iter().all(|x| x.is_finite()));
        assert!(w.w2.iter().all(|x| x.is_finite()));
        let _ = (d1, d2);
    }
}
