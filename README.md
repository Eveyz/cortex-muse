# Cortex-Local

轻量级个人持续学习框架：1.6B 冻结三值认知内核 + 线性状态空间 +
64M 在线可塑海马体。纯标准库 Rust 核心（无 tch/candle/ort），
无全局反向传播——在线学习用解析外积梯度，夜间微调用 MeZO
零阶前向扰动，训练内存峰值 = 推理内存峰值。

**先说清楚定位**：优势不在“比大模型更聪明”——参数、数据、
SFT、RL 的 scaling 定律对谁都一样，1.6B 打不过 7B。优势在
三处：① 三值权重让模型只有约 320MB，跑在任何笔记本 CPU 上，
不用 GPU、不用云、数据不出设备；② 冻结新皮层 + 可塑海马体，
能在使用中持续学习且结构上不灾难性遗忘（这是普通 LLM + RAG
做不到的）；③ DeltaNet 状态 O(1)，万轮对话内存不涨。
如果海马体这条赌注不成立，相对普通 LLM 就没有优势可言。

## 目录

```
cortex-local/
├── Cargo.toml                      # workspace：5 个 crate
├── crates/
│   ├── cortex-kernels/             # Phase 1：Tile16x4/CacheLine16x16、
│   │   └── src/                    #   AVX2/NEON/scalar GEMV、RMSNorm、
│   │       ├── bit158.rs           #   SwiGLU、.ctxb 读写 + CRC32
│   │       ├── delta_net.rs        #   Gated DeltaNet 层/堆叠（chunkwise）
│   │       ├── ops.rs              #   rms_norm / silu / BitSwiGLU
│   │       └── ctxb.rs             #   .ctxb v1 格式（见下）
│   ├── cortex-ssm/                 # Phase 2：BitDeltaNetLayer + INT8
│   │   └── src/                    #   环形滑动窗口注意力（W=512）
│   │       ├── delta.rs            #   包装 kernels 的 DeltaNet，O(1) 状态
│   │       └── ring.rs             #   RingSlidingAttention
│   ├── cortex-hippocampus/         # Phase 3：无锁双缓冲在线记忆 MLP
│   │   └── src/                    #   + 惊奇度门控 + WAL + 版本快照
│   │       ├── lib.rs              #   HippocampusWeights / PlasticityWorker /
│   │       ├── wal.rs              #   SharedWeightArena / CortexMemoryLayer
│   │       └── snapshot.rs         #   WAL 二进制日志 / "HIP1" 快照
│   ├── cortex-sleep/               # Phase 4：MeZO 零阶微调 + 重放 + 正交化
│   │   └── src/
│   │       ├── mezo.rs             #   MezoScaleTuner（3 次前向，z 不存）
│   │       ├── replay.rs           #   ReplayBuffer（80% WAL + 20% 锚点）
│   │       └── orthogonalize.rs    #   Gram-Schmidt 去重 / 软奇异值抑制
│   └── cortex-runtime/             # Phase 5：完整模型 + 分词器 + FSM + CLI
│       └── src/
│           ├── tokenizer.rs        #   Byte-BPE（256 字节基 + 可选合并规则）
│           ├── fsm.rs              #   昼夜四态状态机
│           ├── model.rs            #   CortexModel（N_MACROS 宏周期）
│           ├── tools.rs            #   Tool calling：迷你 JSON 解析、工具注册表、
│           │                       #   5 个内置工具、AgentLoop（见“Tool calling”）
│           └── bin/cortex-cli.rs   #   cortex-cli（/status /sleep /reset-mem /agent /tools）
├── tools/
│   ├── train_seed_qat.py           # Phase 6：PyTorch QAT + .ctxb 导出 + 自检
│   ├── test_ctxb_packing.py        #   无 torch 交叉校验：Python 打包 vs Rust
│   ├── distill_generate.py         # 蒸馏：Qwen3 老师生成训练数据（vLLM），含 tool-use 类别
│   └── distill_train.py            # 蒸馏：学生 QAT 训练循环（见“训练路线”）
└── crates/cortex-runtime/examples/bench_forward.rs   # 前向 benchmark
```

## 快速开始

```bash
export PATH="$HOME/.cargo/bin:$PATH"   # Rust 1.100，经 rustup 安装
cargo test --workspace                  # 51 个测试，全部通过（约 2 分钟）
cargo clippy --workspace --all-targets   # 零警告
cargo fmt --all
cargo build --release
cargo run --release -p cortex-runtime --bin cortex-cli -- \
  --prompt "Hello, Cortex" --tokens 32 --non-interactive
cargo run --release -p cortex-runtime --bin cortex-cli   # 交互 REPL
```

REPL 命令：`/status`（内存与海马体版本）、`/sleep`（手动一次
睡眠巩固：MeZO 5 步 + 海马体正交化/快照到
`temp_dir/cortex-snapshots`）、`/reset-mem`（回滚到倒数第二个
快照，需 ≥2 个快照）、`/agent <请求>`（带工具的 agent 回合，
见下）、`/tools`（列出可用工具）、`/help`、`/quit`。
`--check-ctxb`：加载模型、关海马体、逐行打印 prompt logits
（供 Python 自检用）。

## Tool calling + harness

模型本身只会生成文本，动手靠 `crates/cortex-runtime/src/tools.rs`：

- **5 个内置工具**：`calculator`（+ - * / % 括号）、`read_file`、
  `write_file`、`list_dir`、`get_time`（UTC）。没有 shell 工具。
- **格式**：Qwen ChatML 风格，模型输出
  `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`，
  harness 执行后把真实结果以
  `<tool_result name="...">...</tool_result>` 喂回去，再生成。
  模型在工具调用后幻觉出来的假结果会被截掉，只进真实结果。
- **循环**：生成 → 解析 → 执行 → 回填，直到模型直接回答
  或达到 `--max-steps`（默认 5 轮）。解析失败、参数缺失、
  工具报错都会当作一条 tool result 喂回去，让模型自我纠正。
- **安全**：文件工具限制在 workspace 内（默认 `./cortex-work`，
  不存在则新建）；绝对路径、`..` 越狱、指向外部的符号链接
  都会被拒。agent 生成用贪心采样（temperature 0），每步至少
  128 token（工具调用需要篇幅）。
- **行为从哪来**：蒸馏数据的第 7 类 `tooluse`
 （`distill_generate.py`）：老师生成带工具调用的完整 trace，
  脚本逐条真实执行、把真结果替换掉老师幻觉的结果，终答必须
  引用真结果才保留。工具集、system prompt、trace 格式与 Rust
  侧逐字共享，`test_render_system_prompt_exact` 把 prompt 文本
  钉死，两边改了一边没改另一边测试就会失败。

```bash
cargo run --release -p cortex-runtime --bin cortex-cli -- \
  --agent "帮我算一下 12*3+5，再把结果写进 notes/total.txt" \
  --workspace ./cortex-work --max-steps 5
# REPL 里：/agent <请求>，/tools 看工具列表
```

**如实说明**：toy 模型（随机权重）现在不会调工具——上面
的命令跑完只会得到乱码回答。这是预期的：调用行为来自蒸馏，
没训过就没有。harness 本身已用 scripted 生成器测过
（工具往返、幻觉结果截断、步数上限，见 `tools.rs` 测试）。

推理硬件：纯 CPU，无 CUDA 路径；模型约 320MB（三值主干）+
256MB（海马体 FP32），8GB 内存够用，16GB 舒适；需要 AVX2
（x86）或 NEON（ARM）。

## 训练路线：蒸馏（已定，替代 80GB 从零预训练）

原计划 20B tokens（约 80GB）从零预训练已废弃：老师已经在
几十 T tokens 上做完了昂贵的“发现规律”，学生只学老师的输出，
数据量少 10–50 倍且质量更高。实证：DeepSeek-R1-Distill-Qwen-1.5B
只用 80 万条老师样本，AIME 2024 拿了 28.9%（GPT-4o 9.3%），
MATH-500 83.9%（GPT-4o 74.6%）。

老师：Qwen3-30B-A3B-Instruct-2507（默认，MoE，每 token 只激活
3B）或 Qwen3-32B。**Qwen3 没有 28B 这个尺寸**。全部 Apache 2.0，
蒸馏合法。老师跑在 1 张 H100 80GB 上（BF16 约 60–64GB）+ vLLM。

```bash
pip install vllm
python tools/distill_generate.py --out distill-data.jsonl --limit 20  # 冒烟
python tools/distill_generate.py --out distill-data.jsonl             # 全量，默认约 12.5k 条
python tools/distill_train.py --data distill-data.jsonl --config production \
  --epochs 2 --out student.ctxb --ckpt-dir ckpts
```

七类数据（默认量）：数学 CoT 3000、代码+单测 2000、教材讲解 3000、
指令跟随 2000、双语对话 1500、`<memory_slot>` 记忆问答 1000、
tool-use 调用 trace 2000（见“Tool calling + harness”）。
**验证是核心**：数学种子题对已知答案精确比对，老师自编题要
两次独立求解一致才留；代码真的在隔离临时目录里跑单测，跑通
才留；其余按长度 + 去重过滤。每条记录带 `verified` 字段，
生成可断点续跑。

两个硬约束（不要试图绕过）：
- **只能序列级蒸馏**：老师是 Transformer、词表 15 万，学生是
  DeltaNet SSM、词表 3.2 万——logit 蒸馏需要匹配的输出空间，
  不存在；权重更没法跨架构搬。文本是唯一的蒸馏通道。
- **三值约束还在**：学生必须用 QAT（STE）训出来，`train_seed_qat.py`
  就是这套机器；`distill_train.py` 在此基础上做 completion-only
  loss 的 SFT 蒸馏（loss 只算回答部分），最后导出 `.ctxb`。

预期：明显好于从零训 20B，数学/代码上有望接近 Qwen2.5-1.5B 级别；
仍需 SFT 后的评估（GSM8K / HumanEval / MMLU）验证，GRPO 可选。

## Phase 6：PyTorch QAT 脚本（状态）

`tools/train_seed_qat.py`（约 29KB，`py_compile` 通过）：
`BitLinearQAT`（STE：`w_q = round(w/mean|w|).clamp(-1,1)`，
`(w_q − w).detach() + w`；激活每 token 量化到 [−127,127] 同样 STE；
前向缩放用每行 mean-|w|，与 Rust 存储/导出一致）、torch 版
DeltaNet（同 delta 规则）、环形注意力（校验路径 INT8 KV）、
`HippocampusMLP`（`learn_pair` 镜像 Rust 解析规则，写操作不用
autograd；第 4 阶段接口训练 = 序列前半写海马体、后半读海马体，
LM loss 与海马体 detached）、权重绑定三值 Embedding/LM Head、
`train_seed`（AdamW，默认 50 步自检）、`export_to_ctxb`
（张量名/顺序与 Rust `to_ctxb_tensors` 一致，含 log_decay；
`pack_tiles` 实现 Tile16x4 位布局）、`run_self_check`
（两侧海马体全关，`cortex-cli --check-ctxb` 对 logits，容差 1e-3）。
**舍入对齐**：全部用 `round_half_away`（Rust `f32::round` 是
half-away-from-zero，torch/numpy 是 half-to-even）。

- 打包交叉校验已通过（`tools/test_ctxb_packing.py`，无需 torch）：
  5 种形状逐字节一致。
- torch 2.15.0+cpu 已装在 `~/.venvs/torch-cpu`（注意该 venv
  没有 numpy，`train_seed_qat.py` 需要 torch + numpy）。
- 完整自检**尚未跑**：本机 VM 太低配（4 核共享、7.7GB 内存、
  无 GPU），在用户自己的机器上跑：
  ```
  cargo build --release
  python tools/train_seed_qat.py --self-check --rust-bin ./target/release/cortex-cli
  ```
  期望输出 `self-check: max |logit diff| = ... (tolerance 1e-3)`。
  若失败，最可能的嫌疑（按顺序）：某侧海马体没关掉、激活舍入、
  环形 INT8 反量化顺序、log_decay 没加载。

## 实测（2026-11-21，x86_64，rustc 1.100）

| 项目 | 结果 |
|---|---|
| `cargo test --workspace` | **51/51 通过**（kernels 16、ssm 5、hippocampus 5、sleep 5、runtime lib 18（含 tools 10）、CLI 集成 1、MeZO 下降 1） |
| `cargo clippy --workspace --all-targets` | 0 警告 |
| MeZO DoD 测试（debug） | 89 秒（优化后 unpack_row，之前 208 秒） |
| 前向速度（micro 配置，release） | **11.56 ms/token ≈ 87 tok/s**（200 token，hippo 学习关） |
| CLI 端到端生成 32 token | 0.66 秒（含启动） |
| `.ctxb` 往返 logits 误差 | < 1e-5 |
| Python 打包 vs Rust 打包 | 5 种形状（含 37×523 非对齐）逐字节一致 |

**生产配置警告**：micro 配置 `d_model=256`，生产 `d_model=2048`
时每 token 约 64 倍计算量，当前核（逐行解包 + 点积）下
80+ tok/s 的目标只对 micro 配置实测成立。真正的修复是
融合 AVX2 tile GEMV（查表展开 + 点积，不物化行）——见“下一步”。

## 架构要点（已定案，不再重议）

- **宏周期**：每宏 = 3× BitSSM 层（DeltaNet + 1.58-bit SwiGLU）
  + 1× 记忆层（INT8 环形注意力分支 A + 海马体分支 B 串联 + FFN）。
  micro：`N_MACROS=2`（8 层，d=256）；生产：`N_MACROS=6`（24 层，d=2048）。
  词表层权重绑定（输入 Embedding 与输出 LM Head 共用一张
  INT8 三值矩阵 + 每行 FP32 缩放因子）。
- **转置约定**（`cortex-ssm/src/delta.rs` 有完整说明）：规范本身
  自相矛盾——`u = S_{t-1} k_t` 要求 `S:[dv,dk]`（此时 `o = S q`），
  而 `o_t = S_tᵀ q_t` 要求 `S:[dk,dv]`。实现按分解式字面执行
  （`S:[dv,dk]`，`o = S @ q`），因为只有这种形式的单次写入
  回忆可验证（β=1 写 k→v，查 q=k 精确返回 v）。不要只改转置，
  那会同时破坏 `u` 与秩-1 更新。
- **DeltaNet 单步**：`u = S k`，`S_t = α S + β(v − αu) kᵀ`，
  `O(d_head²)` 而非矩阵乘法的 `O(d_head³)`；chunkwise 用下三角
  系统前向代入，门控用累积衰减归约，`Γ < 1e-3` 回退顺序单步。
- **海马体**：SiLU 双层 MLP，W2 缩小 10 倍保冷启动；
  `SharedWeightArena` 双槽 + 读者计数器（ABA/悬垂的解法）；
  惊奇度三级门控（默认低 0.02 / 高 0.25，base_lr 0.08，
  boost 2.5，动量 0.85，衰减 1e-4，每 4 个高惊奇 token 提交一次）。
  高惊奇样本连同 token 上下文异步写 WAL（供睡眠重放）；
  快照为 `"HIP1"` 格式 + CRC，`hippo-v{version}.snap`，支持回滚。
- **.ctxb v1**：Magic `"CTXB"`，64 字节头，CRC32 IEEE（已知向量
  `crc32("123456789")==0xCBF43926` 已验证），自描述张量
  （kind 0 BitLinear / kind 1 F32Vector）。头保留字节 44..52 存
  `ffn_hidden`/`hippo_hidden`，0 表示按 `2*d_model` 推导。
  **`.ctxb` 只存冻结种子内核**：海马体权重不进文件（在线记忆），
  往返测试需用 `replace_weights` 对齐两侧海马体并关学习；
  每个 SSM 块的 `log_decay` 向量必须随文件走
  （`macro{m}.ssm{s}.delta.log_decay`），缺了它往返误差 0.018。
- **分词器**：Byte-BPE，256 字节基 + 可选合并规则文件
  （`left_id right_id merged_id`，`#` 注释）；生产词表 32768。
  注意：Python 训练侧目前是纯字节级编码（无合并规则），
  生产词表大部分行训不到——32K BPE 合并规则仍待训练，
  见“下一步”。

## 必须如实说明的三处修正

### 1. 海马体单次绑定测试（沿用上一轮结论）

原草案测试（6 步、lr 0.35、断言 MSE 降到 15% 以下）按原样
**跑不通过**：实测 MSE 仅 1.108 → 1.092。原因不是接线错：
隐藏激活 `‖h1‖≈0.2` 且误差梯度按 `1/dim` 归一化，6 步的权重
移动量级只有 `~1e-4`。Python 按同一算法复现
（`/tmp/sim_hippo.py`，Rust 实测与仿真完全一致）：

| 配置 | MSE 后 / 前 |
|---|---|
| 6 步，lr 0.35（原草案） | 0.986（不通过） |
| 6 步，lr 5.0 | 0.033 |
| 20 步，lr 0.35 | 0.411 |
| **60 步，lr 0.35** | **0.002** |

处理：**生产学习率不动，测试改为 60 次曝光**。真正的单次曝光
绑定仍是开放决策（提高 `base_lr` 或放大 W2 初始尺度），
不要悄悄改梯度的 `1/dim` 归一化来凑测试通过。
补充测试：直接 worker、dim 64 / hidden 128、lr 5.0、**12 步**
MSE 降到 9.4%（仿真 `/tmp/sim2.py` 与 Rust 完全一致）。

### 2. MeZO 在玩具模型上的实测结论（最重要的一条）

规范的 MeZO 公式本身已按原样实现（`mezo.rs`：种子再生三次，
z 永不存储，`g_proj=(L+−L−)/(2ε)`，每步恰好两次 loss 调用；
sleep crate 内二次型测试 30 步 loss 降到一半以下，通过）。
但在**随机 toy 的 8 层完整模型**上，规范字面配置（全 φ 24,116 维、
单方向、lr 0.2）会把模型打崩到均匀 logits
（loss 恰好 = ln 260 = 5.5607）。实测 lr 扫描（release）：

| 配置 | 30 步后 loss 比率 |
|---|---|
| 全 φ，lr 0.001 | 0.971（稳定但极慢） |
| 全 φ，lr 0.004 | 3.84（发散） |
| 全 φ，lr 0.01 | 0.984 后塌到均匀 |
| 全 φ，lr 0.2 | 爆炸（loss 103 → 111,813）后塌到均匀 |
| LM 子空间（516 维），K=4 平均方向，eps 0.002，lr 0.002，信任域 5% | **0.793（5.2443 → 4.1596）** |

根因：更新量级 `lr·|g_proj|`（g 为 O(1–10)）相对 φ 量级 ~0.03
太大；单方向在 24k 维中只捕获约 `1/√dim` 的梯度。
**DoD 测试与 CLI `/sleep` 采用最后一行配置**：重复模式序列
（`aaaaaaaa` / `bbbbbbbb` / `abababab` / `babababa`，尺度调参
能真正学会的模式任务）+ 输出尺度子空间
（tied embedding scales + final RMS，`collect_phi_lm` /
`apply_phi_lm`）+ `step_multi` 每步 4 个平均扰动方向 +
每元素步长限幅 `5%·|φ_i|`（`with_trust_region`，opt-in，
默认保持规范公式不变）。测试断言终值 < 0.9×初值，实测 0.793。

### 3. NEON 未实测

NEON 路径（`vqtbl1q_u8` + `vaddl_s8`）可为 aarch64 编译通过，
但本机为 x86_64，**运行时正确性未验证**。不要声称它已验证。

## 路线图：持续进化与模型家族

光靠 L1 记喜好，模型只是高级记事本。真正的进化分三层，
参数会按需增加，局部推理能力也会实打实上涨。

### 持续进化的三层

- **L1 默契层（已建好）**：参数不变，改海马体权重
  （在线外积更新）。记住代码库结构、私人术语、写作口吻，
  省掉每次写几千字背景 Prompt 的麻烦。不提高原生智商，
  但减少因“不了解背景”导致的犯蠢。
- **L2 技能树生长层（v0.4+，设计阶段）**：参数真的会增加。
  主干预留稀疏路由插槽；当睡眠引擎发现你连续两周都在做
  某一类任务（且 WAL 里积累了几千条高惊奇纠错记录），夜间
  自动新建并训练一个 20–50 MB 的 1.58-bit 领域微专家
  adapter 挂上去。总参数从 7B 慢慢长到 7.5B、8B……，但每次
  对话路由器只激活相关的 1 个专家，内存和速度几乎不变。
  前提：睡眠现在没有 autograd，训 adapter 需要新增小规模
  反向传播（或用 MeZO 训，慢得多），路由器本身也要训。
- **L3 智商跃迁层（v0.4，设计已定稿）**：慢思考试错 → 快直觉蒸馏
  （STaR 式自蒸馏）。本地模型靠外部验证器（编译器、解释器、
  单测）试错跑通后，睡眠期提炼成黄金训练对：带修复的推导链
  （保留“报错 → 诊断 → 修正”的关键转折，压缩成一句话诊断，
  不进训练文本的只有成功推导）。两条线并行：A 线每晚本地
  （海马体 anchor + ReplayBuffer 黄金池 + MeZO 调 γ），B 线
  攒够 500 条后离线 SFT 出新 `.ctxb`，评估门通过才上线。
  完整设计见 `docs/l3-self-distillation.md`（2026-11-21）。

### 模型家族

| 版本 | 参数 | 1.58-bit 内存 | 适用硬件 | 定位 |
|---|---|---|---|---|
| Cortex-Nano | 1.6B | ~380 MB | 手机 / 树莓派 / 8GB 笔记本常驻 | 意图路由、日常问答、单步工具调用 |
| Cortex-Pro | 7–8B | ~1.7 GB | 16GB 轻薄本 | 主力：完整模块代码、多步 Agent、复杂指令 |
| Cortex-Max | 28B-A4B MoE | ~6.5 GB 磁盘，每次激活 ~1 GB | 16/32GB 台式机 / Mac | 本地天花板：总参 28B，每词只激活 4B |

如实说明：Pro 的“纯 CPU 40+ tok/s”需要融合 AVX2 tile
GEMV（v0.3 TODO）+ 高端笔记本 CPU，不是开箱即得——当前
内核下 production 1.6B 只有约 1–2 tok/s（见实测）。Max
的 CPU 速度也应按个位数到十几 tok/s 估计，不是“飞快”。
28B-A4B MoE 需要新增 MoE 路由，排在 7B 之后。

### 换血蒸馏：怎么做大杯（不从零预训练）

老师：Qwen3 / Qwen3.5、Gemma 4、DeepSeek 系列
（Apache 2.0 / MIT 优先）。只花从零训练 1%–2% 的算力：

1. **注意力层换血为线性层**（借鉴 MOHAWK 协议）：
   保留老师 FFN 权重做热启动初始化，把每 4 层里的前 3 层
   Attention 替换成 Gated DeltaNet；冻结其余层，喂长文本，
   让新层的隐藏向量拟合老师 Attention 层的输出（MSE）。
2. **渐进式 1.58-bit 压制**：温度系数 λ 从 0.0 线性过渡
   到 1.0，再做序列级 SFT。**注意：logit KL 蒸馏因词表不同
   不可直接算**——老师 Qwen3 是 151K 词表，学生是 32K，
   同一个 token 位置的概率分布对不上。如需分布级蒸馏，
   要么学生换用老师的 tokenizer，要么只做隐藏态匹配。
3. **挂载海马体联合微调**：冻结三值主干，只训第 4 层
   记忆读写投影门控，跑几小时即可。

### 数据集清单（Hugging Face，全部开源）

- **蒸馏底座**：`HuggingFaceFW/fineweb-edu`（score ≥ 4）、
  `HuggingFaceTB/cosmopedia`、`opencsg/chinese-fineweb-edu`、
  `BAAI/IndustryCorpus2`、`open-thoughts/OpenThoughts3-1.2M`
  （85 万数学 + 25 万代码 + 10 万科学，QwQ-32B 标注）。
- **指令跟随**：`allenai/tulu-3-sft-mixture`、
  `BAAI/Infinity-Instruct`、`Magpie-Align/Magpie-Pro-300K-Filtered`。
- **工具调用**：`Salesforce/xlam-function-calling-60k`、
  `NousResearch/hermes-function-calling-v1`、`Team-ACE/ToolACE`、
  `Nanbeige/ToolMind`（36 万条，带推理轨迹）、
  `FunReason-MT`（多步工具依赖）。

### 指令跟随与 Tool Calling 的训练工艺

- **特殊控制 Token（待决策）**：现在用纯文本 `<tool_call>`
  标签（字节序列，不用改词表）。专用 token
  （`<|im_start|>`、`<|tool_call|>`）边界更无歧义、更可靠，
  但要改词表（加 special token + embedding 行）+ 训练格式。
- **Prompt loss masking（已有）**：`distill_train.py` 已是
  completion-only loss，prompt 部分 label 为 -100。
- **负约束对比强化（待做）**：DPO / SimPO，Chosen（严格
  只吐 JSON）vs Rejected（多嘴寒暄 / 超字数），治“大体听懂
  但漏掉细节约束”。
- **工具调用“四大金刚”场景（已补齐，2026-11-21）**：
  成功调用 50%、报错自愈 20%（参数错 / 文件不存在 /
  权限拒绝后换工具或改参数重试）、缺参反问 15%（必填参数
  缺失时停下追问用户，禁止瞎编）、无工具闲聊 15%（挂着工具
  也一个不调，直接说话，防“锤子综合症”）。`gen_tooluse`
  按 50/20/15/15 分流；报错轨迹要求首调必须失败、后调成功，
  失败结果带 Rust harness 同款 `error: ` 前缀，且沙箱临时
  路径会被洗掉，不进训练数据。`tools/test_tooluse_generators.py`
  用 fake teacher 测验证逻辑，9/9 通过。
- **可验证强化学习（v0.5）**：GRPO / RLVR，无需人工打分，
  奖励脚本硬性裁判：JSON 可解析 +1、函数名与 required
  字段全对 +2、沙盒跑出正确结果 +5。
- **Rust 运行时物理保险（v0.5）**：受限解码 FSM——模型吐出
  `<|tool_call|>` 后，采样器按 JSON 语法状态机掩码词表，
  不合语法的 token 概率置为负无穷，保证 100% 合法调用。

### 下一步（按顺序）

1. **定专用控制 token 的决策**（纯文本 `<tool_call>` 标签
   vs 专用 token；tool-use 三大负样本已补，见上）。
2. **跑蒸馏**：在 GPU 机器上跑 `distill_generate.py`（冒烟 →
   全量，含 tool-use 类别）再跑 `distill_train.py`，GSM8K /
   HumanEval / MMLU 评估蒸馏学生，tool-use 用 held-out 任务按
   函数名 + 参数 + 终答精确匹配打分（BFCL 风格），确认效果后
   再决定是否加 GRPO。
3. **torch 自检收尾**（见 Phase 6 状态）：用户机器上的
   `--self-check` 结果，确认最大 |logit 差| < 1e-3。
4. **融合 AVX2 tile GEMV**：查表展开 + 点积，不物化行——
   当前 unpack+dot 是前向瓶颈（约 58 次 GEMV/token），
   也是生产配置 80+ tok/s 的前提。
5. **睡眠重放接线**：`/sleep` 目前只调 LM 子空间 + 正交化，
   WAL 重放（80% 当日高惊奇 + 20% 锚点）尚未接进 CLI。
6. **32K BPE 合并规则**：训练生产词表的合并规则，否则
   Python 训练侧与 Rust 推理侧的词表利用率不对齐。
7. **开放决策**：真正的单次曝光海马体绑定（提高 `base_lr`
   或放大 W2 初始尺度），由用户定，不要悄悄改。
8. **v0.4**：L3 自蒸馏管线（设计已定稿，见
   `docs/l3-self-distillation.md`：Phase L3-1 采集 + 验证器骨架 →
   L3-2 提取 + A 线 → L3-3 code 验证器 → L3-4 B 线）+
   L2 微专家设计（adapter 训练 + 路由器）。
9. **v0.5**：GRPO / RLVR + 受限解码 FSM。
10. **多版本**：7B 等 1.6B 验证后再定（原决策不变）；
    28B-A4B MoE 排最后（需新增 MoE 路由）。

## License

MIT OR Apache-2.0（见 `Cargo.toml`）。
