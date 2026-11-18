# 清洗流水线

本文件描述 `embodied clean` / `embodied convert` 的两阶段设计、真实数据结论、产物契约和校验方式。
架构总览见 [architecture.md](architecture.md)，配置字段见 [configuration.md](configuration.md)。

## 1. 两阶段设计

1. **clean**：原始数据集 → 清洗层
   （清洗后 parquet + 每集切好的视频 + selection/manifest + QC 报告）
2. **convert**：清洗层 → 标准 LeRobot v3 数据集
   （重新编号、重编码视频、重算 stats/metadata，可被 LeRobot 训练入口直接读取；本项目为 HIL-SERL）

两个命令相互独立，中间层是显式契约，可分别重跑、审计和复用。

> 说明：仓库根目录的历史 `note.md` 内容已并入 [operations.md](operations.md)。
> 数据集 repo id 从采集/训练配置读取为 `lilkm/pick_cube_franka_panda_30`。
> 本文所有结果均基于运行时实际加载到的源内容。

## 2. 实际数据结论（2026-11-17 本地录制）

主配置指向这次录制产生的本地数据集：

`~/.cache/huggingface/lerobot/lilkm/pick_cube_franka_panda_30`

- 30 episodes / **666 frames** / 10 fps / 2 路相机（front、wrist，128×128，AV1）。
- **ep19 是明确脏数据**：90 帧（中位数约 20 帧），出现非法 `gripper=0`、23 个静止帧、
  最长 22 帧连续停滞、多次重抓；规则同时命中
  `duration_outlier + action_domain + interior_idle_ratio + interior_idle_run` → **drop**。
- **ep0 开头 15 帧空转**：操作量全为 0，之后是完整成功的抓取；规则判 `trim`，
  裁掉 15 帧保留 19 帧。
- 其余 28 条保持原样（每条开头有 4–7 帧正常启动静止，低于 `min_trim_frames=10`，不裁）。
- 结果：**29 episodes / 561 frames / 58 个 clip**，转换后回读校验 PASS。

> HF Hub 上的 `lilkm/pick_cube_franka_panda_30` 当前是 **2026-05-09 的旧快照（458 帧）**；
> 采集配置里 `push_to_hub: false`，说明这次新录制很可能没有真正上传。
> 流水线以实际加载到的源内容指纹为准：源变了，run key 和报告都会变。
> 想清洗 Hub 快照时，把 `source.local_root` 设为 `null` 即可；该路径也已验证可跑通
> （30/436，ep0 裁 22 帧）。

## 3. 本次运行产物

主配置（本地 666 帧录制）的 run key 为 `d163e86dc4aa__f3795cbc51b5`：

- 清洗层：`data/cleaned/d163e86dc4aa__f3795cbc51b5/`
- QC 报告：`data/reports/d163e86dc4aa__f3795cbc51b5/report.md`
- 最终数据集：`data/curated/d163e86dc4aa__f3795cbc51b5/dataset/`
- 回读校验：`data/curated/d163e86dc4aa__f3795cbc51b5/validation.json`

此前用 HF 旧快照验证加载路径时生成的 run key 是 `e91af696cab0__8b51888d1186`（30/436），
对应产物仍保留在 `data/cleaned/` 与 `data/curated/` 下，可用于对比。

## 4. 产物布局

```
configs/clean/pick_cube.yaml
        │
        │  embodied clean
        ▼
data/cleaned/<run_key>/
  data.parquet                 # 清洗后行，保留原始 schema/索引
  videos/<camera>/episode_XXXXX.mp4
  manifest.parquet / .json     # 每集决策、裁剪、源视频引用、clip checksum
  selection.parquet            # 每个源帧的保留/丢弃与映射
  source_schema.json           # 源 info.json 快照
  tasks.parquet                # 任务标签
  config.snapshot.json
  run.json                     # lineage、指纹、计数、工具版本
  qc/{summary.json,episodes.parquet,report.md}
        │
        │  embodied convert
        ▼
data/curated/<run_key>/
  dataset/                     # 只有合法 LeRobot v3 文件
    meta/{info.json,stats.json,tasks.parquet,episodes/...}
    data/chunk-000/file-000.parquet
    videos/<camera>/chunk-000/file-000.mp4
  conversion.json              # episode 新旧映射、编码参数
  validation.json              # 回读校验结果
  run.json
```

关键设计：

- **清洗层保留原始 `episode_index` / `frame_index` / `index` / `timestamp`**，不重编号；允许空洞。
  它不是可直接加载的 LeRobotDataset（`run.json` 明确标记 `is_lerobot_dataset: false`），
  转换阶段才负责规范化。
- **Stage-1 clip 用 CRF 18 近无损编码**；转换阶段用源编码参数（CRF 30）生成最终 canonical 视频，
  避免二次压缩叠加。
- **所有阈值在 YAML**；每条 episode 的指标、判定原因、warning 都写入 `qc/` 与 `manifest`。
- **确定性 run key** = 源内容指纹 + 配置指纹；同一输入/配置重复执行会复用已有产物；
  不同指纹需显式 `--overwrite`。
- 写入使用 staging 目录 + 原子 rename；转换在验证通过后才发布。
- 默认只写本地；`--push-to-hub` 显式开启，且在回读校验通过后才推送。

## 5. 运行

```bash
# 1) 先看报告，不落盘
.venv/bin/embodied clean --config configs/clean/pick_cube.yaml --dry-run

# 2) 正式清洗：data.parquet + clips + manifest + QC
.venv/bin/embodied clean --config configs/clean/pick_cube.yaml

# 3) 转换 + 自动回读校验
.venv/bin/embodied convert --config configs/clean/pick_cube.yaml

# 4) 对已有数据集单独重跑校验
.venv/bin/embodied validate \
  --config configs/clean/pick_cube.yaml \
  --dataset data/curated/<run_key>/dataset

# 可选：校验通过后推送
.venv/bin/embodied convert --config configs/clean/pick_cube.yaml --push-to-hub
```

退出码：`0` 成功，`1` 校验/策略失败，`2` 配置/输入/契约错误。

## 6. 配置与规则

配置文件：`configs/clean/pick_cube.yaml`。规则阈值、默认值和决策顺序见
[configuration.md](configuration.md#2-清洗配置yaml)。

新增一条规则的位置：

- 帧级派生指标：`src/embodied_data_infra/cleaning/rules.py::add_frame_metrics`
- 集级聚合：`_aggregate_frames`
- 判定原因：`decide_episodes`
- 同时补 `tests/unit/test_rules.py`，再跑真实数据 `--dry-run` 复核。

## 7. 产物契约

清洗层（convert 的输入）必须包含：

- `data.parquet`：源 schema 的清洗后行；`index` 可能不连续。
- `manifest.parquet`：每集一行，含 `decision`、`reasons`、`trim_head/tail`、`n_frames_in/out`、
  `curated_episode_index`、`clean_row_start/stop`、源视频 from/to、clip 路径/checksum/帧数。
- `selection.parquet`：每个**源帧**一行（含 drop），含 `kept`、`clip_frame_index`、
  `curated_episode_index`。
- `videos/<camera>/episode_XXXXX.mp4`：只保留 kept episode 的裁剪后 clip。
- `source_schema.json`、`tasks.parquet`、`run.json`（含 `output_fingerprint`）。

转换阶段启动前会重算清洗层指纹并校验：文件 checksum、data 行数、selection、clip 帧数、
episode 映射连续从 0 开始。任何一项不匹配都会拒绝转换。

> 注意：清洗层的 `timestamp` 仍是源相对时间。ep0 裁剪后首帧 timestamp 是 2.2s 而不是 0；
> 下游请用 `clip_frame_index` 索引 clip，不要用清洗后的 `timestamp` 反推 clip 帧号。

## 8. 回读校验

`convert` 末尾自动执行，也可单独 `validate`：

- `LeRobotDataset(root=dataset/)` 重新加载，检查 total_episodes / total_frames / fps / features。
- `hf_dataset` 全量比对 `observation.state`、`action`、`next.reward`、`next.done`
  与清洗层 parquet。
- 检查每集 timestamp 网格、frame_index、episode_index、task。
- 每集抽首/中/尾帧，与 Stage-1 clip 比较形状、非全黑、相关性/MAE。
- 检查 `meta/stats.json` 存在且包含关键特征。
- 结果写入 `validation.json`；失败返回非 0。

## 9. 测试

```bash
.venv/bin/python -m pytest
```

- `tests/unit/test_config.py`：默认值、未知字段、阈值校验、config hash 稳定性。
- `tests/unit/test_rules.py`：trim 阈值、非法 gripper、NaN、成功缺失、时间戳断裂、frame gap、
  时长离群、中间停滞。
- `tests/unit/test_cleaning.py`：清洗层契约、dry-run 不落盘、幂等、篡改 clip 被指纹拒绝。
- `tests/unit/test_runners.py`：采集/训练配置校验、命令构造、dry-run 不启动子进程。
- `tests/integration/test_roundtrip.py`：合成 mini 数据集 clean → convert → validate，无网络依赖。

所有测试使用 `tmp_path` 合成数据，不访问 HF。

## 10. 设计取舍与后续可优化

1. **转换阶段二次编码**：Stage 1 clip 是近无损，Stage 2 用官方 writer 生成 canonical 视频。
   这是“metadata/stats 绝对合规”优先的取舍；下一步可复用 clip 只重写 metadata，省一次编码。
2. **清洗层不是可训练数据集**：刻意保留源索引与空洞，避免把“清洗”和“规范化”混在一起。
3. **demo 规模内存解码**：每个源视频文件每路相机解码一次到内存（约 32MB/相机）。
   大数据量应改为按 episode seek/流式。
4. **单进程视频编码**：`conversion.parallel_encoding=false`，保证确定性；批量数据可开启。
5. **HF 推送默认关闭**：`--push-to-hub` 需要 token，且只在 validation PASS 后执行。
