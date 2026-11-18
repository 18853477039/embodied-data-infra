# 运维手册

本文覆盖环境准备、日常运行、Hub/W&B 操作和故障排查。架构与配置分别见
[architecture.md](architecture.md) 和 [configuration.md](configuration.md)。

## 1. 环境要求

| 依赖 | 要求 | 用途 |
|---|---|---|
| Linux | 推荐 Ubuntu 22.04+ | 机械臂/相机驱动、PyAV |
| Python | ≥ 3.10（本项目在 3.12 验证） | 全部阶段 |
| ffmpeg | 系统包 | 视频解码/编码 |
| LeRobot | 0.7.x | 录制、数据集、训练 |
| 机械臂 + 相机 + 手柄 | 采集阶段 | `PandaPickCubeGamepad-v0` |
| NVIDIA GPU | 训练阶段 | SAC 训练；CPU 仅用于 smoke test |

## 2. 安装

```bash
# 复用仓库自带 .venv；如需重建：
uv venv --python 3.12 .venv

# 代码 + 开发工具（pytest / ruff / pre-commit）
uv pip install -e ".[dev]" --python .venv/bin/python

# 采集脚本需要 pygame
uv pip install -e ".[collect]" --python .venv/bin/python

# 确认 CLI
.venv/bin/embodied --help
```

凭据放在环境变量中，不要写进配置或提交到仓库：

```bash
cp .env.example .env
export HF_TOKEN=hf_xxx          # 推送数据集时需要
export WANDB_API_KEY=xxx        # 训练日志需要
export WANDB_MODE=offline       # 离线环境可选
```

> `.env` 不会被自动加载；请 `source .env` 或在 shell/CI 中注入。
> `HF_TOKEN` 也可以用 `hf auth login` 写入本地凭据缓存。

## 3. 录制前检查

```bash
# 手柄可见性与轴/按键编号
.venv/bin/python scripts/check_gamepad.py

# 干跑：只校验配置并打印 lerobot-record 命令
.venv/bin/embodied collect \
  --config configs/collect/gym_hil_pick_cube.json --dry-run
```

检查清单：

- [ ] 手柄被 pygame 识别，轴/按键编号符合预期
- [ ] 相机画面正常，front/wrist 的 shape 与配置一致
- [ ] 机械臂已校准，复位姿态可用
- [ ] 磁盘剩余空间足够（视频是主要占用）
- [ ] `dataset.repo_id` / `push_to_hub` / `num_episodes_to_record` 符合本次计划
- [ ] 若需要上传，`HF_TOKEN` 有目标 repo 的写权限

## 4. 日常运行

### 4.1 采集

```bash
# 使用配置中的 repo_id、集数和推送策略
.venv/bin/embodied collect --config configs/collect/gym_hil_pick_cube.json

# 临时覆盖
.venv/bin/embodied collect \
  --config configs/collect/gym_hil_pick_cube.json \
  --repo-id user/pick_cube_run2 \
  --episodes 20 \
  --no-push-to-hub

# 透传 LeRobot 原生参数
.venv/bin/embodied collect --config ... -- --display_data true
```

录制完成后，先在 LeRobot 缓存目录确认 `meta/info.json`、`data/`、`videos/` 完整，
再进入清洗。

### 4.2 清洗

```bash
# 只计算决策并写 QC 报告，不产生清洗层
.venv/bin/embodied clean --config configs/clean/pick_cube.yaml --dry-run

# 正式执行（幂等：同源同配置会复用）
.venv/bin/embodied clean --config configs/clean/pick_cube.yaml

# 源或配置变了、确认要覆盖旧产物时
.venv/bin/embodied clean --config configs/clean/pick_cube.yaml --overwrite
```

阅读顺序：`data/reports/<run_key>/report.md` → `qc/summary.json` → `manifest.parquet`。
若某集被 drop/trim，`reasons` 会给出规则名和指标值。

### 4.3 转换与校验

```bash
# 转换 + 自动回读校验
.venv/bin/embodied convert --config configs/clean/pick_cube.yaml

# 对已有数据集单独校验
.venv/bin/embodied validate \
  --config configs/clean/pick_cube.yaml \
  --dataset data/curated/<run_key>/dataset

# 校验通过后推送 Hub（config.output.repo_id）
.venv/bin/embodied convert --config configs/clean/pick_cube.yaml --push-to-hub
```

校验失败时不会发布新的 curated 目录；查看 `validation.json` 的 `failures` 列表定位问题。

### 4.4 训练（HIL-SERL actor/learner）

> ⚠ 当前 `embodied train` 包装的是标准 `lerobot-train`，**不适用于本项目的 SAC 配置**。
> `configs/train/sac_gym_hil_pick_cube.json` 是 HIL-SERL 的 `TrainRLServerPipelineConfig`，
> 需要先启动 learner、再启动 actor。修复计划见 [../TODO.md](../TODO.md)。

```bash
# 终端 1：learner（先启动；内部启动 gRPC 服务并加载离线 replay buffer）
.venv/bin/python -m lerobot.rl.learner \
  --config_path configs/train/sac_gym_hil_pick_cube.json \
  --dataset.root data/curated/<run_key>/dataset

# 终端 2：actor（连接 learner，在环境里在线采样）
.venv/bin/python -m lerobot.rl.actor \
  --config_path configs/train/sac_gym_hil_pick_cube.json \
  --dataset.root data/curated/<run_key>/dataset
```

说明：

- learner 必须先启动；actor 依赖 `policy.actor_learner_config.learner_host/port`。
- `--dataset.root` 指定 learner 的离线数据集；`dataset.repo_id` 仍是标识符，不会被覆盖。
- 两个进程应使用同一份配置和同一个数据集 root。
- 修复后的统一入口设计（`--role learner|actor|both`）见 [../TODO.md](../TODO.md)。

训练产物：

```
outputs/train/<date>/<time>_<job_name>/
├── checkpoints/
├── logs/
└── ...
```

如果 `wandb.enable=true` 但没有 API key，训练可能启动失败；可先临时：
`-- --wandb.enable false`。

## 5. Hugging Face 操作

### 上传已有数据集

```bash
# 示例：把本地 LeRobot 缓存目录上传为数据集 repo
hf upload <user>/<dataset-repo> \
  ~/.cache/huggingface/lerobot/<user>/<dataset-repo> \
  --repo-type dataset
```

清洗/转换后的数据集也可以用同样方式上传；推荐优先使用
`embodied convert --push-to-hub`，因为它只在校验 PASS 后推送，且使用配置中的
`output.repo_id`。

### 下载/拉取

```bash
hf download <user>/<dataset-repo> --repo-type dataset
```

清洗配置设置 `source.local_root: null` 时，流水线会自动下载到 `source.download_dir`。

### 谨慎操作

删除本地缓存会导致重建 run key、重新上传（源内容不变时指纹相同，但下载耗时）：

```bash
# 确认路径和备份后再执行
rm -rf ~/.cache/huggingface/lerobot/<user>/<dataset-repo>
```

## 6. 退出码

| 码 | 含义 | 处理 |
|---|---|---|
| 0 | 成功（dry-run 也返回 0） | — |
| 1 | 回读校验/策略失败，或子进程训练/录制失败 | 看 `validation.json` / 子进程日志 |
| 2 | 配置、输入、契约错误 | 修配置/路径，不要盲目 `--overwrite` |
| 130 | 用户 Ctrl+C | 清理 staging 后可重跑 |

## 7. 故障排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `No module named embodied_data_infra` | 未安装/旧 editable 指向已删除目录 | `uv pip install -e ".[dev]" --python .venv/bin/python` |
| `No gamepad detected` | USB/权限/驱动 | `lsusb`、`scripts/check_gamepad.py`、udev 规则 |
| 录制没出现在 Hub | `push_to_hub=false` 或 token 无权限 | 检查配置与 `HF_TOKEN`；手动 `hf upload` |
| `Output exists with a different fingerprint` | 源或配置变了 | 先 `--dry-run` 对比报告；确认后 `--overwrite` |
| `fingerprint mismatch`（清洗层） | 清洗层被手动修改 | 不要手改产物；重跑 `clean` |
| 转换后 validation FAIL | 视频/行数/ stats 不一致 | 看 `validation.json` 的 failures；保留产物用于定位 |
| `local_root does not contain meta/info.json` | 路径错/未下载完整 | 检查 `source.local_root`，或设为 `null` 走 Hub |
| PyAV/编码错误 | 缺 ffmpeg 或编码器 | 安装 ffmpeg；确认源视频编码 |
| 内存占用过高 | demo 规模按整文件解码 | 减小单批数据，参考 [pipeline.md](pipeline.md) 的优化项 |
| W&B 无法连接 | 无网络/无 key | `WANDB_MODE=offline` 或 `wandb.enable=false` |
| 测试访问网络 | 未设置离线 | 测试默认 `HF_HUB_OFFLINE=1`；检查自定义 fixture |

## 8. 变更管理

- 修改清洗阈值 → 配置指纹变化 → 新 run key；旧产物保留可对比。
- 修改代码但配置不变 → run key 不变，可能复用旧产物；需要重算时用 `--overwrite`。
- 升级 LeRobot/Polars/PyAV → `run.json` 会记录版本；重要发布前建议重跑一次全流程。
- 发布前检查：`make check`、`embodied clean --dry-run`、`embodied convert`（含 validation）。
