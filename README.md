# Embodied Data Infra

面向 LeRobot 机械臂数据集的端到端数据基础设施：**采集 → 清洗 → 转换 → 训练**。
所有阶段共享同一套目录约定、配置契约和可审计产物，既可以在单机工作站上跑通完整闭环，
也可以作为企业数据流水线的骨架逐步扩展。

---

## 能力范围

| 阶段 | 命令 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| 采集 | `embodied collect` | `configs/collect/*.json` | LeRobot 原始数据集（本地缓存 / Hub） | 包装 `lerobot-record`，支持手柄遥操作、干跑与字段覆盖 |
| 清洗 | `embodied clean` | 原始 LeRobot v3 数据集 + YAML | `data/cleaned/<run_key>/` | Polars 内存清洗、切片编码、QC 报告、逐帧 selection |
| 转换 | `embodied convert` | 清洗层 | `data/curated/<run_key>/dataset/` | 官方 LeRobot writer 重建标准数据集，自动回读校验 |
| 校验 | `embodied validate` | 已转换数据集 | `validation.json` | 独立重跑契约检查与回读比对 |
| 训练 | `embodied train` ⚠ 待适配 | 训练 JSON + curated 数据集 | `outputs/train/...` | 当前包装 `lerobot-train`；示例 SAC 配置实际是 HIL-SERL `TrainRLServerPipelineConfig`，需要 `lerobot.rl.learner` + `lerobot.rl.actor`，见 [TODO.md](TODO.md) |

> 当前示例任务：Franka Panda `PandaPickCubeGamepad-v0`，10 fps，双相机 128×128，
> 30 episodes 本地录制。详细清洗结论见 [docs/pipeline.md](docs/pipeline.md)。

## 端到端流程

```
 ┌──────────────┐   lerobot-record    ┌──────────────────┐
 │  gamepad /   │ ──────────────────► │  raw LeRobot v3  │
 │  gym_hil env │                     │  (local / Hub)   │
 └──────────────┘                     └────────┬─────────┘
                                               │ embodied clean
                                               ▼
                                    ┌──────────────────────┐
                                    │  cleaned layer       │  parquet + clips
                                    │  data/cleaned/<key>  │  manifest + selection
                                    └────────┬─────────────┘  QC report
                                             │ embodied convert
                                             ▼
                                    ┌──────────────────────┐
                                    │  curated dataset     │  canonical LeRobot v3
                                    │  data/curated/<key>  │  stats / metadata
                                    └────────┬─────────────┘
                                             │ embodied train（⚠ 待适配）
                                             ▼
                                    ┌──────────────────────┐
                                    │  outputs/train/...   │  checkpoints + logs
                                    └──────────────────────┘
```

## 目录结构

```
embodied-data-infra/
├── configs/                       # 配置即代码，按阶段分组
│   ├── collect/gym_hil_pick_cube.json
│   ├── clean/pick_cube.yaml
│   └── train/sac_gym_hil_pick_cube.json
├── src/embodied_data_infra/       # 可安装的单一 Python 包
│   ├── cli.py                     # 统一 CLI：collect / clean / convert / validate / train
│   ├── collection/runner.py       # 采集阶段：配置校验 + lerobot-record 启动
│   ├── cleaning/                  # 清洗阶段：加载、规则、切片、报告、转换、校验
│   └── training/runner.py         # 训练阶段：配置校验 + 训练入口启动（⚠ 待适配 HIL-SERL）
├── scripts/check_gamepad.py       # 录制前的手柄自检脚本
├── tests/
│   ├── unit/                      # 规则、配置、阶段包装器
│   └── integration/               # 合成数据 clean → convert → validate 闭环
├── docs/                          # 架构、配置、流水线契约、运维手册
├── data/                          # 本地数据产物（gitignored）
│   ├── raw/  cleaned/  curated/  reports/
├── outputs/                       # 训练产物（gitignored）
├── logs/                          # 历史运行日志（gitignored）
├── Makefile                       # 常用操作入口
├── pyproject.toml                 # 打包、依赖、pytest/ruff/coverage 配置
└── .github/workflows/ci.yml       # lint + test
```

## 快速开始

要求：Linux、Python ≥ 3.10、`ffmpeg`；采集阶段还需要已校准的机械臂、相机和手柄。

```bash
# 1) 安装（项目自带 .venv，推荐复用）
uv pip install -e ".[dev]" --python .venv/bin/python

# 2) 查看统一 CLI
.venv/bin/embodied --help

# 3) 采集（先干跑确认命令）
.venv/bin/embodied collect --config configs/collect/gym_hil_pick_cube.json --dry-run
.venv/bin/embodied collect --config configs/collect/gym_hil_pick_cube.json

# 4) 清洗：先看 QC 决策，不落盘
.venv/bin/embodied clean --config configs/clean/pick_cube.yaml --dry-run
.venv/bin/embodied clean --config configs/clean/pick_cube.yaml

# 5) 转换 + 自动回读校验
.venv/bin/embodied convert --config configs/clean/pick_cube.yaml

# 6) 训练（⚠ 当前 wrapper 尚未适配 HIL-SERL，需手动双进程）
#    先启动 learner；另开终端再启动 actor，两者使用同一配置和数据集
.venv/bin/python -m lerobot.rl.learner \
  --config_path configs/train/sac_gym_hil_pick_cube.json \
  --dataset.root data/curated/<run_key>/dataset

.venv/bin/python -m lerobot.rl.actor \
  --config_path configs/train/sac_gym_hil_pick_cube.json \
  --dataset.root data/curated/<run_key>/dataset

# 或者用 Makefile（清洗 + 转换）
make check          # lint + tests
make pipeline       # clean + convert
# make train 目前不可用于这份 HIL-SERL 配置，原因见 TODO.md
```

详细命令、覆盖参数与排障见 [docs/operations.md](docs/operations.md)；
配置字段见 [docs/configuration.md](docs/configuration.md)。

## 已知问题（TODO）

- **训练包装器与示例配置不匹配（P0）**：`embodied train` 当前构造的是
  `lerobot.scripts.lerobot_train` 命令，而 `configs/train/sac_gym_hil_pick_cube.json`
  是 HIL-SERL 的 `TrainRLServerPipelineConfig`，包含 `algorithm`、`mixer`、`online_ratio`、
  `policy.actor_learner_config` 等字段，标准 `TrainPipelineConfig` 不识别；
  当前命令会在配置解析阶段失败。正确流程是先 `lerobot.rl.learner`、后 `lerobot.rl.actor`，
  详见 [TODO.md](TODO.md)。
- **LICENSE 未指定**：发布前需由项目负责人补充许可证。
- **尚未初始化 git**：当前目录不是 git 仓库，建议纳入版本控制后再做团队协作。

完整待办、影响范围和验收标准见 [TODO.md](TODO.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 分层设计、阶段边界、可复现性与扩展点 |
| [docs/configuration.md](docs/configuration.md) | 采集/清洗/训练三类配置与 CLI 覆盖项 |
| [docs/pipeline.md](docs/pipeline.md) | 清洗算法、真实数据结论、产物契约与回读校验 |
| [docs/operations.md](docs/operations.md) | 环境、运行手册、HF/W&B、故障排查 |
| [docs/experiments.md](docs/experiments.md) | 实验记录：数据、配置、结果与初步结论 |
| [TODO.md](TODO.md) | 已知问题、修复计划与验收标准 |

## 质量保障

```bash
make check      # ruff check + ruff format --check + pytest
make format     # 自动修复与格式化
```

- 清洗规则全部有单元测试，合成数据闭环测试不访问网络。
- 采集/训练包装器有命令构造、配置校验和 dry-run 测试。
- 所有阶段产物带 `run.json`（版本、指纹、计数、lineage），转换只在回读校验通过后发布。

## 当前本地状态

2026-11-17 的 30-episode 录制（666 帧）已完成清洗与转换：

- 清洗结果：29 episodes / 561 frames / 58 clips，ep19 被丢弃，ep0 裁掉 15 帧启动空转。
- 清洗层：`data/cleaned/d163e86dc4aa__f3795cbc51b5/`
- 最终数据集：`data/curated/d163e86dc4aa__f3795cbc51b5/dataset/`
- 回读校验：`PASS`（`data/curated/d163e86dc4aa__f3795cbc51b5/validation.json`）
