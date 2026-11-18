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
| 训练 | `embodied train` | 训练 JSON + curated 数据集 | `outputs/train/<date>/<time>_<job>/` | 包装 `lerobot-train`，一键选用最新 curated 数据集 |

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
                                             │ embodied train
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
│   └── training/runner.py         # 训练阶段：配置校验 + lerobot-train 启动
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

# 6) 训练：自动选择 data/curated 下最新的数据集
.venv/bin/embodied train \
  --config configs/train/sac_gym_hil_pick_cube.json \
  --latest-curated

# 或者用 Makefile
make check          # lint + tests
make pipeline       # clean + convert
make train          # 最新 curated 数据集
```

详细命令、覆盖参数与排障见 [docs/operations.md](docs/operations.md)；
配置字段见 [docs/configuration.md](docs/configuration.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 分层设计、阶段边界、可复现性与扩展点 |
| [docs/configuration.md](docs/configuration.md) | 采集/清洗/训练三类配置与 CLI 覆盖项 |
| [docs/pipeline.md](docs/pipeline.md) | 清洗算法、真实数据结论、产物契约与回读校验 |
| [docs/operations.md](docs/operations.md) | 环境、运行手册、HF/W&B、故障排查 |

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
