# 架构设计

本文说明 `embodied-data-infra` 的分层、阶段边界和可复现性设计。
具体命令与配置字段分别见 [operations.md](operations.md) 和 [configuration.md](configuration.md)。

## 1. 设计目标

1. **阶段隔离**：采集、清洗、转换、训练是四个可独立重跑的阶段，中间产物是显式契约。
2. **可审计**：每一集为什么保留/裁剪/丢弃、每个产物由什么输入和配置生成，都有机器可读记录。
3. **可复现**：run key 由源内容指纹 + 配置指纹决定；写入使用 staging + 原子 rename；版本写入 `run.json`。
4. **配置即代码**：阈值、路径、repo id、训练超参全部在 `configs/`，代码不藏策略。
5. **本地优先**：默认只读写本地目录；推送 Hub 必须显式开启，且发生在校验通过之后。
6. **薄包装**：采集和训练不重复实现 LeRobot 的能力，只做配置校验、参数覆盖和产物定位。

## 2. 分层

```
┌────────────────────────────────────────────────────────────┐
│ CLI 层          src/embodied_data_infra/cli.py             │
│                 collect / clean / convert / validate / train│
├────────────────────────────────────────────────────────────┤
│ 阶段包装层      collection/runner.py   training/runner.py   │
│                 配置校验、CLI 覆盖、子进程启动、数据集选择    │
├────────────────────────────────────────────────────────────┤
│ 领域核心层      cleaning/                                   │
│                 config → source → loader → rules → media    │
│                 → cleaning → conversion → validation        │
├────────────────────────────────────────────────────────────┤
│ 外部依赖        LeRobot（record/train/LeRobotDataset/编码器）│
│                 Polars、PyAV、Hugging Face Hub              │
└────────────────────────────────────────────────────────────┘
```

依赖方向自上而下，`cleaning/` 不依赖 `collection/` 或 `training/`，保证清洗核心可以单独复用。

## 3. 阶段与数据流

| 阶段 | 代码入口 | 输入 | 输出 | 副作用边界 |
|---|---|---|---|---|
| collect | `collection.runner.run_record` | 采集 JSON | 原始 LeRobot 数据集 | 调用 `lerobot-record`，写本地缓存/Hub |
| clean | `cleaning.cleaning.run_clean` | 原始数据集 + YAML | 清洗层 + QC | 只写 `data/cleaned`、`data/reports` |
| convert | `cleaning.conversion.run_convert` | 清洗层 + YAML | canonical 数据集 | 只写 `data/curated`，校验通过后发布 |
| validate | `cleaning.validation` | curated 数据集 | `validation.json` | 只读数据集，写校验结果 |
| train | `training.runner.run_train` | 训练 JSON + curated | checkpoints/logs | 调用 `lerobot-train`，写 `outputs/` |

数据流：

```
raw (LeRobot v3)
  └─ clean ──► cleaned layer（保留源索引，不是可训练数据集）
                 └─ convert ──► curated dataset（标准 LeRobot v3，可直接训练）
                                   └─ train ──► policy checkpoints
```

清洗层与最终数据集是**两个不同的契约**：

- 清洗层保留原始 `episode_index` / `frame_index` / `index` / `timestamp`，允许空洞，便于审计；
  通过 `selection.parquet` 记录每个源帧的保留/丢弃与映射。
- 转换阶段才重新编号、重编码视频、重算 stats/metadata，产出一个严格合法的 LeRobot 数据集。
- 详见 [pipeline.md](pipeline.md)。

## 4. 目录职责

| 路径 | 职责 | 是否入库 |
|---|---|---|
| `configs/collect/` | 录制环境、数据集、机器人/手柄配置 | 是 |
| `configs/clean/` | 清洗阈值、输入输出路径、转换与推送策略 | 是 |
| `configs/train/` | 算法、policy、env、数据集和日志配置 | 是 |
| `src/embodied_data_infra/` | 可安装代码，按阶段分子包 | 是 |
| `scripts/` | 人工运维脚本（手柄自检等），不进入包 | 是 |
| `tests/unit/` | 规则、配置、命令构造、dry-run 行为 | 是 |
| `tests/integration/` | 合成数据 clean → convert → validate 闭环 | 是 |
| `docs/` | 架构、契约、配置、运维 | 是 |
| `data/` | 原始、清洗、curated、报告等本地数据 | 否 |
| `outputs/` | 训练 checkpoint、日志、W&B 关联产物 | 否 |
| `logs/` | 历史运行日志归档 | 否 |

## 5. 可复现性与 lineage

### run key

```
run_key = <源内容指纹前 12 位>__<配置指纹前 12 位>
```

- **源内容指纹**：对源数据集关键文件计算 SHA-256（视频可选，由 `source.hash_videos` 控制）。
- **配置指纹**：YAML 规范化为 JSON 后计算 SHA-256，所有阈值参与。
- 同一输入 + 同一配置 → 同一个 run key → 复用已有产物，不重复计算。
- 指纹不同且目标目录已存在 → 默认拒绝覆盖，必须显式 `--overwrite`。

### 产物元数据

每个阶段目录都包含 `run.json`，记录：

- `pipeline` / `pipeline_version` / 阶段名
- 输入指纹（源数据或上游清洗层）
- 配置快照与配置指纹
- 计数（episodes / frames / clips）
- 依赖版本（polars、pyarrow、av、lerobot）
- 输出指纹（`output_fingerprint`）

`conversion.json` 额外保存 episode 新旧映射、编码参数；`validation.json` 保存回读校验结果。

### 写入协议

1. 写入临时 staging 目录。
2. 计算并校验产物指纹。
3. 原子 rename 到最终目录。
4. 转换阶段在回读校验 PASS 后才发布，失败则不产生半成品。

## 6. 配置与覆盖

- 配置文件是唯一策略来源；CLI 只提供少量操作型覆盖（数据集路径、输出目录、episode 数等）。
- 采集/训练包装器支持 `--` 之后的参数透传，复杂覆盖仍可直接使用 LeRobot 原生参数。
- 清洗阶段不支持单条阈值覆盖：阈值改变必须改 YAML，确保 run key 变化、结果可追溯。

## 7. 扩展点

| 需求 | 扩展位置 |
|---|---|
| 新机器人/新任务录制 | 新增 `configs/collect/<task>.json`，无需改代码 |
| 新清洗规则 | `cleaning/rules.py`：帧级指标、集级聚合、判定原因，并补 `tests/unit/test_rules.py` |
| 新输入格式 | `cleaning/source.py` + `loader.py`，保持 `SourceDataset` 契约 |
| 新训练算法 | 新增 `configs/train/<algo>.json`，交给 `lerobot-train` |
| 新阶段（如评估、回放） | 在 `src/embodied_data_infra/` 增加子包，CLI 注册子命令 |
| 云端/集群执行 | 包装器均为显式 argv，可替换为作业调度器的 command 模板 |

## 8. 非目标

- 不实现机器人驱动、相机采集、手柄映射——这些属于 LeRobot/仿真环境。
- 不实现 policy 网络与训练循环——训练委托给 LeRobot。
- 不在本仓库保存大数据集或 checkpoint；`data/`、`outputs/` 均被 gitignore。
