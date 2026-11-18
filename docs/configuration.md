# 配置参考

配置按阶段分组，全部位于 `configs/`。代码只提供操作型 CLI 覆盖，策略阈值必须改配置文件，
以保证配置指纹和 run key 能准确反映数据是如何被处理的。

```
configs/
├── collect/gym_hil_pick_cube.json   # lerobot-record 的输入
├── clean/pick_cube.yaml             # embodied clean / convert 的输入
└── train/sac_gym_hil_pick_cube.json # HIL-SERL actor/learner 的输入
```

> 相对路径（`data/raw`、`data/cleaned` 等）按**当前工作目录**解析。
> 约定从项目根目录运行命令；Makefile 已遵守该约定。

---

## 1. 采集配置（JSON）

示例：`configs/collect/gym_hil_pick_cube.json`

```jsonc
{
  "env": {
    "name": "gym_hil",
    "task": "PandaPickCubeGamepad-v0",
    "fps": 10,
    "robot": null,
    "teleop": null,
    "processor": {
      "control_mode": "gamepad",
      "observation": { "add_joint_velocity_to_observation": true },
      "gripper": { "use_gripper": true, "gripper_penalty": -0.02 },
      "reset": {
        "fixed_reset_joint_positions": [0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785],
        "reset_time_s": 1.0,
        "control_time_s": 15.0
      }
    },
    "features": { "...": "..." },
    "features_map": { "...": "..." }
  },
  "dataset": {
    "repo_id": "lilkm/pick_cube_franka_panda_30",
    "root": null,
    "task": "PandaPickCubeGamepad-v0",
    "num_episodes_to_record": 30,
    "replay_episode": 0,
    "push_to_hub": false
  },
  "mode": "record",
  "device": "cuda"
}
```

关键字段：

| 字段 | 作用 |
|---|---|
| `env.name` / `env.task` / `env.fps` | 环境标识、任务名、采样率 |
| `env.processor.control_mode` | `gamepad` 遥操作；`control_time_s` 决定单集最长时长 |
| `env.features` | 相机、state、action 的 shape 和类型，必须与清洗/训练配置一致 |
| `dataset.repo_id` | 目标数据集 id；本地录制也用它命名缓存目录 |
| `dataset.root` | 非空时写入指定目录，`null` 时使用 LeRobot 缓存约定 |
| `dataset.num_episodes_to_record` | 本次录制集数 |
| `dataset.push_to_hub` | 录制结束是否上传；默认 `false` |

采集阶段支持的 CLI 覆盖：

```bash
.venv/bin/embodied collect \
  --config configs/collect/gym_hil_pick_cube.json \
  --dataset-root /path/to/output \
  --repo-id user/new_dataset \
  --episodes 10 \
  --push-to-hub \
  --dry-run
```

`--push-to-hub / --no-push-to-hub` 覆盖 `dataset.push_to_hub`。
`--` 之后的参数原样透传给 `lerobot-record`：

```bash
.venv/bin/embodied collect --config ... -- --display_data true
```

---

## 2. 清洗配置（YAML）

示例：`configs/clean/pick_cube.yaml`

```yaml
source:
  repo_id: lilkm/pick_cube_franka_panda_30
  revision: main
  local_root: ~/.cache/huggingface/lerobot/lilkm/pick_cube_franka_panda_30
  download_dir: data/raw
  hash_videos: true

cleaning: { ... }
output:
  clean_root: data/cleaned
  curated_root: data/curated
  report_root: data/reports
  repo_id: lilkm/pick_cube_franka_panda_30_clean
  overwrite: false
conversion:
  parallel_encoding: false
  validate_after: true
  video_corr_min: 0.95
  video_mae_max: 10.0
push:
  enabled: false
  private: true
```

### source

| 字段 | 说明 |
|---|---|
| `repo_id` / `revision` | 源数据集标识；进入源指纹和 lineage |
| `local_root` | 本地数据集根目录；为 `null` 时从 Hub 下载到 `download_dir/<repo_id>` |
| `hash_videos` | 是否把视频内容纳入源指纹；关闭可加速但降低变更检测能力 |

### cleaning

| 规则 | 默认值 | 说明 |
|---|---|---|
| `min_frames` / `max_frames` | 5 / 150 | 绝对长度边界（150 = 15s × 10fps） |
| `duration_outlier_factor` | 3.0 | 长度 > 3 × 中位数 → drop |
| `action.bounds` | `[[0,0],[0,0],[-1,1],[0,2]]` | 逐维动作范围 |
| `action.allowed_values` | `{3: [1,2]}` | gripper 只能取 1/2；出现 0 判脏 |
| `idle.motion_eps` | 1e-4 | 关节状态 L1 差 < 阈值视为静止 |
| `idle.action_eps` | 1e-6 | 平移动作视为零的阈值 |
| `idle.min_trim_frames` | 10 | 只有连续空转 ≥ 1s 才裁，避免误删正常启动帧 |
| `idle.max_trim_ratio` | 0.8 | 允许裁掉长启动空转，但裁后必须 ≥ `min_frames` |
| `idle.interior_idle_ratio_max` | 0.20 | 裁剪后中间静止比例超限 → drop |
| `idle.interior_idle_run_max` | 10 | 裁剪后最长连续停滞超限 → drop |
| `clip_crf` | 18 | 清洗层中间视频近无损编码 |

`cleaning.success` 指定成功信号列（`next.reward` / `next.done`）以及是否强制要求成功。
`cleaning.fps_tolerance_s` 控制时间戳网格校验容差。

决策顺序：

1. 硬失败 → `drop`：`nan_inf`、`timestamp_grid`、`frame_index_gap`、`meta_length_mismatch`、
   `success_missing`、`action_domain`、`too_short`、`too_long`、`duration_outlier`、`all_idle`。
2. 首/尾连续空转 ≥ `min_trim_frames` 且裁剪比例 ≤ `max_trim_ratio` → `trim`；
   裁后短于 `min_frames` → `drop`。
3. 裁剪后的中间停滞超限 → `drop`。
4. 否则 `keep`。

### output

| 字段 | 说明 |
|---|---|
| `clean_root` | 清洗层根目录，每个 run 一个 `<run_key>/` |
| `curated_root` | 最终数据集根目录 |
| `report_root` | QC 报告根目录（dry-run 也会写这里） |
| `repo_id` | 推送 Hub 时使用的目标数据集 id |
| `overwrite` | 指纹不同时是否允许覆盖；建议保持 `false`，用 CLI 显式覆盖 |

### conversion

| 字段 | 说明 |
|---|---|
| `parallel_encoding` | 是否并行编码视频；默认关闭以保证确定性 |
| `validate_after` | 转换后自动回读校验（建议开启） |
| `video_corr_min` / `video_mae_max` | 回读抽帧与清洗层 clip 的相关性/MAE 容差 |

### push

`enabled` 是策略开关；实际推送还需要 `convert --push-to-hub`。推送发生在回读校验 PASS 之后。

---

## 3. 训练配置（JSON）

示例：`configs/train/sac_gym_hil_pick_cube.json`

这是 HIL-SERL `TrainRLServerPipelineConfig` 的完整快照，由
`lerobot.rl.learner` / `lerobot.rl.actor` 消费；它不是标准
`lerobot.scripts.lerobot_train` 的 `TrainPipelineConfig`。主要字段：

| 区块 | 关键字段 | 说明 |
|---|---|---|
| 顶层 | `output_dir` / `job_name` / `seed` / `steps` | `output_dir=null` 时用 LeRobot 默认的 `outputs/train/...` |
| `dataset` | `repo_id` / `use_imagenet_stats` | 默认指向原始数据集；训练 curated 数据时用 `--dataset-root` 覆盖 |
| `algorithm` | `type=sac`、学习率、`utd_ratio`、网络维度 | SAC 超参数 |
| `mixer` | `online_offline`、`online_ratio` | 在线/离线混合比例 |
| `policy` | `type=gaussian_actor`、输入输出特征、归一化 | 特征 shape 必须与数据集一致 |
| `env` | `type=gym_manipulator`、`task`、`fps`、`processor` | 在线采样环境，需与采集配置一致 |
| `wandb` | `enable` / `project` | 日志；关闭后不依赖 W&B |

> ⚠ 当前 `embodied train` 仍按 `lerobot-train` 构造命令，尚未适配 HIL-SERL。
> 目前请手动启动 learner 和 actor（先 learner 后 actor），并使用同一 `--dataset.root`；
> 修复计划见 [../TODO.md](../TODO.md)。

```bash
# 终端 1：learner
.venv/bin/python -m lerobot.rl.learner \
  --config_path configs/train/sac_gym_hil_pick_cube.json \
  --dataset.root data/curated/<run_key>/dataset

# 终端 2：actor
.venv/bin/python -m lerobot.rl.actor \
  --config_path configs/train/sac_gym_hil_pick_cube.json \
  --dataset.root data/curated/<run_key>/dataset
```

P0 修复后，CLI 计划支持以下覆盖（当前尚未生效）：

| 参数 | 计划映射到 `lerobot.rl.*` |
|---|---|
| `--dataset-root` | `--dataset.root <path>` |
| `--repo-id` | `--dataset.repo_id <id>` |
| `--output-dir` | `--output_dir <path>` |
| `--job-name` | `--job_name <name>` |
| `--resume / --no-resume` | `--resume true/false` |

`--latest-curated` 会扫描 `--curated-root` 下包含 `dataset/meta/info.json` 的 run，
按目录 mtime 选择最新一个；它与 `--dataset-root` 互斥。
修复后 `--` 之后的参数将原样透传给 `lerobot.rl.learner` / `lerobot.rl.actor`。
