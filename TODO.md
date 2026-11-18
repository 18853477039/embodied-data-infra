# TODO / Known Issues

按优先级记录当前已知问题。完成后请更新本文件、README 的「已知问题」和相关 docs。

---

## P0：训练包装器与 HIL-SERL 配置不匹配

### 现象

`embodied train` 当前固定构造：

```bash
python -m lerobot.scripts.lerobot_train \
  --config_path configs/train/sac_gym_hil_pick_cube.json \
  --dataset.root data/curated/<run_key>/dataset
```

但 `configs/train/sac_gym_hil_pick_cube.json` 是 HIL-SERL 的
`TrainRLServerPipelineConfig`，包含：

- `algorithm`（SAC 超参数）
- `mixer` / `online_ratio`
- `policy.actor_learner_config`（gRPC learner 地址、端口、参数推送频率）
- `policy.concurrency`

标准 `lerobot.scripts.lerobot_train` 使用 `TrainPipelineConfig`，不包含这些字段，
命令会在配置解析阶段报 unknown fields，不会进入训练。

### 当前正确的运行方式

手动双进程，先 learner、后 actor，两者使用同一份配置和同一 `--dataset.root`：

```bash
# 终端 1
.venv/bin/python -m lerobot.rl.learner \
  --config_path configs/train/sac_gym_hil_pick_cube.json \
  --dataset.root data/curated/<run_key>/dataset

# 终端 2
.venv/bin/python -m lerobot.rl.actor \
  --config_path configs/train/sac_gym_hil_pick_cube.json \
  --dataset.root data/curated/<run_key>/dataset
```

说明：

- learner 必须先启动，actor 依赖 `policy.actor_learner_config.learner_host/port`。
- learner 用 `cfg.dataset` 构建 offline replay buffer；`--dataset.root` 决定离线数据来源。
- `dataset.repo_id` 不会被 `--dataset.root` 覆盖，仍作为数据集标识符。

### 建议修复

1. `training/runner.py` 从“单一 `lerobot_train`”改为 role-aware：
   - `build_learner_command(...)` → `python -m lerobot.rl.learner --config_path ...`
   - `build_actor_command(...)` → `python -m lerobot.rl.actor --config_path ...`
2. CLI 增加角色选择：
   ```bash
   embodied train --role learner --config ...
   embodied train --role actor --config ...
   # 可选：--role both 负责按顺序启动并管理两个进程
   ```
3. `--latest-curated` 保留，解析出的 root 同时注入 learner/actor。
4. `validate_train_config` 改为校验 `TrainRLServerPipelineConfig` 的字段
   （`dataset` 可为空、`algorithm`、`policy.actor_learner_config`、`concurrency`）。
5. 补测试：
   - learner/actor 命令构造
   - 两个角色共享相同 root
   - dry-run 不启动子进程
   - 先用 `TrainRLServerPipelineConfig` 做一次真实解析预检
6. 同步更新 `README.md`、`docs/operations.md`、`docs/configuration.md`。

### 验收标准

- `embodied train --role learner --config ... --latest-curated --dry-run` 生成的命令包含
  `lerobot.rl.learner` 和正确的 `--dataset.root`。
- `embodied train --role actor ...` 同理指向 `lerobot.rl.actor`。
- 真实启动时 learner 先就绪、actor 能连接；训练配置不再出现 unknown-field 错误。
- `make check` 通过。

---

## P1：训练配置与 LeRobot 版本的兼容性预检

当前配置绑定 LeRobot 0.7.1。建议在训练包装器中增加一个预检：

- 用对应的 config dataclass（`TrainRLServerPipelineConfig`）解析配置；
- 解析失败时在启动子进程前给出明确错误，而不是让 LeRobot 在深层报错；
- 在 `run.json` / 训练输出中记录 LeRobot 版本。

---

## P2：文档一致性

- `docs/configuration.md` 第 3 节目前仍按 `lerobot-train` 描述 CLI 映射，需在 P0 修复后
  改为 `lerobot.rl.learner` / `lerobot.rl.actor`。
- `docs/operations.md` 的 4.4 节已加警告，修复后删除警告并恢复统一命令。
- `Makefile` 的 `train` 目标在 P0 修复前不要使用；修复后改为 role-aware 默认值。

---

## P3：仓库治理

- [ ] 指定 LICENSE（README 已标注）。
- [ ] `git init` 并提交当前基线（当前不是 git 仓库）。
- [ ] CI 中固定 LeRobot/依赖版本，并考虑增加配置解析的 smoke test。
- [ ] 评估是否需要 lockfile（当前使用 pyproject 范围依赖）。
