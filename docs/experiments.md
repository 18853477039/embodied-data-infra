# 实验记录

记录实际跑过的训练/数据实验：数据、配置、命令、可量化结果与结论。
用途是验证流程和优化方向，不作为论文级统计结论。每次实验追加一节，保留产物路径。

---

## 实验 001：HIL-SERL 离线数据质量对学习速度的影响（2026-11-18）

### 目的

用同一份 SAC / HIL-SERL 配置，对比两种离线数据对在线学习速度的影响：

- A 组（示例数据）：Hub 原始数据集，含失败 episode 和空闲帧。
- B 组（我的数据）：本地清洗后的 curated 数据集，只保留成功 episode 并裁剪首尾空闲段。

### 环境与配置

| 项 | 值 |
|---|---|
| 训练配置 | `configs/train/sac_gym_hil_pick_cube.json`（SAC，两组相同超参数与 seed） |
| 在线环境 | `PandaPickCubeGamepad-v0`，10 fps，双相机 128×128，state 18 维 |
| 启动方式 | 手动双进程：先 `lerobot.rl.learner`，后 `lerobot.rl.actor` |
| 成功判据 | gym_hil：夹爪与方块距离 < 5 cm 且方块抬起 > 10 cm |

### 数据

| | A 组（示例） | B 组（我的） |
|---|---|---|
| 来源 | `lilkm/pick_cube_franka_panda_30`（Hub） | `data/curated/d163e86dc4aa__f3795cbc51b5/dataset` |
| episodes / frames | 35 / 929 | 29 / 561 |
| 内容 | 31 个成功 + 4 个失败 episode，未裁剪 | 全部成功；首尾空闲段已裁剪 |
| 上游 | — | 本地录制 30 eps / 666 帧，经 `clean → convert → validate` 得到（失败过滤 30→29，裁剪 666→561） |

B 组 `validation.json` 全部检查通过（`meta_counts` / `features` / `tabular` / `timestamps` / `stats` / `video`）。

### 运行方式

```bash
# 终端 1：learner（先启动）
.venv/bin/python -m lerobot.rl.learner \
  --config_path configs/train/sac_gym_hil_pick_cube.json

# 终端 2：actor
.venv/bin/python -m lerobot.rl.actor \
  --config_path configs/train/sac_gym_hil_pick_cube.json
```

B 组训练前在配置的 `dataset` 块中增加了 `root` 指向本地 curated 数据集
（`dataset.root` 对 learner/actor 同时生效）；A 组使用默认 Hub 路径。

### 结果

指标来自 actor 日志的逐 episode 记录（`Global step N: Episode reward: X`，1.0 表示成功）。

| 指标 | A 组（示例数据） | B 组（我的数据） |
|---|---|---|
| 运行时长（env steps） | 37,851 | 19,459 |
| 总 episodes | 490 | 659 |
| 成功 episodes | 124（25.3%） | 499（75.7%） |
| 首次成功 step | 28,385 | 9,219 |
| 人工干预 | 1 次（约 28.1k，5 个 episode / 263 帧） | 0 次 |
| 28–30k 成功率 | 4.3% | — |
| 30–32k 成功率 | 13.6% | — |
| 32–34k 成功率 | 33.3% | — |
| 34–36k 成功率 | 31.0% | — |
| 36–38k 成功率 | **93.6%（102/109）** | — |
| 8–10k 成功率 | — | 13.6% |
| 10–12k 成功率 | — | 0% |
| 12–14k 成功率 | — | 78.9%（此后稳定在 79–100%） |
| 14–16k / 16–18k / 18–20k | — | 89.6% / 92.5% / **100%** |

实验过程中的手工记录与日志核对：

| 手工记录 | 日志验证 |
|---|---|
| 示例数据 2.8w 没有 reward，人工干预一次 | 0–28k 无成功；28.1k 唯一一次干预；首次成功 28,385 |
| 示例数据 3.2w 逐步 | 30–34k 成功率 13.6% → 33.3% |
| 示例数据 3.6w 稳定 | 36–38k 成功率 93.6% |
| 我的数据 9k 出现 reward | 首次成功 9,219 |
| 我的数据 1.3w 稳定出现 | 12–14k 起稳定在 78.9% 以上 |

### 结论（初步）

> 在本机单次对比中，清洗后的 curated 离线数据让首成功从约 28.4k env steps 提前到约 9.2k，
> 稳定成功率（>90%）从约 36k 提前到约 13–16k，且全程不需要人工干预。

理解：原始数据里失败 episode 和空闲帧占比高，offline buffer 中有效演示密度低；
清洗后只保留成功动作并去掉空闲段，offline + online mixer 更快采到高回报 transition，
策略因此更早发现 reward 并稳定。

### 局限

- 每组只有 1 个 seed，成功率是训练期统计，不是独立评测。
- 两组数据同时存在多个差异（35 vs 29 eps、929 vs 561 帧、成功过滤、空闲裁剪），
  不能单独归因到某一个清洗步骤。
- 两轮均由人工 Ctrl-C 结束（shutdown signal 2），不是收敛后自然停止。
- B 组成功 episode 更短（中位数约 10 步 vs A 组约 18 步），建议抽视频确认不是异常捷径。
- learner 目录下的 `dataset/` 与 `dataset_offline/` 是**保存 checkpoint 时的快照**，
  不是完整训练数据（A 组快照到 355 eps / 34.2k，实际跑到 490 eps / 37.8k）；
  完整逐 episode 时间线以 actor 日志为准。

### 产物

- A 组：`outputs/train/2026-11-18/17-27-51_default/`（learner）、`17-27-53_default/`（actor）
- B 组：`outputs/train/2026-11-18/18-46-33_default/`（learner）、`18-46-34_default/`（actor）
- B 组数据：`data/curated/d163e86dc4aa__f3795cbc51b5/`

### 指标提取方法

```bash
# 每个 episode 一行；reward 1.0 即成功
grep -o "Global step [0-9]*: Episode reward: [0-9.]*" \
  outputs/train/2026-11-18/18-46-34_default/logs/actor_default.log
```

---

## 实验模板

```markdown
## 实验 NNN：<主题>（YYYY-MM-DD）

### 目的
### 环境与配置
### 数据
### 运行方式
### 结果
### 结论（初步）
### 局限
### 产物
```
