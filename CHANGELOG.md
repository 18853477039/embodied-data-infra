# Changelog

本项目的重要变更记录在此。格式参考 [Keep a Changelog](https://keepachangelog.com/)，
版本号遵循 [Semantic Versioning](https://semver.org/)。

## [0.2.0] - 2026-11-18

### Added

- 统一的 `embodied` CLI：`collect` / `clean` / `convert` / `validate` / `train`。
- `collection` 阶段包装器：配置校验、CLI 覆盖、dry-run、参数透传。
- `training` 阶段包装器：配置校验、`--latest-curated` 自动选数据集、输出目录/任务名/resume 覆盖。
- `configs/` 按阶段拆分：`collect/`、`clean/`、`train/`。
- `docs/`：架构、配置、清洗流水线与运维手册。
- `Makefile`、GitHub Actions CI、pre-commit、ruff/pytest/coverage 配置。
- 采集/训练包装器单元测试，以及统一 CLI 的 dry-run 覆盖。

### Changed

- Python 包从 `embodied_data_clean` 重命名为 `embodied_data_infra`，清洗模块移入 `cleaning/` 子包。
- 清洗配置路径从 `configs/clean_pick_cube.yaml` 改为 `configs/clean/pick_cube.yaml`。
- 录制配置从根目录 `gym_hil_env.json` 移至 `configs/collect/gym_hil_pick_cube.json`。
- 训练配置从 `train/train_gym_hil_env.json` 移至 `configs/train/sac_gym_hil_pick_cube.json`。
- 手柄自检脚本从根目录 `test_gamepad.py` 移至 `scripts/check_gamepad.py`。
- 运行日志归档到 `logs/archive/`；新增目录约定与 gitignore 规则。

### Removed

- 根目录 `example_config/`（空目录）。
- 根目录 `note.md`（内容并入 `docs/operations.md`）。

## [0.1.0] - 2026-11-17

### Added

- 两阶段清洗流水线：`clean`（原始 → 清洗层 + QC）与 `convert`（清洗层 → 标准 LeRobot v3）。
- 确定性 run key、staging + 原子发布、清洗层契约校验、转换后回读校验。
- 基于合成数据的 config/rules/cleaning/roundtrip 测试。
