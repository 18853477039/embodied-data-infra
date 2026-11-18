# Contributing

## 开发环境

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]" --python .venv/bin/python
pre-commit install
```

## 提交前检查

```bash
make format     # ruff check --fix + ruff format
make check      # lint + pytest
```

要求：

- 新增清洗规则：补 `tests/unit/test_rules.py`，并用真实数据 `embodied clean --dry-run` 复核。
- 新增采集/训练配置：放在 `configs/<stage>/`，在 `docs/configuration.md` 补充字段说明。
- 新增 CLI 参数：同步更新 `README.md` 或对应 docs，并补包装器测试。
- 不提交 `data/`、`outputs/`、`logs/`、`.venv/`、凭据或大文件。

## 目录约定

| 类型 | 位置 |
|---|---|
| 可安装代码 | `src/embodied_data_infra/` |
| 人工脚本 | `scripts/` |
| 阶段配置 | `configs/<stage>/` |
| 单元测试 | `tests/unit/` |
| 跨阶段闭环测试 | `tests/integration/` |
| 文档 | `docs/` |

## 提交信息

建议使用 Conventional Commits：

```
feat(cleaning): add gripper jitter rule
fix(training): resolve latest curated dataset by mtime
docs(operations): document HF upload flow
```
