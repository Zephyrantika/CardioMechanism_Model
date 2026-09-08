# 已有 V0 仓库的 V1 接入说明

本包是 V1 的增量执行规范，不包含 V0 源码、数据、输出、虚拟环境或模型权重。接收方应在自己的 V0 仓库中实施，不要用本包覆盖整个项目。

## 1. 核对 V0 基线

参考 V0 提交为：

```text
4a4a508ab2778f68968f6da91003b024249f62dc
Complete interpretable phenotype network V0
```

接收方先运行：

```bash
git status --short
git rev-parse HEAD
python -m pytest -q
python scripts/verify_v0.py --config configs/default.yaml
```

预期单元测试为 `82 passed`，V0 验收为 15/15。若提交不同但测试和验收通过，应记录差异；不要强制回退或覆盖已有修改。

## 2. 接入方式

将交付包解压到临时目录，只把以下内容复制或合并到 V0 仓库：

```text
V1_IMPLEMENTATION_PLAN.md
docs/v1_handoff/
```

`AGENTS.md`、`IMPLEMENTATION_PLAN.md`、`docs/v0_acceptance.md` 和 `docs/server_migration.md` 是参考副本。若接收方仓库已有这些文件，应比较差异，不要直接覆盖。

从 V0 基线创建独立分支：

```bash
git switch -c codex/v1
```

存在同名分支时切换到已有分支，不要删除或重建。V1 新代码必须放在 `src/phenotype_network_v1/`、`scripts/v1/`、`configs/v1/`、`tests/v1/` 和 `outputs/v1/` 等隔离命名空间中。

## 3. AI 协作纪律

每次只把一个 milestone 交给 AI。先要求 AI 阅读规则并列出将修改的文件，完成后人工检查 `git diff --stat`、`git diff`、测试日志和验收文档，再允许进入下一里程碑。

AI 不得执行以下操作：

- 修改或重新生成 V0 五折划分和冻结指标；
- 使用测试标签、外部验证数据或患者结局调参；
- 把未知疾病–基因对当作确认阴性；
- 伪造患者数据、临床结果或模型提升；
- 覆盖 `data/raw/`，提交数据、checkpoint、`.venv` 或生成输出；
- 未核验许可证就复制 Speos、XGDAG 或其他上游代码；
- 一个任务同时实现多个 milestone；
- 测试失败后跳过测试或降低验收标准。

## 4. 当前临床数据状态

当前没有患者级临床数据。V1-A 正常实施 M0-M8；V1-B 只保留 schema 和接口契约，不进入训练。血清尿酸、LDL-C、hs-CRP、IVUS 斑块负荷和最小管腔面积不得被虚构为患者输入。

## 5. 每个里程碑的交付物

每次提交至少包含：实现代码、合成小型测试、配置、运行命令、结构化日志摘要、数据与配置哈希，以及 `docs/v1_milestone_N_acceptance.md`。建议提交格式：

```text
feat(v1): implement milestone N - short description
```

服务器生成的 checkpoint 和完整实验输出存放在受控存储中，通过 manifest、SHA256 和运行 ID 引用，不进入 Git。
