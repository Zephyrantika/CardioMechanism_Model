# Phenotype Network V1 Handoff Kit

本目录是交给外部开发者或 AI 编程代理的 V1 执行包。V0 已完成，是不可破坏的比较基线；V1 的目标是加入表型条件化异质图学习，同时保持数据隔离、可解释性和服务器可迁移性。

如果接收方已经拥有 V0，只需使用本目录和根目录的
`V1_IMPLEMENTATION_PLAN.md`，不要重新交付或覆盖 V0 源码。具体接入步骤见
`V0_OWNER_HANDOFF.md`。

## 交付内容

- `AI_EXECUTION_PROMPT.md`：复制给 AI 工具的主任务提示词。
- `DATA_CONTRACT.md`：V0 冻结数据、V1 学习数据和未来临床接口的数据契约。
- `ACCEPTANCE_CHECKLIST.md`：M0-M8 的逐项验收和证据要求。
- `V0_OWNER_HANDOFF.md`：已有 V0 仓库时的版本核对、合并和 AI 协作注意事项。
- 仓库根目录 `AGENTS.md`：工程与生物医学数据规则。
- 仓库根目录 `IMPLEMENTATION_PLAN.md`：V0 已实现方案。
- 仓库根目录 `V1_IMPLEMENTATION_PLAN.md`：V1 的正式执行规格。
- `docs/v0_acceptance.md`、`docs/server_migration.md`：V0 验收结果和服务器迁移约束。

## 开始前

执行者必须在仓库根目录工作，先阅读 `AGENTS.md`、`IMPLEMENTATION_PLAN.md`、`V1_IMPLEMENTATION_PLAN.md` 和本目录全部文件。不得把附件、患者数据、原始数据库下载或本机 `.venv` 放入交付包。

V1 的公共疾病级范围是 V1-A，完成 M0-M8 即可验收。当前没有患者级临床数据，因此 V1-B 只能保留接口和 schema，不得创建模拟患者值，也不得把临床变量加入 V1-A 训练图。

## 推荐交付方式

每个里程碑单独提交一次，提交信息使用：

```text
feat(v1): implement milestone N - short description
```

提交必须包含代码、测试、配置、运行日志摘要、数据/配置哈希和对应的 `docs/v1_milestone_N_acceptance.md`。失败时提交失败报告，不得跳过测试或静默改需求。
