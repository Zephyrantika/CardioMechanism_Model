# V1 交付验收清单

## 里程碑证据

| Milestone | 必须交付 | 最低验收证据 |
| --- | --- | --- |
| M0 | 环境、配置、checkpoint 基础 | CPU 测试通过；服务器记录 Python/PyTorch/PyG/CUDA 和确定性设置 |
| M1 | 每折 query 数据和 PyG 图 | 五份泄漏报告、图哈希、节点/边数量对账 |
| M2 | GCN、R-GCN、HGT、Speos、XGDAG 比较 | 上游 SHA/许可证/smoke log；统一候选集合和评价协议 |
| M3 | 表型条件化 R-GCN 和 RWR 残差 | 五项消融；不同表型查询产生不同分数；确定性重放 |
| M4 | PU bagging、调参和不确定性 | 五成员完成；验证集调参访问日志；抽样审计 |
| M5 | Reactome 超图模块 | 模块稳定性、反 hub 检查、Leiden 对照和拒绝记录 |
| M6 | 关系感知路径解释 | 路径来源/关系/权重；边删除、充分性和完备性结果 |
| M7 | 外部验证和扰动 | GWAS/GTEx/GO/BioGRID 隔离；STRING 700/900 和扰动分析 |
| M8 | 最终评估和发布 | 五折指标、家族 bootstrap、模型卡、迁移验证、完整测试 |

## 最终门槛

- `pytest -q` 和 GPU 标记测试按环境通过或明确跳过。
- 所有 fold leakage report、checkpoint hash、配置 hash 和数据 hash 可复核。
- V1 主模型至少在 3/5 折超过 V0 RWR 的 Recall@10 或 MRR，且提升不能由覆盖率解释。
- 报告度数偏差、扰动稳定性、PU 不确定性和解释保真度。
- 失败的科学门槛必须如实记录为负结果，不得用测试集继续调参。

## 交付前人工检查

确认没有 raw data、患者数据、checkpoint、`.venv`、临时下载和未许可的上游代码进入提交；确认 README、配置示例、运行命令、Git commit 和服务器环境记录齐全。
