# Phenotype Network V0

面向心血管免疫代谢研究的表型驱动异质网络疾病基因排序与机制模块发现原型。

本项目第一版用于验证以下核心假设：

> 将疾病 HPO 表型转换为网络种子信号，在由表型、基因、蛋白互作和 Reactome 通路构成的异质网络中传播，能否比纯表型语义相似度更准确、更稳定地恢复疾病相关基因，并形成可解释的机制模块与传播路径。

第一版强调：

- 可解释；
- 可复现；
- 防止数据泄漏；
- 可审计的数据清洗；
- 稀疏矩阵传播；
- 模块稳定性；
- 路径级解释。

第一版不使用深度学习、图神经网络、大语言模型或 GPU。

> 当前状态：V0 已完成。Milestone 0-13 均已实现并通过验收；完整测试为 `82 passed`，跨里程碑验证为 15/15 通过。

---

## 1. 当前功能范围

V0 计划实现：

1. HPO 本体解析；
2. 心血管疾病–表型、疾病–基因基准集及临床特征注册表；
3. STRING 主 PPI 与可选 BioGRID 敏感性网络清洗；
4. Reactome 免疫代谢通路与可选 GO 功能注释清洗；
5. MONDO 疾病家族不重叠五折划分；
6. 标签泄漏检查；
7. Resnik / BMA 语义基线；
8. 异质网络 Random Walk with Restart；
9. 节点度数偏差校正；
10. 冻结 GWAS/GTEx/GO 外部验证与 Leiden / DIAMOnD 模块识别；
11. 模块扰动稳定性分析；
12. 表型到候选基因的解释路径；
13. 冠状动脉粥样硬化、心肌梗死和心力衰竭案例及临床特征桥接。

详细设计见：

- `AGENTS.md`
- `IMPLEMENTATION_PLAN.md`
- `docs/milestone_flow.md`
- `docs/server_migration.md`

---

## 2. 项目结构

```text
phenotype-network-v0/
├── AGENTS.md
├── IMPLEMENTATION_PLAN.md
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── data.yaml
│   ├── graph.yaml
│   └── experiment.yaml
│
├── data/
│   ├── raw/
│   │   ├── hpo/
│   │   ├── mondo/
│   │   ├── string/
│   │   ├── reactome/
│   │   └── gene_mapping/
│   ├── interim/
│   ├── processed/
│   ├── folds/
│   └── graphs/
│
├── src/
│   └── phenotype_network_v0/
│
├── scripts/
├── tests/
│   └── fixtures/
│
├── outputs/
│   ├── qc/
│   ├── rankings/
│   ├── modules/
│   ├── paths/
│   ├── metrics/
│   └── cases/
│
└── docs/
```

---

## 3. 环境要求

推荐环境：

```text
Python 3.11
```

Milestone 0 基础依赖：

```text
numpy
PyYAML
```

可选依赖按用途分组：`data` 提供 pandas、PyArrow、Pronto 和 tqdm；`graph`
提供 SciPy、scikit-learn、NetworkX、igraph、Leidenalg 和 statsmodels；`dev`
提供 pytest 与 pytest-cov；`all` 包含 `data` 和 `graph`。

---

## 4. 安装

### 4.1 克隆或进入项目目录

```bash
cd phenotype-network-v0
```

### 4.2 创建虚拟环境

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4.3 安装项目

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

按后续任务安装可选功能：

```bash
pip install -e ".[data,dev]"
pip install -e ".[graph,dev]"
pip install -e ".[all,dev]"
```

### 4.4 配置

统一配置入口为：

```text
configs/default.yaml
```

该文件按顺序包含 `data.yaml`、`graph.yaml` 和 `experiment.yaml`。配置路径相对于
声明它的配置文件解析，不依赖当前工作目录。数据和输出目录可通过环境变量覆盖，
例如 `PHENOTYPE_NETWORK_RAW_DIR` 和 `PHENOTYPE_NETWORK_OUTPUTS_DIR`。

## 5. 运行测试

运行全部测试：

```bash
pytest -q
```

运行某个测试目录：

```bash
pytest tests/data -q
```

运行某个具体测试：

```bash
pytest tests/models/test_rwr.py -q
```

所有自动测试必须只依赖 `tests/fixtures/` 中的小型人工数据，不得依赖完整外部数据库。

---

## 6. 数据放置

原始数据库文件由用户手动下载并放入 `data/raw/`。

原始文件不得被脚本修改或覆盖。
`data/raw/`、中间数据、处理结果和 `outputs/` 默认被 Git 忽略；仅目录占位文件进入版本库。

### 6.1 HPO

```text
data/raw/hpo/hp.obo
data/raw/hpo/phenotype.hpoa
data/raw/hpo/genes_to_disease.txt
data/raw/hpo/genes_to_phenotype.txt
```

### 6.2 MONDO

```text
data/raw/mondo/mondo.obo
```

### 6.3 STRING

```text
data/raw/string/9606.protein.links.full.txt.gz
data/raw/string/9606.protein.aliases.txt.gz
```

### 6.4 Reactome

```text
data/raw/reactome/NCBI2Reactome_All_Levels.txt
data/raw/reactome/ReactomePathwaysRelation.txt
data/raw/reactome/ReactomePathways.txt
```

### 6.5 基因映射

推荐放置 NCBI Gene、HGNC、Ensembl 或 UniProt 的标识符映射文件：

```text
data/raw/gene_mapping/
```

实际文件名通过 `configs/data.yaml` 配置。

---

## 7. 数据目录规则

```text
data/raw/
```

保存原始下载文件，永远不修改。

```text
data/interim/
```

保存中间解析、映射、未解决 ID 和审计文件。

```text
data/processed/
```

保存完成标准化、去重和质量控制的数据表。

```text
data/folds/
```

保存 disease-disjoint 交叉验证划分及 fold 特异的 HPO–Gene 边。

```text
data/graphs/
```

保存节点索引、关系稀疏矩阵和转移矩阵。

---

## 8. 计划执行顺序

完整 V0 计划按以下顺序执行：

```bash
python scripts/01_parse_hpo.py --config configs/default.yaml
python scripts/02_build_cardiovascular_benchmark.py --config configs/default.yaml
python scripts/03_clean_ppi.py --config configs/default.yaml
python scripts/04_clean_pathways.py --config configs/default.yaml
python scripts/05_make_disease_folds.py --config configs/default.yaml
python scripts/06_build_graph.py --config configs/default.yaml
python scripts/07_run_semantic_baseline.py --config configs/default.yaml
python scripts/08_run_rwr.py --config configs/default.yaml
python scripts/09_calibrate_scores.py --config configs/default.yaml
python scripts/10_prepare_external_validation.py --config configs/default.yaml
python scripts/10_run_ablations.py --config configs/default.yaml
python scripts/10_evaluate_and_validate.py --config configs/default.yaml
python scripts/11_detect_modules.py --config configs/default.yaml
python scripts/12_build_explanations.py --config configs/default.yaml
python scripts/13_build_case_reports.py --config configs/default.yaml
```

`01_parse_hpo.py` 至 `13_build_case_reports.py` 均已实现；正式运行状态与验收结果见 `docs/milestone_*_acceptance.md`。

---

## 9. 里程碑

### Milestone 0：工程初始化

目标：

- 创建 Python 3.11 项目；
- 建立 `src` 布局；
- 创建配置加载、日志和随机种子工具；
- 创建测试框架；
- 创建数据目录；
- 完成基础 README 和 `.gitignore`。

### Milestone 1：HPO 解析

输入：

```text
data/raw/hpo/hp.obo
```

输出：

```text
data/processed/hpo_nodes.parquet
data/processed/hpo_edges.parquet
data/interim/obsolete_hpo_mapping.parquet
data/interim/unresolved_hpo_terms.tsv
outputs/qc/hpo_qc.json
```

### Milestone 2：心血管疾病与表型基准集

输出：

```text
data/processed/cardiovascular_disease_phenotypes.parquet
data/processed/cardiovascular_disease_genes.parquet
data/processed/cardiovascular_disease_samples.parquet
data/processed/cardiovascular_negative_phenotypes.parquet
data/processed/clinical_feature_registry.parquet
data/interim/conflicting_phenotype_annotations.tsv
outputs/qc/cardiovascular_benchmark_qc.json
```

### Milestone 3：心血管 PPI 网络

输出：

```text
data/processed/string_gene_edges_400.parquet
data/processed/string_gene_edges_700.parquet
data/processed/string_gene_edges_900.parquet
outputs/qc/ppi_qc.json
```

### Milestone 4：免疫代谢通路与功能注释

输出：

```text
data/processed/reactome_gene_pathway.parquet
data/processed/reactome_pathway_hierarchy.parquet
data/processed/reactome_pathways.parquet
data/processed/immunometabolic_pathway_registry.parquet
outputs/qc/pathway_qc.json
```

### Milestone 5：心血管疾病家族不重叠划分

每个 fold 输出：

```text
data/folds/fold_0/
├── train_diseases.txt
├── val_diseases.txt
├── test_diseases.txt
├── train_hpo_ic.parquet
├── phenotype_gene_edges.parquet
├── candidate_gene_universe.parquet
└── leakage_report.json
```

### Milestone 6：心血管异质图

输出：

```text
data/graphs/fold_0/node_map.parquet
data/graphs/fold_0/*.npz
data/graphs/fold_0/graph_qc.json
```

### Milestone 7：心血管表型语义基线

每个 fold 输出三种 Resnik/BMA 聚合方法的完整候选基因排序：

```text
outputs/rankings/fold_0/semantic/rankings.parquet
outputs/rankings/fold_0/semantic/semantic_qc.json
```

### Milestone 8：表型驱动 RWR

每个 fold 输出验证集参数选择、测试集完整排名和收敛审计：

```text
outputs/rankings/fold_0/rwr/rankings.parquet
outputs/rankings/fold_0/rwr/validation_metrics.json
outputs/rankings/fold_0/rwr/rwr_qc.json
```

### Milestone 9：网络偏差校正

每个 fold 使用仅来自训练疾病的 1,000 次匹配 null bootstrap，输出完整校正排名、匹配记录、表型特征审计和度数相关性：

```text
outputs/rankings/fold_0/calibrated/rankings.parquet
outputs/rankings/fold_0/calibrated/null_query_matches.parquet
outputs/rankings/fold_0/calibrated/profile_audit.parquet
outputs/rankings/fold_0/calibrated/bias_correlations.parquet
outputs/rankings/fold_0/calibrated/calibration_qc.json
```

### Milestone 10–13

依次完成正式评价、消融及冻结外部验证，机制模块发现，路径解释，以及三个预注册心血管案例与临床特征桥接。详见 `IMPLEMENTATION_PLAN.md`。

---

## 10. 数据泄漏原则

本项目最重要的规则之一是：

> 测试疾病及其已知疾病–基因标签不能进入训练传播图。

具体要求：

- train、validation、test 疾病必须互斥；
- HPO–Gene 边只能由训练疾病构建；
- 测试疾病–基因关系不得存在于背景图；
- 测试标签不能进入节点特征；
- 等价的重复数据库关系也必须排除；
- 外部验证数据不得参与训练。

每个 fold 必须生成：

```text
leakage_report.json
```

所有关键检查通过后才能运行模型。

---

## 11. 核心模型

### 11.1 表型种子

对疾病 \(d\) 的表型 \(\phi\)：

\[
w_{d,\phi}
=
frequency(d,\phi)\times IC(\phi)
\]

归一化：

\[
s_d^{(0)}[\phi]
=
\frac{w_{d,\phi}}
{\sum_{\phi'\in\Phi_d}w_{d,\phi'}}
\]

Gene 和 Pathway 节点初始值为 0。

### 11.2 Random Walk with Restart

使用列向量：

\[
s^{(t+1)}
=
\alpha s^{(0)}
+
(1-\alpha)P^\top s^{(t)}
\]

默认参数：

```text
restart_probability = 0.30
tolerance = 1e-10
max_iterations = 500
```

### 11.3 偏差校正

对随机匹配表型查询构建每个基因的背景分布：

\[
Z_g
=
\frac{s_g^{real}-\mu_g}
{\sigma_g+\epsilon}
\]

最终同时保留：

- raw score；
- null mean；
- null standard deviation；
- Z-score；
- empirical percentile；
- corrected rank。

---

## 12. 评价指标

主要排序指标：

```text
Recall@1
Recall@5
Recall@10
Recall@20
MRR
Average Precision
AUPRC
```

模块指标：

```text
模块大小
内部边密度
已知疾病基因富集
功能富集
扰动稳定性
```

偏差指标：

```text
Spearman(raw score, gene degree)
Spearman(corrected score, gene degree)
```

解释指标：

```text
路径长度
路径证据覆盖
路径删除后的目标分数变化
路径重复稳定性
```

---

## 13. 质量控制

每个数据脚本必须生成 QC JSON，至少包含：

```text
input_records
output_records
removed_records
duplicate_records
unmapped_records
obsolete_records
replaced_records
unresolved_records
```

图构建还应报告：

```text
nodes_by_type
edges_by_relation
isolated_nodes
connected_components
largest_component_fraction
degree_summary
matrix_shape
matrix_nnz
row_sum_error
```

任何无法映射或删除的数据都必须可追踪。

---

## 14. 编码规范

- 使用 Python 3.11；
- 公共函数必须有类型注解；
- 公共函数必须有简洁 docstring；
- 使用 `pathlib.Path`；
- 禁止硬编码绝对路径；
- 使用结构化日志，不使用零散 `print`；
- 所有随机操作必须接受 seed；
- 处理后表格优先使用 Parquet；
- 原始文件不可覆盖；
- 不静默丢弃异常数据；
- 不创建无法运行的空 TODO 模块；
- 不提前实现后续里程碑。

---

## 15. Git 工作流

初始化：

```bash
git init
git add .
git commit -m "Initial project specification"
```

每完成一个里程碑：

```bash
git status
git diff
pytest -q
git add .
git commit -m "Complete milestone X"
```

建议每个 Codex 任务只完成一个里程碑，并在合并前审查全部 diff。

---

## 16. Codex 使用建议

每次任务开头要求 Codex 阅读：

```text
AGENTS.md
IMPLEMENTATION_PLAN.md
README.md
```

推荐提示：

```text
本次只实现当前 Milestone，不要提前实现后续模块。
完成后运行目标测试和 pytest -q。
如果测试失败，修复实现，不要删除测试或降低断言。
最后报告修改文件、运行命令、测试结果和遗留限制。
```

不要一次要求 Codex 完成整个项目。

---

## 17. 当前限制

V0 不是临床诊断系统，也不是因果推断系统。

当前版本不会：

- 处理患者级 IVUS 原始影像；
- 直接预测临床结局；
- 给出治疗建议；
- 证明某个基因是疾病因果基因；
- 直接整合全部代谢组和单细胞数据；
- 使用深度学习。

模型输出应被解释为：

- 候选疾病基因；
- 候选机制模块；
- 可追溯传播路径；
- 可供外部验证的机制假设。

---

## 18. 后续升级

V0 验证有效后，再逐步升级：

```text
V1 将固定关系权重 RWR 逐步升级为表型条件化的可学习长距离传播；
APPNP 只是一个可能的基础实现，不是 V0 依赖。
```

每次升级必须保持原有数据划分和评价框架，并与上一版本进行消融比较。

---

## 19. License

在正式公开代码前，需要根据项目归属、数据库许可和合作协议确定许可证。

在未明确之前，不建议自动添加 MIT、Apache-2.0 或 GPL 等许可证。

---

## 20. Citation

项目尚处于开发阶段，目前没有正式引用格式。

后续发布论文或软件版本后，在本节补充 BibTeX。
