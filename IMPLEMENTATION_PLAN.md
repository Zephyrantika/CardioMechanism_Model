# IMPLEMENTATION_PLAN.md

## 1. 项目名称

**Phenotype Network V0：表型驱动异质网络疾病基因排序与机制模块发现**

---

## 2. 项目目标

本项目第一版旨在实现一个透明、可复现、可审计的计算原型：

> 给定一个疾病的 HPO 表型集合，将表型转换为图上的初始信号，在由 HPO、基因、PPI 和 Reactome 通路构成的异质网络中传播，输出候选基因排序、机制模块、解释路径以及稳定性与偏差校正结果。

第一版重点验证以下问题：

1. 表型语义信息能否恢复已知疾病基因；
2. 异质网络传播是否优于纯表型语义相似度；
3. 网络传播结果是否受到高连接基因偏差影响；
4. 高分基因是否能够形成稳定、功能一致的机制模块；
5. 是否能够输出从表型到候选基因的可追溯解释路径。

第一版必须保持可解释性，不使用深度学习、图神经网络、大语言模型或 GPU。

---

## 3. 第一版范围

### 3.1 输入

每个疾病样本包含：

- 规范化疾病 ID；
- 疾病名称；
- HPO 表型集合；
- 表型频率或权重；
- 已知相关基因集合；
- 证据来源。

形式化表示：

\[
\Phi_d = \{\phi_1,\phi_2,\ldots,\phi_m\}
\]

其中每个表型具有初始权重：

\[
w_{d,\phi}
=
frequency(d,\phi)\times IC(\phi)
\]

归一化后得到：

\[
\hat w_{d,\phi}
=
\frac{w_{d,\phi}}
{\sum_{\phi'\in\Phi_d}w_{d,\phi'}}
\]

### 3.2 输出

每个查询疾病应输出：

1. 全候选基因排序；
2. Top-K 候选基因；
3. 原始传播得分；
4. 偏差校正得分；
5. 机制模块；
6. 模块稳定性；
7. 表型到候选基因的解释路径；
8. 评价指标；
9. 数据和模型质量控制报告。

### 3.3 节点类型

第一版仅包含三种图节点：

| 节点类型 | 内部类型 | 推荐规范 ID |
|---|---|---|
| HPO 表型 | `phenotype` | HPO ID |
| 人类基因 | `gene` | NCBI Gene ID |
| Reactome 通路 | `pathway` | Reactome Stable ID |

疾病作为查询样本和监督标签单位，不作为第一版传播图中的节点。

### 3.4 关系类型

| 起点 | 关系 | 终点 | 传播方向 |
|---|---|---|---|
| HPO | `is_a` | HPO | 双向 |
| HPO | `associated_with` | Gene | 双向 |
| Gene | `interacts_with` | Gene | 无向 |
| Gene | `participates_in` | Pathway | 双向 |
| Pathway | `part_of` | Pathway | 双向，可选 |

### 3.5 第一版明确不做

- 患者级连续临床数值建模；
- IVUS 原始影像建模；
- 代谢物节点；
- 免疫细胞节点；
- 单细胞训练；
- 知识图谱嵌入；
- 图神经网络；
- HGT、R-GCN、APPNP；
- 大语言模型；
- 因果结论；
- 自动化生物学结论生成。

---

## 4. 技术原则

### 4.1 可复现

所有实验必须记录：

- Git commit；
- Python 版本；
- 依赖版本；
- 数据库版本；
- 下载日期；
- 文件 SHA256；
- 配置文件；
- 随机种子；
- 运行命令；
- 输出路径；
- 评价结果。

### 4.2 防止数据泄漏

必须物理隔离：

- 背景传播图；
- 训练标签；
- 验证标签；
- 测试标签；
- 外部验证数据。

对每个交叉验证 fold，HPO–Gene 边只能由训练疾病构建。

测试疾病及其已知基因不得通过任何等价关系进入训练传播图。

### 4.3 原始数据不可修改

`data/raw/` 下的文件必须只读。

所有清洗、映射、过滤和聚合结果输出到：

- `data/interim/`
- `data/processed/`
- `data/folds/`
- `data/graphs/`

### 4.4 不静默丢弃数据

所有无法映射、格式异常、ID 冲突、废弃术语和重复记录都必须写入审计表或 QC 报告。

### 4.5 使用稀疏矩阵

全图传播必须使用 `scipy.sparse`。

NetworkX 只能用于小规模解释子图和路径搜索，不能用于全图随机游走。

---

## 5. 推荐开发环境

### 5.1 Python

```text
Python 3.11
```

### 5.2 核心依赖

```text
pandas
pyarrow
numpy
scipy
scikit-learn
networkx
python-igraph
leidenalg
pronto
pyyaml
tqdm
statsmodels
pytest
```

### 5.3 可选依赖

```text
obonet
rich
typer
joblib
matplotlib
```

---

## 6. 项目目录

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
│       ├── __init__.py
│       ├── config.py
│       ├── logging_utils.py
│       ├── random_utils.py
│       ├── schemas.py
│       │
│       ├── data/
│       │   ├── hpo_parser.py
│       │   ├── mondo_parser.py
│       │   ├── disease_benchmark.py
│       │   ├── string_parser.py
│       │   ├── reactome_parser.py
│       │   ├── gene_mapping.py
│       │   └── qc.py
│       │
│       ├── graph/
│       │   ├── node_index.py
│       │   ├── relation_matrix.py
│       │   ├── graph_builder.py
│       │   └── transition_matrix.py
│       │
│       ├── models/
│       │   ├── semantic_similarity.py
│       │   ├── semantic_baseline.py
│       │   ├── rwr.py
│       │   └── calibration.py
│       │
│       ├── modules/
│       │   ├── leiden.py
│       │   ├── diamond.py
│       │   └── stability.py
│       │
│       ├── explanation/
│       │   ├── subgraph.py
│       │   ├── path_search.py
│       │   └── path_ablation.py
│       │
│       └── evaluation/
│           ├── ranking_metrics.py
│           ├── module_metrics.py
│           ├── bias_metrics.py
│           └── bootstrap.py
│
├── scripts/
│   ├── 00_download_data.py
│   ├── 01_parse_hpo.py
│   ├── 02_build_cardiovascular_benchmark.py
│   ├── 03_clean_ppi.py
│   ├── 04_clean_pathways.py
│   ├── 05_make_disease_folds.py
│   ├── 06_build_graph.py
│   ├── 07_run_semantic_baseline.py
│   ├── 08_run_rwr.py
│   ├── 09_calibrate_scores.py
│   ├── 10_evaluate_and_validate.py
│   ├── 11_detect_modules.py
│   ├── 12_extract_paths.py
│   └── 13_run_cardiovascular_cases.py
│
├── tests/
│   ├── fixtures/
│   ├── data/
│   ├── graph/
│   ├── models/
│   ├── modules/
│   ├── explanation/
│   └── evaluation/
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

## 7. 规范数据格式

### 7.1 节点表

```text
node_index
canonical_id
node_type
name
source_databases
is_obsolete
feature_available
```

### 7.2 边表

```text
source_id
target_id
relation_type
weight
confidence
evidence_type
source_databases
publication_count
is_directed
split
```

### 7.3 疾病样本表

```text
disease_id
disease_name
phenotype_ids
phenotype_weights
positive_gene_ids
phenotype_count
gene_count
source
```

### 7.4 ID 映射表

```text
raw_id
raw_name
raw_type
source_database
canonical_id
canonical_name
canonical_type
mapping_method
mapping_confidence
is_obsolete
```

---

## 8. 配置文件设计

### 8.1 `configs/data.yaml`

```yaml
paths:
  raw: "data/raw"
  interim: "data/interim"
  processed: "data/processed"
  folds: "data/folds"
  graphs: "data/graphs"
  outputs: "outputs"

hpo:
  ontology_file: "data/raw/hpo/hp.obo"
  annotation_file: "data/raw/hpo/phenotype.hpoa"
  gene_disease_file: "data/raw/hpo/genes_to_disease.txt"
  gene_phenotype_file: "data/raw/hpo/genes_to_phenotype.txt"

mondo:
  ontology_file: "data/raw/mondo/mondo.obo"

string:
  species_taxonomy_id: 9606
  links_file: "data/raw/string/9606.protein.links.full.txt.gz"
  aliases_file: "data/raw/string/9606.protein.aliases.txt.gz"
  thresholds: [400, 700, 900]

reactome:
  gene_pathway_file: "data/raw/reactome/NCBI2Reactome_All_Levels.txt"
  pathway_hierarchy_file: "data/raw/reactome/ReactomePathwaysRelation.txt"
  pathway_metadata_file: "data/raw/reactome/ReactomePathways.txt"
  species_name: "Homo sapiens"

identifiers:
  canonical_phenotype: "HPO"
  canonical_disease: "MONDO"
  canonical_gene: "NCBI_GENE"
  canonical_pathway: "REACTOME"
```

统一入口 `configs/default.yaml` 设置相对于自身位置解析的项目根目录，并按顺序合并组件配置：

```yaml
includes:
  - data.yaml
  - graph.yaml
  - experiment.yaml

project_root: ".."
```

### 8.2 `configs/graph.yaml`

```yaml
relation_budget:
  phenotype:
    phenotype_is_a_phenotype: 0.40
    phenotype_associated_with_gene: 0.60

  gene:
    gene_associated_with_phenotype: 0.15
    gene_interacts_with_gene: 0.60
    gene_participates_in_pathway: 0.25

  pathway:
    pathway_contains_gene: 0.70
    pathway_part_of_pathway: 0.30

rwr:
  restart_probability: 0.30
  tolerance: 1.0e-10
  max_iterations: 500

pathway:
  minimum_genes: 5
  maximum_genes: 500
```

### 8.3 `configs/experiment.yaml`

```yaml
random_seed: 42
number_of_folds: 5

benchmark:
  minimum_phenotypes_per_disease: 3
  minimum_genes_per_disease: 1

folds:
  disease_disjoint: true
  train_folds: 3
  validation_folds: 1
  test_folds: 1

semantic_baseline:
  aggregation_methods:
    - max
    - top5_mean
    - weighted_sum

calibration:
  development_null_queries: 100
  final_null_queries: 1000
  matched_pool_size: 25
  matching_temperature: 0.5
  zero_variance_tolerance: 1.0e-12
  rwr_batch_size: 64

module_detection:
  candidate_gene_count: 200
  minimum_module_size: 5
  maximum_module_size: 100
  perturbation_runs: 50
  minimum_consensus_frequency: 0.60

explanation:
  maximum_path_length: 4
  paths_per_gene: 3
  target_gene_count: 20
```

---

# 9. 里程碑总览

| Milestone | 名称 | 主要产物 |
|---|---|---|
| 0 | 工程初始化 | 包结构、配置、测试框架 |
| 1 | HPO 本体解析 | 表型节点、层级边、废弃术语审计 |
| 2 | 心血管疾病与表型基准集 | 疾病级样本、临床特征注册表 |
| 3 | 心血管 PPI 网络 | STRING 主网络、BioGRID 敏感性网络 |
| 4 | 免疫代谢通路与功能注释 | Reactome 主图、GO 外部注释 |
| 5 | 心血管疾病家族不重叠划分 | 五折数据、fold 特异 IC、泄漏检查 |
| 6 | 心血管异质图构建 | 节点索引、稀疏关系与转移矩阵 |
| 7 | 心血管表型语义基线 | fold 特异 Resnik/BMA 排序 |
| 8 | 表型驱动 RWR | 原始传播排序与收敛审计 |
| 9 | 网络偏差校正 | 匹配 null queries、校正排序 |
| 10 | 评价、消融与外部验证 | 指标、消融、冻结 GWAS/GTEx/GO 验证 |
| 11 | 心血管免疫代谢模块 | Leiden、DIAMOnD、稳定性与富集 |
| 12 | 证据约束解释路径 | Top 路径、来源证据与删除实验 |
| 13 | 心血管案例与临床桥接 | 三个预注册案例及临床特征映射 |

---

# 10. Milestone 0：工程初始化

## 10.1 目标

建立可安装、可测试、可配置的 Python 项目。

## 10.2 实现内容

1. 创建 `pyproject.toml`；
2. 使用 Python 3.11；
3. 创建 `src` 布局；
4. 创建配置加载器；
5. 创建日志初始化函数；
6. 创建随机种子函数；
7. 创建数据目录；
8. 创建 `.gitignore`；
9. 创建 `tests/fixtures/`；
10. 更新 README；
11. 运行全部测试。

## 10.3 输出

```text
pyproject.toml
configs/data.yaml
configs/graph.yaml
configs/experiment.yaml
src/phenotype_network_v0/
tests/
README.md
.gitignore
```

## 10.4 验收标准

- 包可以导入；
- YAML 可以加载；
- 随机种子具有确定性；
- 必需目录存在；
- `pytest -q` 全部通过；
- 不下载真实数据库；
- 不提前实现后续算法。

---

# 11. Milestone 1：HPO 本体解析

## 11.1 输入

```text
data/raw/hpo/hp.obo
```

## 11.2 输出

```text
data/processed/hpo_nodes.parquet
data/processed/hpo_edges.parquet
data/interim/obsolete_hpo_mapping.parquet
data/interim/unresolved_hpo_terms.tsv
outputs/qc/hpo_qc.json
```

## 11.3 解析字段

节点：

```text
hpo_id
hpo_name
definition
synonyms
is_obsolete
replaced_by
```

边：

```text
child_hpo_id
parent_hpo_id
relation_type
```

## 11.4 处理规则

1. 使用 `pronto` 解析；
2. 只提取 HPO 节点；
3. 只提取 `is_a`；
4. 有唯一 `replaced_by` 的废弃节点自动替换；
5. 无唯一替代节点写入审计表；
6. 原始 ID 必须保留；
7. 不静默删除异常记录。

## 11.5 测试 fixture

`tests/fixtures/hpo_small.obo` 应包含：

- 正常节点；
- 父子关系；
- obsolete 节点；
- `replaced_by`；
- 无法解决的 obsolete 节点。

## 11.6 验收标准

- 所有边两端节点存在；
- 最终节点表无未处理废弃术语；
- 无重复节点；
- 无重复边；
- QC 报告包含输入、输出、替换和 unresolved 数量；
- `pytest -q` 全部通过。

---

# 12. Milestone 2：心血管疾病与表型基准集

## 12.1 定位

构建面向心血管免疫代谢研究的疾病级公开数据基准。V0 的监督单位是疾病，不是患者；首要案例为冠状动脉粥样硬化、心肌梗死和心力衰竭，同时纳入满足质量门槛的更广泛心血管疾病，以支持无泄漏交叉验证。

## 12.2 输入

```text
data/raw/hpo/phenotype.hpoa
data/raw/hpo/genes_to_disease.txt
data/raw/mondo/mondo.obo
data/processed/hpo_nodes.parquet
data/interim/obsolete_hpo_mapping.parquet
```

可选补充来源必须单独标记，不得覆盖主来源：

```text
data/raw/monarch/
data/raw/gene_mapping/
```

## 12.3 心血管范围

- 使用预先登记的 MONDO 心血管疾病根节点、后代关系和明确 xref 建立范围；
- 禁止通过模糊名称匹配自动纳入疾病；
- 保留 `cardiovascular_inclusion_reason`、根疾病和疾病家族；
- 三个核心案例预先标记，但不得因此获得额外训练边；
- 非心血管疾病只可作为预先定义的负向/敏感性参照，不作为确认负例。

## 12.4 临床特征桥接

建立独立注册表记录血清尿酸、LDL-C、hs-CRP、IVUS 斑块负荷和最小管腔面积等特征：

```text
feature_id
feature_name
feature_type
canonical_system
canonical_id
unit
value_domain
role
source
```

有可靠 HPO 或 LOINC 对应时记录映射；没有可靠对应时保留本地规范 ID，禁止强行映射为 HPO。V0 不使用患者级数值训练，注册表仅用于案例解释和未来临床队列扩展。

## 12.5 输出

```text
data/processed/cardiovascular_disease_phenotypes.parquet
data/processed/cardiovascular_disease_genes.parquet
data/processed/cardiovascular_disease_samples.parquet
data/processed/cardiovascular_negative_phenotypes.parquet
data/processed/clinical_feature_registry.parquet
data/interim/cardiovascular_disease_mapping.parquet
data/interim/unresolved_cardiovascular_diseases.tsv
data/interim/unresolved_genes.tsv
data/interim/unresolved_frequencies.tsv
data/interim/conflicting_phenotype_annotations.tsv
outputs/qc/cardiovascular_benchmark_qc.json
```

## 12.6 清洗与验收

- 仅保留人类疾病，阳性与否定表型分离；
- HPO obsolete ID 映射到当前 ID，疾病优先使用可靠 MONDO ID，基因使用 NCBI Gene ID；
- 支持百分数、比例、范围、HPO 频率术语和缺失频率；无法解析的非空值进入审计；
- 每个纳入疾病至少 3 个阳性表型、1 个相关基因；
- 报告疾病家族、核心案例覆盖、表型/基因覆盖和所有映射失败；
- 人工复核全部核心案例及至少 5 个其他心血管疾病；
- 不计算最终 IC，fold 特异 IC 在 Milestone 5 生成。

---

# 13. Milestone 3：心血管 PPI 网络清洗

## 13.1 输入与角色

```text
data/raw/string/9606.protein.links.full.txt.gz
data/raw/string/9606.protein.aliases.txt.gz
```

STRING 是主传播 PPI。BioGRID 可作为独立敏感性网络，不能与 STRING 静默合并：

```text
data/raw/biogrid/
```

## 13.2 输出

```text
data/processed/string_gene_edges_400.parquet
data/processed/string_gene_edges_700.parquet
data/processed/string_gene_edges_900.parquet
data/processed/biogrid_gene_edges.parquet              # 可选
data/interim/unresolved_string_proteins.tsv
data/interim/unresolved_biogrid_identifiers.tsv        # 可选
outputs/qc/ppi_qc.json
```

## 13.3 规则与验收

- 仅保留人类数据，全部映射到 NCBI Gene ID；一对多且不可消歧的映射进入审计；
- 同一基因的蛋白异构体合并，删除自环，统一无向边顺序；
- 重复 PPI 的各证据通道取最大值，避免重复记录放大；
- STRING 生成 400、700、900 三套网络，主分析预设 700；
- 保留原始蛋白 ID、来源、证据通道和版本；
- 报告候选基因覆盖、核心案例阳性基因覆盖、最大连通分量和高连接节点；
- BioGRID 仅用于网络来源敏感性实验，不参与主模型调参。

---

# 14. Milestone 4：免疫代谢通路与功能注释

## 14.1 输入

```text
data/raw/reactome/NCBI2Reactome_All_Levels.txt
data/raw/reactome/ReactomePathwaysRelation.txt
data/raw/reactome/ReactomePathways.txt
data/raw/go/                              # 可选功能验证
```

## 14.2 输出

```text
data/processed/reactome_gene_pathway.parquet
data/processed/reactome_pathway_hierarchy.parquet
data/processed/reactome_pathways.parquet
data/processed/immunometabolic_pathway_registry.parquet
data/processed/go_gene_annotations.parquet              # 可选
data/interim/excluded_reactome_pathways.tsv
outputs/qc/pathway_qc.json
```

## 14.3 规则与验收

- 仅保留 Homo sapiens，基因为 NCBI Gene ID，通路为 Reactome Stable ID；
- 通路名称和物种来自 `ReactomePathways.txt`，不能从层级文件猜测；
- 主图保留含 5–500 个基因的通路，排除项全部审计；
- 基因–通路权重为 `1 / sqrt(|G_p|)`；
- 免疫、炎症、脂质代谢、嘌呤/尿酸代谢、氧化应激和心肌能量代谢标签必须由可追溯通路 ID 规则产生；
- GO 仅用于功能富集/外部注释，默认不加入主传播图；
- 人工复核每类至少 3 条代表通路并报告基因覆盖。

---

# 15. Milestone 5：心血管疾病家族不重叠划分

## 15.1 划分设计

先按 MONDO 等价关系、父子关系和预定义疾病家族生成不可拆分 group，再进行固定五折划分。对第 `k` 次实验：

```text
test = fold k
validation = fold (k + 1) mod 5
train = 其余 3 folds
```

若合格疾病或疾病家族数量不足以支撑五折，流程必须失败并报告原因，不得通过拆分同一家族或复制样本凑足 fold。

## 15.2 fold 特异产物

```text
data/folds/fold_0/train_diseases.txt
data/folds/fold_0/val_diseases.txt
data/folds/fold_0/test_diseases.txt
data/folds/fold_0/train_hpo_ic.parquet
data/folds/fold_0/phenotype_gene_edges.parquet
data/folds/fold_0/candidate_gene_universe.parquet
data/folds/fold_0/leakage_report.json
```

## 15.3 关键规则

- IC 仅由训练疾病的表型注释计算；global IC 仅用于敏感性实验；
- HPO–Gene 边仅由训练疾病标签派生，并保留贡献疾病数和来源；
- 候选基因全集是清洗后 STRING 与 Reactome 基因节点的并集，测试标签不能补充候选集；
- 同一家族、等价疾病和可直接还原答案的重复关系不得跨集合；
- 核心案例可作为预先指定测试病例，但不能因此改变训练图；
- 报告 candidate coverage 和不可评价阳性基因。

## 15.4 验收

所有 fold 的疾病及疾病家族互斥；测试/验证标签未进入 HPO–Gene 边、节点特征或调参过程；每个 `leakage_report.json` 的强制检查全部通过后才能建图。

---

# 16. Milestone 6：心血管异质图构建

## 16.1 图边界

V0 主图节点固定为 HPO、Gene、Reactome Pathway。临床连续指标、代谢物、免疫细胞和 IVUS 图像不在 V0 主图中；完整 HPO 层级与全局 PPI/通路背景保留，监督性 HPO–Gene 边按 fold 隔离。

## 16.2 产物

```text
data/graphs/fold_0/node_map.parquet
data/graphs/fold_0/A_hpo_hpo.npz
data/graphs/fold_0/A_hpo_gene.npz
data/graphs/fold_0/A_gene_gene.npz
data/graphs/fold_0/A_gene_pathway.npz
data/graphs/fold_0/A_pathway_pathway.npz
data/graphs/fold_0/transition_matrix.npz
data/graphs/fold_0/graph_qc.json
```

## 16.3 传播矩阵

- 每种关系独立构建 CSR 矩阵并进行关系内行归一化；
- 关系预算按源节点类型逐行应用，不能做无掩码的全局矩阵求和；
- 节点缺少某类外边时，预算按比例重分配给实际存在的外出关系；
- 完全孤立节点加入自环；最终所有行和接近 1；
- fold 间只有训练派生的 HPO–Gene 关系允许不同。

## 16.4 验收

矩阵形状一致、权重非负、索引唯一、行归一化误差达标；报告节点/关系数量、孤立节点、连通分量、候选覆盖和心血管核心案例种子覆盖；人工小图结果可手算复核。

---

# 17. Milestone 7：心血管表型语义基线

使用 fold 特异 IC 实现 Resnik 与 BMA，并以训练疾病的已知基因生成候选基因排序。实现最大相似疾病、Top-5 平均和全训练疾病加权和三种聚合。

输出：

```text
outputs/rankings/fold_0/semantic/rankings.parquet
```

必须验证相似度对称、相同集合相似度最大、无测试注释参与 IC 或评分、每个查询覆盖完整候选基因全集，并单独报告心血管疾病家族和三个核心案例的表现。

---

# 18. Milestone 8：表型驱动 RWR 传播

种子仅放在查询疾病的 HPO 节点：

\[
s_d^{(0)}[\phi] =
\frac{frequency(d,\phi)\,IC_{train}(\phi)}
{\sum_{\phi'\in\Phi_d}frequency(d,\phi')\,IC_{train}(\phi')}
\]

使用稀疏矩阵执行：

\[
s^{(t+1)}=\alpha s^{(0)}+(1-\alpha)P^\top s^{(t)}
\]

验证集搜索 `alpha ∈ {0.1, 0.2, 0.3, 0.5, 0.7}`；测试集不参与选择。输出原始得分、排名、迭代次数、最终误差和配置哈希。必须测试 `alpha=1`、概率质量、收敛、确定性、非负性、方向和人工小图。

收敛上限使用 500 次；真实 fold 2 在保持 `1e-10` 容差时需要 202 次迭代，因此原 200 次上限缺少安全余量。该调整不改变传播公式、容差或参数选择规则。

---

# 19. Milestone 9：网络偏差校正

为每个真实心血管查询构造匹配 null queries，匹配表型数量、HPO 深度、训练 IC 分布和顶层表型系统；null 抽样池只能使用当前 fold 允许的数据。

开发阶段使用 100 次，最终实验使用 500–1000 次。输出 raw score、null mean/std、Z-score、经验百分位、校正排名和基因度数。必须报告校正前后得分与基因度数的 Spearman 相关，并对零方差背景做显式审计。

V0 的 null 池严格限定为当前 fold 的训练疾病，使用表型数量、全局 HPO 深度、训练 IC 四分位和顶层临床系统距离选择最近 25 个训练表型谱，再以温度 0.5 固定种子 bootstrap 1,000 次。HPO 表型异常细分到 `HP:0000118` 的器官系统，其他合法 HPO 类别保留全局一级类别；无法赋予系统的宽泛根注释进入审计。校正排名以 ties 中位秩经验百分位为主、Z-score 为次；零方差 Z-score 置零并逐行标记。null RWR 按 64 列分批仅为降低内存，不改变算法。

---

# 20. Milestone 10：正式评价、消融与外部验证

## 20.1 内部评价

报告 Recall@1/5/10/20、MRR、Average Precision、AUPRC、macro mean、median 和疾病家族分层的 95% bootstrap CI；同时报告 candidate coverage，不可评价阳性不能被静默删除。

## 20.2 基线与消融

比较随机、基因度数、Resnik/BMA、仅 HPO–Gene、加入 PPI、加入 Reactome、未校正完整模型和校正完整模型。逐一去掉 HPO 层级、PPI、Reactome、表型频率、训练 IC 和偏差校正；比较 STRING 700/900，BioGRID 仅作网络来源敏感性。

## 20.3 冻结外部验证

```text
GWAS Catalog：疾病/性状相关基因或位点映射的独立支持
GTEx：心脏、冠状动脉/主动脉等相关组织的表达或 eQTL 支持
GO：功能富集与机制注释
```

外部资源必须按版本冻结，不能参与训练、调参或候选全集扩充。GTEx 表达只能作为组织相关支持，不能单独视为疾病因果证据。

## 20.4 通过标准

多数 fold 的 Recall@10 或 MRR 稳定优于语义基线；提升不由度数偏差或候选覆盖差异解释；关键结论在 STRING 700/900 下方向一致；无泄漏且可复现。

---

# 21. Milestone 11：心血管免疫代谢机制模块

在 Milestone 10 冻结的主排序 raw RWR Top-200 候选基因 PPI 子图上运行 Leiden；校准 Z 仅作为模块评分中的偏差敏感性分量，不以 corrected rank 替换主候选排序。DIAMOnD 作为依赖预测种子的备选扩展。模块大小限定 5–100，不能由单一 hub 支配。该变更依据见 `docs/design_decisions.md`。

模块评分结合 MeanZ、内部连通性和扰动稳定性。至少进行 50 次扰动：删除 10% HPO、删除 10% PPI、STRING 700/900 切换和表型频率噪声。共识节点阈值预设为 0.60。

输出模块、成员、稳定性以及 Reactome/GO 富集结果；重点标记免疫炎症、脂质代谢、嘌呤/尿酸代谢、氧化应激和心肌能量代谢，但标签必须由统计富集和可追溯数据库证据支持，不能按预期机制人工命名。

---

# 22. Milestone 12：证据约束解释路径

允许路径：

```text
HPO → HPO → Gene
HPO → Gene → Gene
HPO → Gene → Pathway → Gene
HPO → HPO → Gene → Gene
```

限制路径长度 2–4、目标为 Top-20、每个目标最多 3 条、禁止重复节点。边成本由 `-log(weight + epsilon)` 与预注册关系惩罚组成。

每条路径输出节点、关系、来源数据库、最低置信度和路径成本。删除最高分路径的关键边并重新传播；贡献接近 0 的路径不得作为主要解释。路径是模型内证据链，不得表述为已证实因果链。

---

# 23. Milestone 13：心血管案例与临床桥接

## 23.1 预注册案例

- 冠状动脉粥样硬化；
- 心肌梗死；
- 心力衰竭。

疾病必须使用明确 MONDO/xref 定义并记录纳入版本。若公开表型或阳性基因不足，报告不可评价，不得手工补标签。

## 23.2 公共数据案例报告

每个案例输出输入 HPO、Top-100/Top-20 基因、已知基因命中、机制模块、稳定性、解释路径、传播轨迹、GWAS/GTEx 支持、数据版本和限制。

## 23.3 临床特征桥接

血清尿酸、LDL-C、hs-CRP、IVUS 斑块负荷和最小管腔面积通过 Milestone 2 的临床特征注册表与案例结果对齐。没有合规队列时只做文献/数据库证据映射；获得去标识化队列后，派生 IVUS 数值可用于独立关联或外部验证，但不得把 IVUS 原始影像、患者结局或连续指标混入当前疾病级训练图。

## 23.4 解释边界

允许输出候选机制模块、可能涉及的生物过程、与已有证据一致的可验证假设。禁止宣称证实因果、提供临床诊断或治疗建议。患者级预测、原始 IVUS 影像、免疫细胞/代谢物节点及可学习传播属于后续临床扩展阶段。

---
# 24. 单元测试要求

## 24.1 数据测试

```text
test_no_obsolete_hpo_nodes
test_all_hpo_edges_have_valid_nodes
test_all_genes_use_canonical_ids
test_no_string_self_loops
test_no_duplicate_undirected_edges
test_pathway_size_filter
test_train_test_diseases_disjoint
test_no_test_labels_in_train_graph
```

## 24.2 图测试

```text
test_relation_matrices_same_shape
test_transition_weights_nonnegative
test_transition_rows_sum_to_one
test_node_indices_unique
test_node_index_roundtrip
```

## 24.3 模型测试

```text
test_rwr_alpha_one_returns_seed
test_rwr_probability_mass
test_rwr_convergence
test_rwr_deterministic
test_handcrafted_graph_result
```

## 24.4 评价测试

```text
test_recall_at_k
test_mrr
test_multiple_positive_genes
test_empty_prediction_handling
test_tied_scores_handling
```

## 24.5 测试原则

- 自动测试只使用 `tests/fixtures/`；
- 不依赖大型外部数据库；
- 不允许通过删除测试或放宽断言掩盖错误；
- 每个里程碑完成后运行：

```bash
pytest -q
```

---

# 25. 数据质量报告

每个清洗脚本必须输出 QC JSON，至少包含：

```text
input_records
output_records
removed_records
duplicate_records
unmapped_records
obsolete_records
replaced_records
unresolved_records
node_count
edge_count
connected_components
largest_component_fraction
```

图构建报告还必须包含：

```text
nodes_by_type
edges_by_relation
isolated_nodes
degree_summary
matrix_shape
matrix_nnz
row_sum_error
```

---

# 26. 日志与错误处理

## 25.1 日志

所有脚本必须记录：

- 开始时间；
- 输入文件；
- 配置；
- 处理进度；
- 关键统计；
- 输出路径；
- 结束状态。

## 25.2 错误处理

- 输入文件不存在：立即失败；
- 必需列不存在：立即失败；
- 无法映射记录：写入审计表；
- 过高未映射率：失败并提示；
- 输出路径冲突：根据显式参数决定覆盖；
- 禁止默默捕获异常。

---

# 27. 命令行接口

每个脚本至少支持：

```text
--config
--input
--output
--seed
--overwrite
--log-level
```

推荐示例：

```bash
python scripts/01_parse_hpo.py \
  --config configs/data.yaml \
  --log-level INFO
```

---

# 28. 一键运行目标

V0 完成后，应支持：

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
python scripts/10_evaluate_and_validate.py --config configs/default.yaml
python scripts/11_detect_modules.py --config configs/default.yaml
python scripts/12_extract_paths.py --config configs/default.yaml
python scripts/13_run_cardiovascular_cases.py --config configs/default.yaml
```

后续可增加：

```bash
make test
make build-data
make benchmark
make evaluate
```

但不是 Milestone 0 的强制要求。

---

# 29. 8 周建议排期

## 第 1 周

- 工程初始化；
- HPO 解析；
- HPO QC。

## 第 2 周

- 疾病–表型；
- 疾病–基因；
- MONDO 映射；
- disease samples。

## 第 3 周

- STRING；
- Reactome；
- ID 对齐；
- 网络 QC。

## 第 4 周

- disease-disjoint folds；
- 泄漏检查；
- Resnik/BMA。

## 第 5 周

- 稀疏异质图；
- RWR；
- 人工小图验证。

## 第 6 周

- Null query；
- Z-score；
- 正式评价；
- 消融。

## 第 7 周

- Leiden；
- DIAMOnD；
- 稳定性分析。

## 第 8 周

- 解释路径；
- 心血管案例；
- V0 报告。

---

# 30. 第一版最终交付物

## 29.1 数据产品

- HPO 节点表；
- HPO 层级边；
- 疾病样本表；
- STRING 基因网络；
- Reactome 基因–通路关系；
- 五折疾病划分；
- 节点映射表；
- 图矩阵；
- QC 报告；
- 泄漏报告。

## 29.2 算法产品

- Resnik/BMA；
- RWR；
- Null calibration；
- Leiden；
- DIAMOnD；
- 稳定性分析；
- 路径搜索；
- 路径删除实验。

## 29.3 实验产品

- 全部基线；
- 五折评价；
- 参数敏感性；
- STRING 阈值实验；
- 消融；
- 偏差分析；
- 心血管案例。

---

# 31. V0 完成标准

只有同时满足以下条件，V0 才算完成：

1. 全流程可以从清洗后数据运行到最终结果；
2. 五折实验无疾病和标签泄漏；
3. 所有随机操作可复现；
4. 全部自动测试通过；
5. 完整模型在多数 fold 上优于纯语义基线；
6. 改进不是由节点度数造成；
7. STRING 700 和 900 结论大体一致；
8. 心血管案例可输出模块和解释路径；
9. 所有结果可追溯到原始数据和版本；
10. 不产生未经验证的因果结论。

---

# 32. V1 升级路径

V0 稳定后，保持数据和评测框架不变，按以下顺序升级：

```text
V1 将固定关系权重 RWR 逐步升级为表型条件化的可学习长距离传播；
APPNP 只是一个可能的基础实现，不是 V0 依赖。
```

优先替换传播器，不要同时改数据、图谱、损失和评价方式。

每增加一个模块，必须与上一版本进行严格消融。

---

# 33. Codex 执行纪律

Codex 每次只实现一个 Milestone。

每个 Milestone 的工作流程：

```text
阅读 AGENTS.md
→ 阅读本文件
→ 检查当前仓库
→ 实现当前里程碑
→ 编写测试
→ 运行目标测试
→ 运行 pytest -q
→ 自查 diff
→ 报告修改和限制
```

必须遵守：

- 不提前实现后续里程碑；
- 不下载未授权或不必要的大型数据；
- 不修改 `data/raw/`；
- 不把未标注基因当作确认负例；
- 不把外部验证数据混入训练图；
- 不通过删除测试让测试通过；
- 不硬编码绝对路径；
- 不创建空 TODO 函数；
- 不声称网络预测具有因果性。
