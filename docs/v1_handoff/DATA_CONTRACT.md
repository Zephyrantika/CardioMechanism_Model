# V1 数据与临床接口契约

## 冻结的 V0 输入

所有 V1 模型必须使用同一批 16,606 个候选 NCBI Gene ID、同一组 451 个心血管 MONDO 疾病和同一套五折 MONDO 疾病家族互斥划分。V0 RWR 分数是冻结的可选残差特征，不能在训练时用测试标签重新计算。

每折 query 实例至少包含：

```text
fold, split, disease_id, disease_family_ids,
hpo_ids, hpo_weights, positive_gene_ids,
covered_positive_gene_ids, unresolved_positive_gene_ids, source_version
```

训练图可以使用 HPO、Gene、PPI 和 Reactome 节点，但必须删除验证/测试疾病及其家族的标签、等价关系、父子重复关系和可直接恢复标签的交叉引用。每折写出图哈希和泄漏报告。

## 未标注数据策略

未知疾病–基因对是 `unlabelled`，不是确认阴性。默认采用按图度数分层的五成员 PU bagging；每个抽样记录保留 `label_role=sampled_unlabelled` 和派生 seed。均匀随机阴性只能作为非主分析。

## 预留临床接口

没有患者数据时，V1-A 不读取临床表。未来 V1-B 使用长表：

```text
patient_id, visit_id, site_id, visit_datetime,
disease_id, feature_id, value, unit, missing_reason, source_record_id
```

允许的 `feature_id` 为 `CLIN:SUA`、`CLIN:LDL_C`、`CLIN:HS_CRP`、`CLIN:IVUS_PLAQUE_BURDEN` 和 `CLIN:IVUS_MIN_LUMEN_AREA`。适配器必须保留原始值、单位、来源和未解析记录；不能静默换算单位或强制映射为 HPO。

未来接口位置：`src/phenotype_network_v1/clinical/schema.py`、`adapter.py`、`split.py` 和 `configs/v1/clinical.yaml`。启用前必须具备合规证明、脱敏数据字典、缺失报告、结局定义及患者/医院/时间感知拆分方案。
