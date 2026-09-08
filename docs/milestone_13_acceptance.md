# Milestone 13 Acceptance

Milestone 13 evaluates the three preregistered cardiovascular case studies: coronary atherosclerosis (`MONDO:0021661`), myocardial infarction (`MONDO:0005068`), and heart failure (`MONDO:0005252`). The frozen public HPOA benchmark contains no HPO annotations or known-gene labels for these exact MONDO records, and none belongs to the family-disjoint evaluation folds.

All three cases are therefore reported as unevaluable rather than supplemented with manual labels. Each JSON report preserves the MONDO cross-references and explicitly records the missing inputs. Ranking, propagation, known-gene recovery, module, explanation-path, GWAS, and GTEx fields remain empty. The reports state that no patient-level prediction, diagnosis, treatment recommendation, or causal claim is produced.

The clinical feature bridge contains the five preregistered features for each case, for 15 rows total. It records identifiers, units, roles, and permitted future use for serum uric acid, LDL cholesterol, high-sensitivity C-reactive protein, IVUS plaque burden, and IVUS minimum lumen area. No cohort values or raw IVUS images are available, and none of these features entered the training graph.

Acceptance checks pass: three frozen cases and three reports are present, all unavailable inputs are audited, no manual HPO or gene labels are added, the bridge contains exactly 15 rows, clinical variables are isolated from model training, and unevaluable cases do not receive fabricated rankings or evidence support.
