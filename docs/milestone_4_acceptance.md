# Milestone 4 Acceptance

Reactome v97 produced 1,816 primary pathways, 59,199 gene-pathway edges, and 1,698 retained hierarchy edges after the registered 5–500 gene filter. All genes use NCBI Gene IDs and all pathways use human Reactome stable IDs. The graph covers 195 of 417 Milestone 2 genes (46.76%).

The traceable immunometabolic registry contains 82 immune/inflammation, 49 lipid, 6 purine/urate, 12 oxidative-stress, and 10 cardiac-energy pathways. Manual review tightened broad `mitochond` and `purine` matches to avoid mitochondrial apoptosis, DNA repair, and purinergic-receptor false positives. Representative retained terms include IL-6 signaling, cholesterol homeostasis, purine salvage/catabolism, glutathione pathways, respiratory electron transport, and pyruvate metabolism.

Verification: 36 tests passed; no duplicate gene-pathway records, invalid canonical IDs, out-of-range pathway sizes, or hierarchy edges with missing endpoints. Source versions and SHA256 hashes are in `outputs/qc/pathway_qc.json`.