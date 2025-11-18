# RecSys Candidate Generation

This module provides a lightweight retrieval stage for recommendation experiments.

## Quick start

```
python recsys/tools/run_candidates.py \
  --dataset_id demo --schema recsys/configs/schema.yaml \
  --candidates recsys/configs/candidates.yaml \
  --profile recsys/configs/profiles/scout.yaml \
  --data_interactions recsys/tests/fixtures/tiny_interactions.csv \
  --data_items recsys/tests/fixtures/tiny_items.csv \
  --data_queries recsys/tests/fixtures/tiny_queries.csv \
  --out_path artifacts/recsys/demo/scout/candidates.parquet
```

Profiles overview:

| profile | generators |
| --- | --- |
| scout | covis, pop, session_ngram, item2vec |
| gate | scout + mf_als + graph_ppr |
| full | gate + lightgcn/twotower/content_image (when available) |
| panic | covis + pop |

All profiles share the unified schema defined in `configs/schema.yaml`.
