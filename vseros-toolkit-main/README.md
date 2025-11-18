# vseros-toolkit

### Models API (train_cv, blend, calibration, thresholds)

```python
from common.models import gbdt, linear, blend, calibration, thresholds, eval as ME

# 1) тренируем двух кандидатов
run_tab = gbdt.train_cv(X_dense_tr, y, X_dense_te, folds,
                        task="binary", lib="lightgbm",
                        params={"n_estimators":500,"max_depth":8,"learning_rate":0.05},
                        show_progress=True,
                        verbose=False)

run_txt = linear.train_cv(X_sparse_tr, y, X_sparse_te, folds,
                          task="binary", algo="lr", params={"C":2.0,"max_iter":200},
                          show_progress=True,
                          verbose=False)

# 2) выбираем бленд
scorer = ME.get_scorer("binary","roc_auc")
run_bl  = blend.equal_weight([run_tab, run_txt])  # или weight_search / ridge_level2

# 3) калибруем и подбираем порог на OOF
cal = calibration.fit(run_bl.oof_true, run_bl.oof_pred, method="platt")
oof_prob_cal = calibration.apply(cal, run_bl.oof_pred)
tau = thresholds.find_global_tau(run_bl.oof_true, oof_prob_cal, scorer)

# 4) применяем к test
test_prob_cal = calibration.apply(cal, run_bl.test_pred)
# далее — τ или top-K в 04_eval_post
```

## Configs

Система конфигураций собрана слоями (`configs/core` → `configs/recsys` → `configs/datasets/<id>` → overrides/env/CLI) с глубокой мердж-логикой и валидацией через JSONSchema.

* Быстрый просмотр резолв-конфига: `python tools/config_show.py --subsystem recsys --profile scout --dataset_id s5e11 --section recsys.candidates`.
* Линт с валидацией: `python tools/config_lint.py --subsystem recsys --profile gate --dataset_id s5e11`.
* Хеш настроек (fingerprint) для артефактов/кэша: `python tools/config_fingerprint.py --subsystem recsys --profile full`.

Слои и приоритеты описаны в `configs/core/defaults.yaml`, профили лежат в `configs/recsys/profiles/`, а датасетные переопределения — в `configs/datasets/<dataset_id>/`.
