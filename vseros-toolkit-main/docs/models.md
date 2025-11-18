# Слой моделей: устройство и примеры

Документ описывает модуль `common.models`, который добавляет единый API для обучения моделей
по фолдам, блендинга, калибровки и подбора порогов. Акцент сделан на простых импортируемых
функциях без CLI/конфигов: всё управляется из ноутбуков или скриптов через явные параметры.

## Структура папок и ключевые объекты

```
common/models/
  types.py        # TaskType, ModelRun — контейнер результата
  artifacts.py    # run_id, пути, manifest.json, сохранение массивов и моделей
  eval.py         # метрики и подсчёт CV по фолдам
  gbdt.py         # LightGBM/CatBoost/XGBoost cross-val тренер
  linear.py       # Logistic/Ridge/SGD cross-val тренер (dense/CSR)
  blend.py        # equal_weight, weight_search, ridge_level2
  calibration.py  # Platt / Isotonic fit/apply
  thresholds.py   # подбор τ и top-K
```

### ModelRun

`types.ModelRun` — dataclass, в котором возвращаются все результаты одного запуска:

* `run_id`: строковый идентификатор (используется как имя папки в `artifacts/models/<run_id>`).
* `task`: тип задачи (`regression|binary|multiclass|multilabel`).
* `oof_true`, `oof_pred`, `test_pred`: массивы истинных значений и прогнозов (proba для
  классификации, числа — для регрессии).
* `fold_scores`, `cv_mean`, `cv_std`: значения метрики по фолдам и сводные статистики.
* `artifacts_path`: путь к каталогу артефактов запуска.
* `manifest`: словарь с метаданными (параметры модели, количество фолдов, метрика и т.д.).

## Folds и входные данные

Все тренеры принимают `folds: List[tuple[np.ndarray, np.ndarray]]`, где каждый элемент —
`(train_idx, val_idx)`. Предполагается, что индексы строятся заранее на стороне фич (см. `02_features`).

Типы входных матриц не ограничены: `gbdt.train_cv` работает с dense/NumPy/LightGBM DMatrix,
`linear.train_cv` — с dense и CSR (sklearn). Для задач multi* допускаются one-hot/one-vs-rest
формы таргета (см. сигнатуры тренеров).

## Артефакты и возобновление обучения

`common.models.artifacts` отвечает за единообразное хранение моделей и массивов:

* `make_run_id(task, model, feat_hash, seed)` — формирует человекочитаемый `run_id`
  (`YYYYMMDD-hhmmss-task-model-feathash-seed`).
* `path(run_id)` — создаёт каталог `artifacts/models/<run_id>` и возвращает `Path`.
* `fold_path(run_dir, k, ext)` — путь к модели конкретного фолда (`model_fold_0.lgb` и т.п.).
* `existing_folds(run_dir, ext_candidates)` — множество уже обученных фолдов (для `resume=True`).
* `save_manifest(run_dir, manifest)` и `load_manifest` — запись/чтение `manifest.json`.
* `save_array(run_dir, name, arr)` — удобный хелпер для `*.npy` или JSON-списков.

Использование: тренеры сохраняют модель и кусок OOF после каждого фолда. При `resume=True`
пропускаются фолды, для которых уже есть сохранённая модель.

```python
from common.models import artifacts as A
run_id = A.make_run_id(task="binary", model="lgbm", feat_hash="abc123", seed=42)
run_dir = A.path(run_id)
model_path = A.fold_path(run_dir, k=0, ext=".lgb")
existing = A.existing_folds(run_dir)
```

## Метрики и подсчёт CV

`common.models.eval.get_scorer(task, metric)` возвращает функцию `score(y_true, y_pred)` с
поддержкой метрик:

* Regression: `mae|rmse|r2`
* Binary: `roc_auc|pr_auc|f1|logloss`
* Multiclass: `f1_macro|acc|logloss`
* Multilabel: `f1_micro|f1_macro|map@k`

`cv_scores_by_folds(y_true, y_pred, folds, scorer)` вычисляет метрику по каждому фолду,
используя OOF-маску `val_idx`.

```python
from common.models import eval as ME
scorer = ME.get_scorer("binary", "roc_auc")
fold_scores = ME.cv_scores_by_folds(y, oof_prob, folds, scorer)
cv_mean, cv_std = float(np.mean(fold_scores)), float(np.std(fold_scores))
```

## GBDT: LightGBM/CatBoost/XGBoost

Функция `gbdt.train_cv` обучает модель по фолдам, сохраняет артефакты и возвращает `ModelRun`.

```python
from common.models import gbdt
run = gbdt.train_cv(
    X_train=X_dense_tr,
    y=y,
    X_test=X_dense_te,
    folds=folds,
    params={"n_estimators": 500, "learning_rate": 0.05, "max_depth": 8},
    lib="lightgbm",       # или "catboost" / "xgboost"
    task="binary",
    seed=42,
    n_jobs=8,
    save=True,
    resume=True,
    show_progress=True,
    verbose=False,
)
print(run.cv_mean, run.cv_std, run.artifacts_path)
```

Особенности:

* Поддерживаются задачи `regression|binary|multiclass`.
* Для `lib="lightgbm"` используется `lgb.train` с ранней остановкой через `valid_sets`.
* Для `catboost` — `CatBoostClassifier/Regressor` с `eval_set`; для `xgboost` — `XGBClassifier/Regressor`.
* Прогнозы: числа для регрессии, вероятность положительного класса для binary, матрица вероятностей для
  multiclass (one-hot, если библиотека возвращает метки).
* После каждого фолда модель сохраняется в `model_fold_k.<ext>` (`.lgb`, `.cbm`, `.xgb` или `.joblib`).
* При `resume=True` уже обученные фолды пропускаются (по наличию файла модели).
* `show_progress=True` включает tqdm прогресс-бар по фолдам, `verbose=True` выводит подробные логи и verbose-режим библиотек.
* В `manifest` пишутся параметры, метрика, число фолдов и сводные значения CV.

### Минимальный пример без test-матрицы

```python
folds = [(tr_idx, val_idx) for tr_idx, val_idx in folds_kfold]  # заранее подготовленные индексы
run = gbdt.train_cv(X_train, y, X_test=None, folds=folds, params={"n_estimators": 200}, lib="catboost")
print(run.oof_pred.shape)  # (n_samples,)
```

## Линейные модели: dense и sparse

`linear.train_cv` работает как с плотными матрицами, так и с CSR (TF-IDF). Поддерживаются
`binary/multiclass/multilabel/regression` задачи.

```python
from common.models import linear
run_tfidf = linear.train_cv(
    X_train=X_sparse_tr,
    y=y,
    X_test=X_sparse_te,
    folds=folds,
    algo="lr",            # "lr"|"sgd"|"ridge"|"lasso"
    task="binary",
    params={"C": 2.0, "max_iter": 200, "n_jobs": 8},
    seed=13,
    show_progress=True,
    verbose=False,
)
```

Особенности:

* Классификация: `LogisticRegression` или `SGDClassifier` (elastic net). Для multi* используется
  `OneVsRestClassifier`.
* Регрессия: `Ridge`, `Lasso`, `SGDRegressor`.
* Артефакты фолдов сохраняются как `model_fold_k.joblib`; доступен `resume=True`.
* Метрики выбираются через `params["metric"]` или дефолты: ROC-AUC для binary, F1-macro для multiclass,
  F1-micro для multilabel, RMSE для регрессии.
* `show_progress=True` включает tqdm прогресс-бар по фолдам, `verbose=True` выводит подробные шаги и информацию о сохранении.

### Пример для multilabel

```python
run_ml = linear.train_cv(
    X_train=X_csr,
    y=y_onehot,           # shape (n_samples, n_classes)
    X_test=X_csr_test,
    folds=folds,
    task="multilabel",
    algo="sgd",
    params={"alpha": 1e-4, "loss": "log_loss", "penalty": "elasticnet"},
)
print(run_ml.oof_pred.shape)  # (n_samples, n_classes)
```

## Блендинг и level-2

`blend.py` предлагает три стратегии:

1. `equal_weight(runs)` — простое усреднение OOF/Test нескольких `ModelRun` (вернёт новый `ModelRun`).
2. `weight_search(runs, y_true, scorer, nonneg=True, sum_to_one=True)` — поиск весов по OOF
   под заданную метрику (возвращает dict с весами и значением метрики).
3. `ridge_level2(runs, y_true, alpha=1.0)` — обучает Ridge/LogReg на стэке OOF, применяет к стэку Test,
   сохраняет собственный `run_id` и артефакты (second-level модель).

```python
from common.models import blend, eval as ME
scorer = ME.get_scorer("binary", "roc_auc")
run_bl = blend.equal_weight([run_tab, run_txt])
weights_info = blend.weight_search([run_tab, run_txt], y_true=run_tab.oof_true, scorer=scorer)
run_lvl2 = blend.ridge_level2([run_tab, run_txt], y_true=run_tab.oof_true, alpha=0.5)
```

## Калибровка вероятностей

`calibration.fit(oof_true, oof_prob, method="platt"|"isotonic")` подбирает калибратор по OOF.

```python
from common.models import calibration
cal = calibration.fit(run_bl.oof_true, run_bl.oof_pred, method="platt")
oof_prob_cal = calibration.apply(cal, run_bl.oof_pred)
```

Особенности:

* Возвращается сериализуемый объект (dict/joblib), который можно сохранить в артефакты.
* Для multiclass допускается список калибраторов по классам.
* Применение: `calibration.apply(cal, prob)` — для OOF/Test прогнозов.

## Пороги и top-K

Модуль `thresholds.py` помогает выбрать решающую границу или top-K:

* `find_global_tau(y_true, y_prob, scorer)` — брутфорс по одному τ для binary.
* `find_per_class_tau(y_true_onehot, y_prob, scorer)` — пороги для каждого класса
  (multiclass one-vs-rest или multilabel).
* `apply_topk(scores, k)` — возвращает индексы или one-hot top-K предсказаний.

```python
from common.models import thresholds
from common.models import eval as ME
scorer = ME.get_scorer("binary", "f1")
tau = thresholds.find_global_tau(run_bl.oof_true, oof_prob_cal, scorer)
```

## Полный пример: обучение, бленд, калибровка, порог

```python
from common.models import gbdt, linear, blend, calibration, thresholds, eval as ME

# 1) CV-тренировки
run_tab = gbdt.train_cv(X_dense_tr, y, X_dense_te, folds, task="binary", lib="lightgbm",
                        params={"n_estimators": 400, "learning_rate": 0.05, "max_depth": 8})
run_txt = linear.train_cv(X_sparse_tr, y, X_sparse_te, folds, task="binary", algo="lr",
                          params={"C": 2.0, "max_iter": 200})

# 2) Бленд по ROC-AUC
scorer = ME.get_scorer("binary", "roc_auc")
run_bl = blend.equal_weight([run_tab, run_txt])

# 3) Калибровка Platt + подбор τ под F1
cal = calibration.fit(run_bl.oof_true, run_bl.oof_pred, method="platt")
oof_cal = calibration.apply(cal, run_bl.oof_pred)
tau = thresholds.find_global_tau(run_bl.oof_true, oof_cal, scorer=ME.get_scorer("binary", "f1"))

# 4) Применение к тесту
pred_test_cal = calibration.apply(cal, run_bl.test_pred)
# далее можно применить tau или top-K в пост-обработке
```

## Где смотреть артефакты

После любого запуска в каталоге `artifacts/models/<run_id>/` появятся:

* `manifest.json` — параметры модели, метрика, кол-во фолдов, сводка CV.
* `model_fold_k.*` — сохранённые модели по фолдам (расширение зависит от библиотеки).
* `oof_true.npy`, `oof_pred.npy`, `test_pred.npy` — массивы для воспроизводимости.
* при level-2 или калибровке можно добавлять свои файлы (например, `calibrator.joblib`).

Эти артефакты позволяют возобновлять обучение (`resume=True`) и воспроизводить инференс
без повторного тренинга.
