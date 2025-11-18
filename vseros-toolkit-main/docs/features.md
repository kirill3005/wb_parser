# Слой фич: обзор и рецепты

Документ описывает, как устроен новый слой фич в `common/features` и как им пользоваться в ноутбуках без CLI/YAML. Все примеры нацелены на работу с `pandas.DataFrame` и возвращают стандартизированный `FeaturePackage`, который можно комбинировать через `FeatureStore` и `assemble`.

## Архитектура и базовые сущности
- **`FeaturePackage` (`common/features/types.py`)** — контейнер с полями `train`, `test`, `kind` (`dense`/`sparse`), списком колонок и метаданными (`meta`).
- **`FeatureStore` (`common/features/store.py`)** — in-memory регистр пакетов. Добавляет пакеты, возвращает их по имени, умеет сохранять/читать кэш через `common/cache.py`.
- **Кэш (`artifacts/features/<block>/<key>`).** Используется всеми билдерами. Ключ строится через `cache.make_key(...)` и включает параметры, отпечаток кода и штамп данных. При повторном запуске билдеры читают готовый пакет, экономя время.
- **Сборка (`common/features/assemble.py`).** `make_dense` и `make_sparse` объединяют выбранные пакеты в общую матрицу, проверяют валидность колонок и возвращают паспорт фич.

### Структура артефактов
- `artifacts/features/img_stats/<key>/train.parquet` — быстрые статистики по изображениям, `meta.json` рядом.
- `artifacts/features/img_embed/<key>/{train.parquet,test.parquet,meta.json,part_*.parquet}` — эмбеддинги с чекпойнтами.
- `artifacts/features/<block>/<key>/*.parquet` или `.npz` — кэш табличных и текстовых фич (dense/sparse).
  Ключ включает `cache_key_extra`, параметры и отпечаток кода, поэтому совпадающие вызовы попадают в один кеш.

### Быстрый старт
```python
from common.features import store, assemble, num_basic, cat_freq

FS = store.FeatureStore()
FS.add(num_basic.build(train, test, num_cols=["price", "age"], log_cols=["price"]))
FS.add(cat_freq.build(train, test, cat_cols=["city", "segment"]))

X_tr, X_te, catalog = assemble.make_dense(FS, include=FS.list())
```

## Числовые фичи: `num_basic.build`
- Автодетект числовых колонок (`num_cols=None`).
- Импутация (`median`/`mean`/константа), опциональный лог на выбранных колонках.
- Клип по квантилям, скейлинг (`standard` или `minmax`).
- Имена колонок: `f"{prefix}__{col}"` (по умолчанию `prefix="num"`).
- Возвращает `FeaturePackage(kind="dense")`, кэшируется по параметрам трансформаций.

Пример:
```python
FS.add(num_basic.build(train, test, num_cols=["price", "age"],
                       impute="median", log_cols=["price"],
                       clip_quant=(0.01, 0.99), scale="standard"))
```

## Категориальные частоты: `cat_freq.build`
- Автодетект категориальных колонок, если `cat_cols=None`.
- Считает частоты и доли значений, редкие значения агрегирует в `RARE`, незнакомые в тесте → `UNK`.
- На выходе числовые фичи с колонками `"{prefix}__{col}__freq"` и `"{prefix}__{col}__ratio"` (`prefix="catf"`).
- Не использует таргет, поэтому безопасен от утечек.

Пример:
```python
FS.add(cat_freq.build(train, test, cat_cols=["city", "channel"], rare_threshold=0.02))
```

## OOF target encoding: `cat_te_oof.build`
- Чёткий OOF-процесс: на каждом фолде кодер обучается на train-fold и применяется к val-fold.
- Поддерживает метод `target` с m-estimate сглаживанием; WOE/CTR пока заглушки с понятной ошибкой.
- Требует таргет `y` и объект `folds` (`List[Tuple[train_idx, val_idx]]`).
- Колонки: `"{prefix}__{col}__{method}"` (`prefix="te"`), `meta['oof']=True`.

Пример:
```python
FS.add(cat_te_oof.build(train, y, test, folds=cv_splits, cat_cols=["city", "channel"],
                        method="target", smoothing="m-estimate"))
```

## Текст: `text_tfidf.build`
- TF-IDF по словам или символам (`use_char=True`).
- `ngram_range`, `min_df` настраиваются; опционально SVD (`svd_k`).
- Возвращает разреженный CSR `FeaturePackage(kind="sparse")`, имена колонок `"{prefix}__{i}"` или после SVD.

Пример:
```python
FS.add(text_tfidf.build(train, test, text_col="description",
                        min_df=5, ngram_range=(1, 2), use_char=False,
                        svd_k=200, prefix="tfidf"))
```

## Гео-фичи
### `geo_grid.build`
- Переводит широту/долготу в квадратные бины (метры → градусы), считает количество и долю точек в бинах.
- Возвращает dense-фичи с префиксом `geo`.

### `geo_neighbors.build`
- При наличии `scikit-learn` строит `BallTree` (haversine) и считает число соседей и плотности в заданных радиусах.
- Если `sklearn` недоступен, выводит предупреждение и возвращает пустой пакет (0 колонок).

Пример:
```python
FS.add(geo_grid.build(train, test, lat_col="lat", lon_col="lon", steps_m=(300, 1000)))
FS.add(geo_neighbors.build(train, test, lat_col="lat", lon_col="lon", radii_m=(300, 1000)))
```

## Работа с изображениями
### Индекс: `img_index.build_from_dir` / `build_from_csv`
- Собирает `Dict[id, List[path]]` из структуры папок или CSV.
- Фильтрует по расширениям (`.jpg/.jpeg/.png/.webp`), нормализует пути, отбрасывает отсутствующие, сортирует и ограничивает `max_per_id`.

Пример индексации каталога:
```python
from common.features.img_index import build_from_dir
all_ids = pd.concat([train["id"], test["id"]]).astype(str).unique()
id2imgs = build_from_dir("data/images", ids=all_ids, pattern="{id}/*.jpg", max_per_id=4)
```

### Быстрый fallback: `img_stats.build`
- Цветовые/геометрические статистики без нейросетей: ширина/высота/ratio, HSV mean/std, доля тёмных/светлых пикселей, резкость (Variance of Laplacian при наличии `cv2`).
- Агрегирует `mean` и `max` по картинкам одного объекта, возвращает dense-фичи `float32`.
- Кэшируется в `artifacts/features/img_stats/<key>`.

Пример:
```python
from common.features import img_stats
FS.add(img_stats.build(train, test, id_col="id", id_to_images=id2imgs, prefix="imgstats"))
```

### Эмбеддинги CNN/ViT: `img_embed.build`
- Поддерживает бэкбоны: `resnet50`, `mobilenet_v3_small`, `vit_b_16`, `clip_vit_b32` (если установлен `open_clip` или CLIP из `torchvision`).
- Батч-инференс с DataLoader, CPU/GPU (`device="auto"`), AMP (`precision="auto"`), аккуратная обработка битых файлов.
- Пулинг внутри сети: `avg`/`gem` для CNN, `token` для ViT/CLIP; агрегация по картинкам объекта: `mean`/`max`/`gem`/`first`.
- Резюмирование: каждые `checkpoint_every` id записываются партиции parquet, при рестарте достраивает незавершённое.
- Кэш: `artifacts/features/img_embed/<key>/{train.parquet,test.parquet,meta.json}`; dtype `float16` или `float32`.

Пример использования в ноутбуке (из докстринга):
```python
from common.features import store
from common.features.img_index import build_from_dir
from common.features.img_embed import build as img_build

ids_train = train["id"].astype(str)
ids_test  = test["id"].astype(str)
id2imgs = build_from_dir("data/images", pd.concat([ids_train, ids_test]).unique(),
                         pattern="{id}/*.jpg", max_per_id=4)

FS = store.FeatureStore()
FS.add(img_build(train, test, id_col="id", id_to_images=id2imgs,
                 backbone="resnet50", image_size=224, agg="mean", pool="avg",
                 batch_size=64, device="auto", precision="auto", dtype="float16"))
```

## Сборка матриц
- `assemble.make_dense(store, include=...)` склеивает dense-пакеты в `pandas.DataFrame` с проверкой совпадения колонок train/test.
- `assemble.make_sparse(store, include=...)` делает то же для CSR-матриц.
- Возвращается `(X_train, X_test, catalog)`, где `catalog` — словарь с метаданными всех пакетов в `FeatureStore`.

Пример с изображениями и табличными фичами:
```python
FS = store.FeatureStore()
FS.add(num_basic.build(train, test))
FS.add(cat_freq.build(train, test))
FS.add(img_stats.build(train, test, id_col="id", id_to_images=id2imgs))
# при наличии torch/моделей можно добавить эмбеддинги
# FS.add(img_embed.build(...))

X_dense_tr, X_dense_te, catalog = assemble.make_dense(FS, include=FS.list())
```

## Лучшие практики
- **Контроль данных:** перед сборкой проверяйте совпадение размеров train/test и наличие нужных колонок; `validators.py` помогает ловить NaN/inf и дубликаты.
- **Кэширование:** включайте `use_cache=True` (по умолчанию), чтобы повторные запуски брали готовые артефакты. Чистить кэш можно удалением папок в `artifacts/features/<block>/`.
- **Воспроизводимость:** фиксируйте сиды через `common/seed.py::set_global_seed` перед построением тяжёлых фич.
- **Резюмирование эмбеддингов:** если процесс прервался, оставшиеся `part_*.parquet` будут подхвачены при следующем вызове `img_embed.build` с теми же параметрами/ключом.
- **Память:** для больших текстовых/изображений фич используйте `svd_k` или более лёгкие бэкбоны (`mobilenet_v3_small`) и `dtype="float16"`.

## Диагностика и smoke-тесты
- Мини-проверка изображения и статистик: `python tests/images_smoke.py` (эмбеддинги запускаются только при наличии `torch`/модели).
- Для табличных билдеров можно быстро проверить импорт/сборку через `python -m compileall common` и запуск небольшого ноутбука с примерами выше.
