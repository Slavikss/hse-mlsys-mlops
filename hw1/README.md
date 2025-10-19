# Fraud Detection MLOps Pipeline

Проект для упаковки модели детекции мошеннических транзакций в сервис, соответствующий требованиям ДЗ по MLOps.

## Структура репозитория

- `source/train/preprocess_data.py` — шаблон препроцессинга, превращающий сырой CSV в набор фичей, совместимый с CatBoost. Логика подходит для офлайн- и онлайн-обработки; скрипт можно запускать отдельно через CLI.
- `source/train/train.py` — обучение модели CatBoost с поддержкой случайного (`--split-strategy random`) и хронологического (`--split-strategy time`) hold-out. Скрипт подбирает порог по F1, сохраняет метрики, лучшую итерацию и параметры.
- `inference.py` — скрипт инференса, объединяющий этапы загрузки, препроцессинга и скоринга. Формирует `sample_submission.csv`, top-5 feature importances (`feature_importances.json`) и график распределения предсказаний (`prediction_density.png`); нужные файлы и порог можно переопределять флагами.
- `requirements.txt`, `Dockerfile`, `.dockerignore` — окружение и упаковка в Docker.
- `artifacts/` — каталог для обученной модели (`catboost_model.cbm`) и метаданных (`feature_metadata.json`). После запуска `train.py` заполняется автоматически.

## Локальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Предобработка данных

```bash
python source/train/preprocess_data.py \
  --input-path source/data/train.csv \
  --output-path artifacts/train_features.parquet
```

Скрипт повторяет всю логику фичеинжиниринга, использующуюся при обучении и инференсе. Параметр `--output-path` опционален: без него файл не сохраняется, но в консоль выводится форма и список колонок.

### Обучение модели

```bash
python source/train/train.py \
  --train-path source/data/train.csv \
  --model-path artifacts/catboost_model.cbm \
  --metrics-path artifacts/metrics.json \
  --metadata-path artifacts/feature_metadata.json \
  --validation-size 0.25 \
  --split-strategy random
```

По умолчанию используется стратифицированный случайный сплит 20% (`--split-strategy random`, `--validation-size 0.2`). Для воспроизведения хронологического hold-out укажите `--split-strategy time` и подходящую долю валидации, как в примере выше.

Скрипт сохранит:
- модель CatBoost (`artifacts/catboost_model.cbm`);
- JSON с метриками валидации (`artifacts/metrics.json`);
- описание признаков и рекомендованный порог классификации (`artifacts/feature_metadata.json`).

### Локальный инференс

```bash
python inference.py \
  --input-dir source/data \
  --output-dir output \
  --model-path artifacts/catboost_model.cbm \
  --metadata-path artifacts/feature_metadata.json
```

На выходе в каталоге `output/` появятся:
- `sample_submission.csv` — предсказания в формате Kaggle;
- `feature_importances.json` — top-5 наиболее важных признаков;
- `prediction_density.png` — график плотности распределения предсказанных вероятностей.

> Порог для перевода вероятностей в метки и порядок колонок берутся из `artifacts/feature_metadata.json`. При необходимости можно переопределить его через `--threshold`, а также изменить имена входного и выходного файлов соответствующими аргументами CLI.

## Docker

1. Собрать образ:
   ```bash
   docker build -t fraud-inference .
   ```
2. Подготовить директории:
   ```bash
   mkdir -p input output
   cp source/data/test.csv input/
   ```
3. Запустить контейнер:
   ```bash
   docker run --rm \
     -v "$(pwd)/input":/app/input \
     -v "$(pwd)/output":/app/output \
     fraud-inference
   ```
   После выполнения в `output/` появятся все артефакты инференса.

> **Важно:** перед сборкой Docker-образа убедитесь, что обученная модель и метаданные сохранены в `artifacts/`.
