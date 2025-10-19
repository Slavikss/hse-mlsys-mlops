# Fraud Detection MLOps Pipeline

Проект для упаковки модели детекции мошеннических транзакций в сервис, соответствующий требованиям ДЗ по MLOps.

## Структура репозитория

- `source/train/preprocess_data.py` — шаблон препроцессинга, превращающий сырой CSV в набор фичей, совместимый с CatBoost. Логика подходит для офлайн- и онлайн-обработки.
- `source/train/train.py` — обучение модели CatBoost с хронологическим hold-out (последние 10% датасета) и сеткой гиперпараметров; итоговые метрики, best-итерация и оптимальный порог по F1 сохраняются вместе с моделью.
- `inference.py` — скрипт инференса, объединяющий этапы загрузки, препроцессинга и скоринга. Формирует `sample_submission.csv`, top-5 feature importances (`feature_importances.json`) и график распределения предсказаний (`prediction_density.png`).
- `requirements.txt`, `Dockerfile`, `.dockerignore` — окружение и упаковка в Docker.
- `artifacts/` — каталог для обученной модели (`catboost_model.cbm`) и метаданных (`feature_metadata.json`). После запуска `train.py` заполняется автоматически.

## Локальное окружение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Обучение модели

```bash
python source/train/train.py \
  --train-path source/data/train.csv \
  --model-path artifacts/catboost_model.cbm \
  --metrics-path artifacts/metrics.json \
  --metadata-path artifacts/feature_metadata.json \
  --validation-size 0.1 \
  --split-strategy time
```

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

> Порог для перевода вероятностей в метки берётся из метаданных (`decision_threshold`). При необходимости можно переопределить его через `--threshold`.

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
