from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from ml.preprocessing import UsedCarsPreprocessor


DATA_PATH = Path("../data/cars.csv")
ARTIFACTS_DIR = Path("../api/artifacts")
PIPELINE_PATH = ARTIFACTS_DIR / "used_cars_pipeline.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Не найден датасет: {DATA_PATH.resolve()}\n"
            "Положи cars.csv рядом со скриптом или измени DATA_PATH."
        )

    print(f"Загрузка датасета: {DATA_PATH.resolve()}")
    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "manufacturer",
        "model",
        "year",
        "mileage",
        "engine",
        "transmission",
        "drivetrain",
        "fuel_type",
        "mpg",
        "exterior_color",
        "interior_color",
        "accidents_or_damage",
        "one_owner",
        "personal_use_only",
        "seller_name",
        "seller_rating",
        "driver_rating",
        "driver_reviews_num",
        "price_drop",
        "price",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют колонки: {sorted(missing)}")

    # Это фильтры из исходного ноутбука. Они применяются до split,
    # но статистики preprocessing считаются только внутри train.
    df = df[
        (df["price"] < 1_000_000)
        & (df["year"] >= 2000)
        & (df["mileage"] <= 800_000)
    ].copy()

    X = df.drop(columns=["price"])
    y = df["price"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=228,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", UsedCarsPreprocessor()),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=30,
                    min_samples_split=30,
                    min_samples_leaf=10,
                    random_state=228,
                ),
            ),
        ]
    )

    print("Обучение pipeline...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
    print(f"R2: {r2_score(y_test, predictions):.6f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH, compress=3)

    preprocessor = pipeline.named_steps["preprocessor"]
    metadata = {
        "model_type": "DecisionTreeRegressor",
        "model_version": "1.0.0",
        "target": "price",
        "pipeline_file": PIPELINE_PATH.name,
        "feature_names": list(preprocessor.get_feature_names_out()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "filters": {
            "price_lt": 1_000_000,
            "year_gte": 2000,
            "mileage_lte": 800_000,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Pipeline сохранён: {PIPELINE_PATH.resolve()}")
    print(f"Метаданные сохранены: {METADATA_PATH.resolve()}")


if __name__ == "__main__":
    main()
