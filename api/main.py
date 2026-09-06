from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Запускаем из корня проекта: python -m uvicorn api.main:app --reload
from api.schemas import CarRequest, PredictionResponse

# Важно: импортируем модуль, где объявлен UsedCarsPreprocessor.
# Это нужно joblib при восстановлении Pipeline из файла.
from ml import preprocessing as _preprocessing  # noqa: F401


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "artifacts" / "used_cars_pipeline.joblib"),
    )
)
METADATA_PATH = MODEL_PATH.with_name("metadata.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает Pipeline один раз при старте приложения."""

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Файл модели не найден: {MODEL_PATH.resolve()}. "
            "Положи used_cars_pipeline.joblib в api/artifacts."
        )

    app.state.pipeline = joblib.load(MODEL_PATH)

    if METADATA_PATH.exists():
        app.state.metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )
    else:
        app.state.metadata = {"model_version": "unknown"}

    yield
    app.state.pipeline = None


app = FastAPI(
    title="Used Cars Price API",
    description="API для прогнозирования цены подержанного автомобиля.",
    version="1.0.0",
    lifespan=lifespan,
)

# Всё содержимое api/static будет доступно по адресу /static/...
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

# @app.get("/")
# def index():
#     return {
#         "message": "Used Cars API is running",
#         "web_client": "/static/index.html",
#         "swagger": "/docs",
#     }
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    pipeline = getattr(request.app.state, "pipeline", None)
    return {
        "status": "ok" if pipeline is not None else "model_not_loaded",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    car: CarRequest,
    request: Request,
) -> PredictionResponse:
    """Принимает сырые признаки автомобиля и возвращает прогноз цены."""

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        raw_car = pd.DataFrame([car.model_dump()])
        prediction = pipeline.predict(raw_car)

        return PredictionResponse(
            predicted_price=float(prediction[0]),
            model_version=request.app.state.metadata.get(
                "model_version", "unknown"
            ),
        )
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=f"Ошибка входных данных: {error}",
        ) from error
    except Exception as error:
        print(f"Prediction error: {error}")
        raise HTTPException(
            status_code=500,
            detail="Не удалось выполнить прогноз",
        ) from error

print("Загружен файл:", __file__)
print(
    "Маршруты:",
    [
        (
            route.path,
            getattr(route, "methods", None)
        )
        for route in app.routes
    ]
)