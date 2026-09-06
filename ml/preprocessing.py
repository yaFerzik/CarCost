from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


COLOR_GROUPS = [
    ("black", ["black", "noir", "onyx", "caviar", "ebony", "dark"]),
    ("white", ["white", "blanc", "pearl", "whiite", "fresh powder", "light"]),
    ("silver", ["silver"]),
    ("gray", ["gray", "grey", "charcoal", "graphite", "slate", "ash", "stone", "cement"]),
    ("red", ["red", "burgundy", "maroon", "cherry", "crimson", "scarlet ember", "cognac"]),
    ("blue", ["blue", "navy", "indigo", "lunar", "area 51"]),
    ("green", ["green", "emerald", "olive"]),
    ("brown", ["brown", "tan", "espresso", "bronze", "chocolate", "mocha", "cocoa"]),
    ("yellow", ["yellow", "gold", "champagne"]),
    ("orange", ["orange", "copper", "sand", "dune"]),
    ("purple", ["purple", "violet"]),
    ("metal", ["metal", "steel", "platinum", "pewter", "magnetic", "billet"]),
    ("beige", ["beige", "birch", "ivory", "parchment", "wheat", "sandstone", "cappuccino", "almond", "blond", "taupe", "cream"]),
]


def get_train_type(value: Any) -> str:
    text = str(value).lower()
    if re.match(r".*(all|four|4wd|awd)", text):
        return "AWD"
    if re.match(r".*(front|fwd)", text):
        return "FWD"
    if re.match(r".*(rear|rwd)", text) or re.match(r".*m.?t", text):
        return "RWD"
    return "Unknown"


def get_trans_type(value: Any) -> str:
    text = str(value).lower()
    if re.match(r".*cvt", text) or re.match(r".*variable.*", text) or re.match(r".*ivt.*", text):
        return "CVT"
    if re.match(r".*a.?t", text):
        return "AT"
    if re.match(r".*man", text) or re.match(r".*m.?t", text):
        return "MT"
    if re.match(r".*dct", text) or re.match(r".*dual", text):
        return "DCT"
    return "Unknown"


def get_fuel_type(value: Any) -> str:
    text = str(value).lower()
    if re.match(r".*(hybrid|phev|4wd|electric fuel)", text):
        return "Hybrid"
    if re.match(r".*natural", text):
        return "Natural Gas"
    if re.match(r".*(gasoline|gas|unleaded)", text):
        return "Gasoline"
    if re.match(r".*(flex|e85)", text):
        return "Flex Fuel"
    if re.match(r".*diesel", text):
        return "Diesel"
    if re.match(r".*electric", text):
        return "Electic"
    return "Unknown"


def split_mpg(value: Any) -> tuple[float, float]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return np.nan, np.nan

    parts = str(value).lower().split("-")
    if len(parts) == 1:
        left = right = parts[0]
    elif len(parts) == 2:
        left, right = parts
    else:
        return np.nan, np.nan

    try:
        return float(left), float(right)
    except ValueError:
        return np.nan, np.nan


def simplify_color(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unknown"

    text = str(value).lower()
    for group, words in COLOR_GROUPS:
        if any(word in text for word in words):
            return group
    return "other"


class UsedCarsPreprocessor(BaseEstimator, TransformerMixin):
    """Преобразует сырые записи cars.csv в признаки для DecisionTreeRegressor.

    Все статистики вычисляются в fit только по train. После этого transform
    использует сохранённые значения и не изучает данные нового запроса.
    """

    def __init__(
        self,
        smoothing: int = 5,
        n_splits: int = 5,
        mileage_power: float = 0.3206534790699,
        year_power: float = 4.764746051087997,
        mileage_scale_max: float = 800000.0,
        random_state: int = 42,
    ):
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.mileage_power = mileage_power
        self.year_power = year_power
        self.mileage_scale_max = mileage_scale_max
        self.random_state = random_state

    @staticmethod
    def _deterministic_transform(X: pd.DataFrame) -> pd.DataFrame:
        prepared = X.copy(deep=True)

        mpg_values = prepared["mpg"].apply(split_mpg)
        prepared["mpg_l"] = mpg_values.map(lambda value: value[0])
        prepared["mpg_r"] = mpg_values.map(lambda value: value[1])
        prepared.drop(columns=["mpg"], inplace=True)

        prepared["engine_l"] = (
            prepared["engine"].astype("string")
            .str.extract(r"(\d+\.?\d*)\s*L", expand=False)
            .astype(float)
        )
        prepared["dt_type"] = prepared["drivetrain"].apply(get_train_type)
        prepared["tm_type"] = prepared["transmission"].apply(get_trans_type)
        prepared["fu_type"] = prepared["fuel_type"].apply(get_fuel_type)
        prepared["int_col_simp"] = prepared["interior_color"].apply(simplify_color)
        prepared["ext_col_simp"] = prepared["exterior_color"].apply(simplify_color)

        prepared.drop(
            columns=[
                "engine",
                "drivetrain",
                "transmission",
                "fuel_type",
                "interior_color",
                "exterior_color",
            ],
            inplace=True,
        )
        return prepared

    def _base_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        prepared = self._deterministic_transform(X.reset_index(drop=True))

        prepared["mileage"] = pd.to_numeric(prepared["mileage"], errors="coerce")
        prepared["mileage"] = prepared["mileage"].fillna(self.mileage_mean_)
        prepared["mileage"] = prepared["mileage"].clip(lower=0, upper=self.mileage_scale_max)
        prepared["mileage"] = (prepared["mileage"] / self.mileage_scale_max).pow(self.mileage_power)

        prepared["year"] = pd.to_numeric(prepared["year"], errors="coerce")
        prepared["year"] = prepared["year"].fillna(self.year_min_)
        denominator = self.year_max_ - self.year_min_
        prepared["year"] = ((prepared["year"] - self.year_min_) / denominator).clip(0, 1)
        prepared["year"] = prepared["year"].pow(self.year_power)

        prepared.fillna(
            {
                "accidents_or_damage": 0,
                "one_owner": 1,
                "personal_use_only": 0,
            },
            inplace=True,
        )
        for column, value in self.medians_.items():
            if column in prepared.columns:
                prepared[column] = prepared[column].fillna(value)

        prepared.drop(
            columns=["model", "seller_name", "price_drop", "seller_rating"],
            inplace=True,
            errors="ignore",
        )
        return prepared

    @staticmethod
    def _mapping(categories: pd.Series, y: pd.Series, smoothing: int, global_mean: float) -> dict[str, float]:
        stats = pd.DataFrame({"category": categories.astype(str), "target": y})
        stats = stats.groupby("category")["target"].agg(["mean", "count"])
        values = (
            stats["count"] * stats["mean"] + smoothing * global_mean
        ) / (stats["count"] + smoothing)
        return {str(key): float(value) for key, value in values.to_dict().items()}

    def _encode_with_full_mappings(self, base: pd.DataFrame) -> pd.DataFrame:
        encoded = base.copy(deep=True)
        for column, mapping in self.target_mappings_.items():
            encoded[column + "_enc"] = (
                encoded[column].astype(str).map(mapping).fillna(self.global_target_mean_)
            )
            encoded.drop(columns=[column], inplace=True)
        return encoded

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.reset_index(drop=True).copy()
        y = pd.Series(y).reset_index(drop=True)

        if len(X) != len(y):
            raise ValueError("X и y должны содержать одинаковое количество строк")
        if len(X) < self.n_splits:
            raise ValueError(f"Для n_splits={self.n_splits} нужно минимум {self.n_splits} строк")

        self.mileage_mean_ = float(pd.to_numeric(X["mileage"], errors="coerce").mean())
        valid_years = pd.to_numeric(X["year"], errors="coerce").dropna()
        self.year_min_ = float(valid_years.min())
        self.year_max_ = float(valid_years.max())
        if self.year_max_ <= self.year_min_:
            raise ValueError("year_max должен быть больше year_min")

        deterministic = self._deterministic_transform(X)
        self.medians_ = {}
        for column in ["driver_rating", "mpg_l", "mpg_r", "engine_l"]:
            self.medians_[column] = float(pd.to_numeric(deterministic[column], errors="coerce").median())

        base = self._base_transform(X)
        self.global_target_mean_ = float(y.mean())
        self.categorical_columns_ = {
            "manufacturer": self.smoothing,
            "dt_type": self.smoothing,
            "tm_type": self.smoothing,
            "fu_type": self.smoothing,
            "int_col_simp": self.smoothing,
            "ext_col_simp": self.smoothing,
        }
        self.target_mappings_ = {
            column: self._mapping(base[column], y, smoothing, self.global_target_mean_)
            for column, smoothing in self.categorical_columns_.items()
        }

        full_encoded = self._encode_with_full_mappings(base)
        self.feature_names_out_ = list(full_encoded.columns)
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(full_encoded[self.feature_names_out_])
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None, **fit_params):
        if y is None:
            raise ValueError("Для обучения preprocessing требуется y")
        self.fit(X, y)

        X = X.reset_index(drop=True).copy()
        y = pd.Series(y).reset_index(drop=True)
        base = self._base_transform(X)
        out = base.copy(deep=True)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for column, smoothing in self.categorical_columns_.items():
            values = pd.Series(index=base.index, dtype=float)
            for train_idx, validation_idx in kf.split(base):
                mapping = self._mapping(
                    base.iloc[train_idx][column],
                    y.iloc[train_idx],
                    smoothing,
                    self.global_target_mean_,
                )
                values.iloc[validation_idx] = (
                    base.iloc[validation_idx][column]
                    .astype(str)
                    .map(mapping)
                    .fillna(self.global_target_mean_)
                    .to_numpy()
                )
            out[column + "_enc"] = values.to_numpy()
            out.drop(columns=[column], inplace=True)

        out = out.reindex(columns=self.feature_names_out_)
        return self.scaler_.transform(out)

    def transform(self, X: pd.DataFrame):
        if not hasattr(self, "scaler_"):
            raise RuntimeError("Preprocessor ещё не обучен. Сначала вызови fit().")
        base = self._base_transform(X)
        encoded = self._encode_with_full_mappings(base)
        encoded = encoded.reindex(columns=self.feature_names_out_)
        return self.scaler_.transform(encoded)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)
