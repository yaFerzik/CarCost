from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

pipeline = joblib.load(
    Path("../api/artifacts/used_cars_pipeline.joblib")
)

df = pd.read_csv('../data/cars.csv')

df = df[
    (df["price"] < 1_000_000)
    & (df["year"] >= 2000)
    & (df["mileage"] <= 800_000)
].copy()

df_train, df_test = train_test_split(df, test_size=0.2, random_state=228)

X_test = df_test.drop(columns=["price"])
y_test = df_test["price"]

y_pred = pipeline.predict(X_test)

print("Предсказанная цена:", float(y_pred[0]))

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))
