import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target

print("Shape:", X.shape)
print("Feature columns:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5, weights="distance"))
])

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

y_pred_rf  = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # <-- fixed here
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name} Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.4f}")
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}

results = []
results.append(evaluate_model("Random Forest", y_test, y_pred_rf))
results.append(evaluate_model("XGBoost", y_test, y_pred_xgb))
results.append(evaluate_model("KNN Regressor", y_test, y_pred_knn))

results_df = pd.DataFrame(results)
print("\n\nComparison Table (sorted by RMSE):")
print(results_df.sort_values("RMSE"))

results_sorted = results_df.sort_values(["RMSE", "R2", "MAE"], ascending=[True, False, True])
best = results_sorted.iloc[0]

print("\nBest Model Analysis:")
print(
    f"The best performing model is **{best['Model']}** "
    f"with RMSE = {best['RMSE']:.4f}, MAE = {best['MAE']:.4f}, and R² = {best['R2']:.4f}."
)

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred_rf,  alpha=0.5, label='Random Forest', color='blue')
plt.scatter(y_test, y_pred_xgb, alpha=0.5, label='XGBoost',      color='green')
plt.scatter(y_test, y_pred_knn, alpha=0.5, label='KNN Regressor', color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal Fit Line')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Comparison (RF vs XGB vs KNN)")
plt.legend()
plt.show()
