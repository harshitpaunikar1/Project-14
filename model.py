"""
CO2 emissions prediction pipeline.
Trains regression models to estimate vehicle/country CO2 emissions from features.
Compares Linear, Ridge, Random Forest, and Gradient Boosting regressors.
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CO2EmissionsPredictor:
    """
    Regression pipeline for predicting CO2 emissions.
    Handles mixed numeric/categorical features, missing value imputation,
    and cross-validation based model selection.
    """

    def __init__(self, numeric_features: List[str], categorical_features: List[str],
                 target_col: str = "co2_emissions"):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.models: Dict[str, object] = {}
        self.results: List[Dict] = []
        self.best_model_name: Optional[str] = None
        self.preprocessor = None

    def _build_preprocessor(self) -> object:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required.")
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                 self.categorical_features))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _build_models(self) -> Dict[str, object]:
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        }

    def fit_and_evaluate(self, df: pd.DataFrame, test_size: float = 0.2, cv_folds: int = 5) -> pd.DataFrame:
        """
        Train all models on the dataset and evaluate with RMSE, MAE, R2, and CV score.
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required.")
        all_cols = self.numeric_features + self.categorical_features + [self.target_col]
        df = df[all_cols].dropna(subset=[self.target_col])
        for col in self.numeric_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna("unknown")

        X = df[self.numeric_features + self.categorical_features]
        y = df[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        self.preprocessor = self._build_preprocessor()
        base_models = self._build_models()
        self.results = []

        for name, estimator in base_models.items():
            pipe = Pipeline([("preprocessor", self.preprocessor), ("model", estimator)])
            cv_scores = cross_val_score(pipe, X_train, y_train,
                                        cv=cv_folds, scoring="neg_root_mean_squared_error")
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            cv_rmse = float(-cv_scores.mean())
            self.models[name] = pipe
            self.results.append({
                "model": name,
                "rmse": round(rmse, 3),
                "mae": round(mae, 3),
                "r2": round(r2, 4),
                "cv_rmse": round(cv_rmse, 3),
            })

        results_df = pd.DataFrame(self.results).sort_values("rmse").reset_index(drop=True)
        self.best_model_name = results_df.iloc[0]["model"]
        return results_df

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Run inference using the best fitted model."""
        if self.best_model_name is None or self.best_model_name not in self.models:
            raise RuntimeError("Models not trained. Call fit_and_evaluate first.")
        return self.models[self.best_model_name].predict(df[self.numeric_features + self.categorical_features])

    def feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importances for tree-based best model, if available."""
        if self.best_model_name not in self.models:
            return None
        pipe = self.models[self.best_model_name]
        estimator = pipe.named_steps["model"]
        if not hasattr(estimator, "feature_importances_"):
            return None
        preprocessor = pipe.named_steps["preprocessor"]
        try:
            feature_names = (
                self.numeric_features
                + list(preprocessor.named_transformers_["cat"].get_feature_names_out(self.categorical_features))
            )
        except Exception:
            feature_names = [f"feature_{i}" for i in range(len(estimator.feature_importances_))]
        return pd.DataFrame({
            "feature": feature_names,
            "importance": estimator.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def emission_band(self, value: float) -> str:
        """Classify a CO2 value into a regulatory band."""
        if value < 100:
            return "A (very low)"
        elif value < 150:
            return "B (low)"
        elif value < 200:
            return "C (moderate)"
        elif value < 250:
            return "D (high)"
        else:
            return "E (very high)"


if __name__ == "__main__":
    np.random.seed(42)
    n = 2000
    engine_size = np.random.uniform(1.0, 5.0, n)
    cylinders = np.random.choice([4, 6, 8], n)
    fuel_type = np.random.choice(["gasoline", "diesel", "hybrid"], n)
    city_mpg = np.random.uniform(10, 40, n)
    highway_mpg = np.random.uniform(15, 50, n)
    co2 = 120 + 30 * engine_size - 1.5 * city_mpg + np.random.normal(0, 15, n)

    df = pd.DataFrame({
        "engine_size": engine_size,
        "cylinders": cylinders.astype(float),
        "city_mpg": city_mpg,
        "highway_mpg": highway_mpg,
        "fuel_type": fuel_type,
        "co2_emissions": co2,
    })

    predictor = CO2EmissionsPredictor(
        numeric_features=["engine_size", "cylinders", "city_mpg", "highway_mpg"],
        categorical_features=["fuel_type"],
    )
    results = predictor.fit_and_evaluate(df)
    print("Model comparison:")
    print(results.to_string(index=False))
    print(f"\nBest model: {predictor.best_model_name}")

    importances = predictor.feature_importance()
    if importances is not None:
        print("\nTop feature importances:")
        print(importances.head(5).to_string(index=False))

    sample_pred = predictor.predict(df.head(5))
    for val in sample_pred:
        print(f"  Predicted CO2: {val:.1f} -> Band: {predictor.emission_band(val)}")
