import os
import math
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

class TravelRecommender:
    def __init__(self, data_path="destinations.csv", models_dir="models"):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, "traveller_model.joblib")
        self.imputer_path = os.path.join(self.models_dir, "traveller_imputer.joblib")

        self.df_raw = self._load_csv(self.data_path)
        self.work = self._preprocess(self.df_raw)
        self.features = ["popularity", "rating", "num_attractions", "sub_mean_cost"]
        self.model, self.imputer = self._load_or_train()

    def _load_csv(self, path):
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        raise FileNotFoundError(f"Could not read {path}")

    def _preprocess(self, df):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}
        get = lambda *names: next((cols[n] for n in names if n in cols), None)
        loc_col = get("location", "location_name", "name", "place") or df.columns[0]
        cost_col = get("cost", "avg_cost_per_person", "avg_cost", "price", "price_per_person")
        rating_col = get("rating", "ratings", "rate")
        pop_col = get("popularity", "pop", "popularity_score")
        attractions_col = get("num_attractions", "attractions")

        work = df.rename(columns={loc_col: "location_name"}).copy()
        work["base_cost"] = pd.to_numeric(work[cost_col], errors="coerce").fillna(150) if cost_col else 150
        work["rating"] = pd.to_numeric(work[rating_col], errors="coerce").fillna(4) if rating_col else 4
        work["popularity"] = pd.to_numeric(work[pop_col], errors="coerce").fillna(50) if pop_col else 50
        work["num_attractions"] = pd.to_numeric(work[attractions_col], errors="coerce").fillna(3) if attractions_col else 3
        work["sub_mean_cost"] = work["base_cost"] * 0.3
        work["est_cost_1day_per_person"] = work["base_cost"] + work["sub_mean_cost"] * 0.5
        return work

    def _load_or_train(self):
        if os.path.exists(self.model_path) and os.path.exists(self.imputer_path):
            return joblib.load(self.model_path), joblib.load(self.imputer_path)
        X = self.work[self.features]
        y = self.work["est_cost_1day_per_person"]
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, self.model_path)
        joblib.dump(imp, self.imputer_path)
        return model, imp

    def get_best_recommendation(self, budget_per_person=200, days=2):
        df = self.work.copy()
        X_all = df[self.features]
        X_all_imp = self.imputer.transform(X_all)
        pred_1day = self.model.predict(X_all_imp)
        df["pred_cost_total_per_person"] = pred_1day * days

        def norm(c): return (c - c.min()) / (c.max() - c.min() + 1e-9)
        df["score"] = (1 - norm(df["pred_cost_total_per_person"])) * 0.6 + norm(df["rating"]) * 0.25 + norm(df["popularity"]) * 0.15
        candidates = df[df["pred_cost_total_per_person"] <= budget_per_person]
        if candidates.empty: return None
        best = candidates.sort_values("score", ascending=False).iloc[0]
        return {
            "location_name": best["location_name"],
            "pred_cost_total_per_person": float(best["pred_cost_total_per_person"]),
            "rating": float(best["rating"]),
            "popularity": float(best["popularity"]),
            "score": float(best["score"])
        }
