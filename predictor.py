import os, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class TravelRecommender:
    def __init__(self, data_path="destinations.csv", models_dir="models"):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, "traveller_model.joblib")
        self.imputer_path = os.path.join(self.models_dir, "traveller_imputer.joblib")
        self.df_raw = self._load_csv()
        self.work = self._preprocess(self.df_raw)
        self.features = ["popularity", "rating", "num_attractions", "sub_mean_cost"]
        self.model, self.imputer = self._load_or_train()

    def _load_csv(self):
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                return pd.read_csv(self.data_path, encoding=enc)
            except Exception:
                continue
        raise FileNotFoundError("Cannot read destinations.csv")

    def _preprocess(self, df):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}
        get = lambda *names: next((cols[n] for n in names if n in cols), None)
        loc_col = get("location", "location_name", "name", "place") or df.columns[0]
        cost_col = get("cost", "avg_cost", "avg_cost_per_person", "price")
        rating_col = get("rating", "rate")
        pop_col = get("popularity", "popularity_score")
        attract_col = get("num_attractions", "attractions")
        df = df.rename(columns={loc_col: "location_name"})
        df["base_cost"] = pd.to_numeric(df.get(cost_col, 150), errors="coerce").fillna(150)
        df["rating"] = pd.to_numeric(df.get(rating_col, 4), errors="coerce").fillna(4)
        df["popularity"] = pd.to_numeric(df.get(pop_col, 50), errors="coerce").fillna(50)
        df["num_attractions"] = pd.to_numeric(df.get(attract_col, 3), errors="coerce").fillna(3)
        df["sub_mean_cost"] = df["base_cost"] * 0.3
        df["est_cost_1day_per_person"] = df["base_cost"] + df["sub_mean_cost"] * 0.5
        return df

    def _load_or_train(self):
        if os.path.exists(self.model_path) and os.path.exists(self.imputer_path):
            return joblib.load(self.model_path), joblib.load(self.imputer_path)
        X = self.work[self.features]
        y = self.work["est_cost_1day_per_person"]
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_imp, y)
        joblib.dump(model, self.model_path)
        joblib.dump(imp, self.imputer_path)
        return model, imp

    def get_best_recommendation(self, budget_per_person=200, days=2):
        df = self.work.copy()
        X_imp = self.imputer.transform(df[self.features])
        pred_1day = self.model.predict(X_imp)
        df["pred_cost_total_per_person"] = pred_1day * days
        df["affordability"] = budget_per_person / (df["pred_cost_total_per_person"] + 1e-9)
        df["affordability"] = df["affordability"].clip(0, 1.2)
        def normalize(col): return (col - col.min()) / (col.max() - col.min() + 1e-9)
        df["score"] = (
            normalize(df["affordability"]) * 0.45 +
            normalize(df["rating"]) * 0.30 +
            normalize(df["popularity"]) * 0.20 +
            normalize(df["num_attractions"]) * 0.05
        )
        df = df[df["pred_cost_total_per_person"] <= budget_per_person * 1.2]
        if df.empty: return None
        best = df.sort_values("score", ascending=False).iloc[0]
        return dict(location_name=best["location_name"],
                    pred_cost_total_per_person=float(best["pred_cost_total_per_person"]),
                    rating=float(best["rating"]),
                    popularity=float(best["popularity"]),
                    score=float(best["score"]))
