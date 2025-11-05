def get_best_recommendation(self, budget_per_person=200, days=2):
    df = self.work.copy()

    # Predict 1-day costs
    X_all = df[self.features]
    X_all_imp = self.imputer.transform(X_all)
    pred_1day = self.model.predict(X_all_imp)
    df["pred_cost_total_per_person"] = pred_1day * days

    # Compute affordability ratio (how much of the budget is used)
    df["affordability"] = budget_per_person / (df["pred_cost_total_per_person"] + 1e-9)
    df["affordability"] = df["affordability"].clip(0, 1.2)  # limit ratio for extreme values

    # Weighted scoring system
    def normalize(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-9)

    df["score"] = (
        normalize(df["affordability"]) * 0.45 +
        normalize(df["rating"]) * 0.30 +
        normalize(df["popularity"]) * 0.20 +
        normalize(df["num_attractions"]) * 0.05
    )

    # Only keep destinations within ~20% over budget
    df = df[df["pred_cost_total_per_person"] <= budget_per_person * 1.2]

    if df.empty:
        return None

    best = df.sort_values("score", ascending=False).iloc[0]
    return {
        "location_name": best["location_name"],
        "pred_cost_total_per_person": float(best["pred_cost_total_per_person"]),
        "rating": float(best["rating"]),
        "popularity": float(best["popularity"]),
        "score": float(best["score"]),
    }
