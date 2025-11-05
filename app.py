from flask import Flask, render_template, request, jsonify
from predictor import TravelRecommender

app = Flask(__name__, template_folder="templates", static_folder="static")

recommender = TravelRecommender(data_path="destinations.csv", models_dir="models")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        budget = float(data.get("budget", 200))
        days = int(data.get("days", 2))
        result = recommender.get_best_recommendation(budget_per_person=budget, days=days)

        if result is None:
            return jsonify({"ok": False, "message": "No destinations fit your budget."}), 200

        return jsonify({"ok": True, "result": result}), 200
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
