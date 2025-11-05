document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("btn");
  btn.addEventListener("click", async () => {
    const budget = parseFloat(document.getElementById("budget").value);
    const days = parseInt(document.getElementById("days").value);
    document.getElementById("status").textContent = "Predicting...";
    try {
      const res = await fetch("/predict", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({budget, days})
      });
      const data = await res.json();
      if (!data.ok) {
        document.getElementById("status").textContent = data.message;
        return;
      }
      const r = data.result;
      document.getElementById("r_loc").textContent = r.location_name;
      document.getElementById("r_cost").textContent = r.pred_cost_total_per_person.toFixed(2);
      document.getElementById("r_rate").textContent = r.rating.toFixed(1);
      document.getElementById("r_pop").textContent = r.popularity.toFixed(0);
      document.getElementById("r_score").textContent = r.score.toFixed(3);
      document.getElementById("result").classList.remove("hidden");
      document.getElementById("status").textContent = "Done.";
    } catch (e) {
      document.getElementById("status").textContent = "Error: " + e.message;
    }
  });
});
