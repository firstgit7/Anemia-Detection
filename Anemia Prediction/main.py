from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gender = float(request.form.get("gender"))
    hemoglobin = float(request.form.get("hemoglobin"))
    mch = float(request.form.get("mch"))
    mchc = float(request.form.get("mchc"))
    mcv = float(request.form.get("mcv"))

    X = np.array([[gender, hemoglobin, mch, mchc, mcv]])
    pred = model.predict(X)[0]

    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0][int(pred)]) * 100

    if int(pred) == 1:
        label = "Anemia Detected"
        description = "Your pattern suggests anemia. Please consult a doctor for confirmation and guidance."
        color = "#ff4b5c"
    else:
        label = "No Anemia Detected"
        description = "Your pattern does not suggest anemia in this model. Keep following medical advice and regular checkups."
        color = "#00c48c"

    inputs = {
        "Gender": "Male" if gender == 1 else "Female",
        "Hemoglobin (g/dL)": hemoglobin,
        "MCH (pg)": mch,
        "MCHC (g/dL)": mchc,
        "MCV (fL)": mcv,
    }

    return render_template(
        "predict.html",
        label=label,
        description=description,
        probability=prob,
        color=color,
        inputs=inputs,
    )

if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000)

