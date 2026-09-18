# app.py
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model/model.pkl"
VECT_PATH = "model/vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise SystemExit("Run train.py first to generate model files.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

def predict_message(msg):
    X = vectorizer.transform([msg])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return pred, float(prob[1])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if not message:
            return render_template("index.html", error="Please enter a message.")

        pred, score = predict_message(message)

        if pred == 1:
            result = "⚠️ Fake Placement Alert Detected"
            color = "red"
        else:
            result = "✅ Real / Official Message"
            color = "green"

        return render_template(
            "results.html",
            result=result,
            color=color,
            message=message,
            probability=f"{score*100:.1f}% fake"
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
# to run this click run button and it give http link open it