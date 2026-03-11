from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# model load
model_path = os.path.join(os.getcwd(), "model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    # user se open price lena
    open_price = float(request.form["open_price"])

    # prediction
    prediction = model.predict([[open_price]])
    predicted_price = prediction[0]

    # trend check
    if predicted_price > open_price:
        trend = "Price may go HIGH 📈"
    else:
        trend = "Price may go LOW 📉"

    return render_template(
        "index.html",
        prediction_text=f"Predicted Price: {predicted_price:.2f}",
        trend_text=trend
    )

if __name__ == "__main__":
    app.run(debug=True)