from flask import Flask, render_template, request
from joblib import load # Corrected import
import numpy as np
import pickle

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")
@app.route("/dataset")
def dataset():
    return render_template("chronickidneydisease.html")
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/model")
def model():
    return render_template("model.html")



@app.route("/predict", methods=["POST"])
def hell():
    age = float(request.form["age"])
    bp = float(request.form["bp"])
    sg = float(request.form["sg"])
    al = float(request.form["al"])
    su = float(request.form["su"])
    rbc = float(request.form["rbc"])
    pc = float(request.form["pc"])
    pcc = float(request.form["pcc"])
    ba = float(request.form["ba"])
    bgr = float(request.form["bgr"])
    bu = float(request.form["bu"])
    sc = float(request.form["sc"])
    sod = float(request.form["sod"])
    pot = float(request.form["pot"])
    hemo = float(request.form["hemo"])
    pcv = float(request.form["pcv"])
    wc = float(request.form["wc"])
    rc = float(request.form["rc"])
    htn = float(request.form["htn"])
    dm = float(request.form["dm"])
    cad = float(request.form["cad"])
    appet = float(request.form["appet"])
    pe = float(request.form["pe"])
    ane = float(request.form["ane"])

    features = np.array([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]])
    new_data_reshaped = np.array(features).reshape(1, -1)

    filename = 'kidney_logistic_regression.pkl'  # Replace with your actual filename

    # Open the file in binary read mode
    with open(filename, 'rb') as f:
      # Load the model from the file
      model = pickle.load(f)

    prediction = model.predict(new_data_reshaped)
    if prediction == 1:
        prediction_text = "You have Chronic Kidney Disease"
    else:
        prediction_text = "You don't have Chronic Kidney Disease"

    return render_template("output.html", prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)