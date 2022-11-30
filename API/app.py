
from flask import Flask, request, json, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app= Flask(__name__)
model_path = "model.joblib"
model = joblib.load(model_path)

@app.route('/',methods=['GET']) # the template of the home page of the app
def home():
    return render_template("home.html")

@app.route("/help", methods=["GET"])
def docu():
    return render_template("help.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_f = pd.DataFrame([final])
    prediction = model.predict(data_f)
    prediction = int(prediction[0])
    print('la prevision est : ', prediction)
    return render_template('home.html', pred='THE WINE QUALITY : {} / 10'.format(prediction))


@app.route('/api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict(data["input"])
    output = [float(x) for x in prediction]
    return jsonify({"predictions": output}),200       

if __name__ == "__main__":
    app.run(debug=True)