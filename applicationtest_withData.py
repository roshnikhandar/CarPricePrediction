from flask import Flask, render_template, request
import sklearn
import pandas as pd
import pickle
import numpy as np

#car = pd.read_csv(r"Cleaned_data.csv")

with open("LinearRegressionModelRoshni.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
@app.route('/predict', methods= ['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    return str(np.around(prediction[0],2))

if __name__ == "__main__":
    app.run()