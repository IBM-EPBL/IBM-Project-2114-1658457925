from flask import Flask,render_template,request,redirect
import numpy as np
from tensorflow import keras
from keras.models import load_model
import joblib
import scipy
import requests


# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "qBVjfLpxSGroBr5cd8BNRjoC3pZhoHyrC9bKxMZ1IePl"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


app = Flask(__name__)
model = load_model('D:/19BCS018_ROJER T/IBM Proj/Project Development Phase/Sprint 4/Integrate flask with scoring end points/crudeoilprediction.h5')

@app.route('/',methods=["GET"])
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST","GET"])
def predict():
    if request.method == "POST":
        string = request.form['val']
        string = string.split(',')
        x_input = [eval(i) for i in string]
        

        sc = joblib.load("D:/19BCS018_ROJER T/IBM Proj/Project Development Phase/Sprint 4/Integrate flask with scoring end points/scaler.save") 

        x_input = sc.fit_transform(np.array(x_input).reshape(-1,1))

        x_input = np.array(x_input).reshape(1,-1)

        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1,10,1))
        b = x_input.tolist()
        print(b)

        model = load_model('D:/19BCS018_ROJER T/IBM Proj/Project Development Phase/Sprint 4/Integrate flask with scoring end points/crudeoilprediction.h5')
        output = model.predict(b)
        print(output[0][0])

        val = sc.inverse_transform(output)
        payload_scoring = {"input_data": [{   "values": [[b]]    }]}

        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/7179b8e9-1924-45b8-9dec-5f1ea85e6aa2/predictions?version=2022-11-14', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
        predictions = response_scoring.json()
        print(response_scoring.json())
        return render_template('index.html' , prediction = val[0][0])

if __name__=="__main__":
    app.run(debug=True)



