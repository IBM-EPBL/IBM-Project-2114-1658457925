from flask import Flask,render_template,request,redirect
import numpy as np
from tensorflow import keras
from keras.models import load_model
from io import BytesIO
import joblib
import scipy


app = Flask(__name__)
model = load_model('crudeoilprediction.h5')

@app.route('/',methods=["GET"])
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST","GET"])
def predict():
    if request.method == "POST":
        string = request.form['val']
        string = string.split(',')
        x_input = [eval(i) for i in string]
        

        sc = joblib.load("scaler.save") 

        x_input = sc.fit_transform(np.array(x_input).reshape(-1,1))

        x_input = np.array(x_input).reshape(1,-1)

        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1,10,1))
        print(x_input.shape)

        model = load_model('crudeoilprediction.h5')
        output = model.predict(x_input)
        print(output[0][0])

        val = sc.inverse_transform(output)
        
        return render_template('index.html' , prediction = val[0][0])
    if request.method=="GET":
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)



