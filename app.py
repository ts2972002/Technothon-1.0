import numpy as np
import pickle
from flask import Flask, request, render_template

with open('model_pickle.pkl','rb') as f:
   model = pickle.load(f)
   f.close()

# print(model)

app = Flask(__name__)




@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():
 
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)
 
    output = prediction
 
    # Check the output values and retrieve the result with html tag based on the value
    if output == 1:
        return render_template('index.html', result = 'The person is having stress!')
    else:
        return render_template('index.html', result = 'The person is not having  stress!')

if __name__ == '__main__':
    app.run()