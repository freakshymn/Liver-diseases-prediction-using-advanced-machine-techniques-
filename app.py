from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

try:
    sc = pickle.load(open('sc.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading pickled objects: {e}")
    sc, model = None, None

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/predict', methods=['POST'])
def predict():
    if sc is None or model is None:
        return "Error: Model or scaler not loaded properly. Check your files."

    try:
        inputs = [float(x) for x in request.form.values()]
    except ValueError:
        return "Error: Invalid input data. Please provide numeric values."

    inputs = np.array([inputs])
    inputs = sc.transform(inputs)
    output = model.predict(inputs)

    # Assuming the output is a probability between 0 and 1
    if output < 0.5:
        output = 0
    else:
        output = 1

    return render_template('end.html', prediction=output)

if __name__ == '__main__':
    app.run(debug=True)
