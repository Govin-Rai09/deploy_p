from flask import Flask, request, redirect, render_template, jsonify
import numpy as np
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_feat = [int(x) for x in request.form.values()]
    fin_feat = [np.array(int_feat)]
    pred = model.predict(fin_feat)
    output = round(pred[0], 2)

    return render_template('index.html', prediction_text='Classified as {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
