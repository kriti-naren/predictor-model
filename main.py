from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
app = Flask(__name__)


@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=['POST'])
def predict():
    a = request.form.get('a', default=0, type=int)
    b = request.form.get('b', default=0, type=int)
    c = request.form.get('c', default=0, type=int)
    d = request.form.get('d', default=0, type=int)
    e = request.form.get('e', default=0, type=int)
    f = request.form.get('f', default=0, type=int)
    g = request.form.get('g', default=0, type=int)
    h = request.form.get('h', default=0, type=int)
    i = request.form.get('i', default=0, type=int)
    j = request.form.get('j', default=0, type=int)
    k = request.form.get('k', default=0, type=int)

    input_query = [a, b, c, d, e, f, g, h, i, j, k]

    result = (np.array(input_query)).reshape(1,-1)
    result_predict = model.predict(result)
    ans = result_predict.tolist()
    return jsonify(prediction= ans)

if __name__ == '__main__':
    app.run(debug=True)