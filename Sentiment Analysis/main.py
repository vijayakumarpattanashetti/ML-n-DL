import pickle
from flask import Flask, request, jsonify
from model.ml_model import predict_score

#creating a flask app and naming it "text_scorer"
app = Flask('text_scorer')

@app.route('/predict', methods=['POST'])
def predict():
    txt = request.get_json()
    with open('./model/model.bin', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    result = predict_score(txt, model)

    result = {
        'score': str(result)
    }
    return jsonify(result)

@app.route('/call', methods=['GET'])
def call():
    return "Calling Model..."

