from flask import Flask
from flask import jsonify
from flask import request
from utils import *

app = Flask(__name__)

# @app.route("/")
# def index():
#     return "<h1>Hello World</h1>"

@app.route("/crp", methods=["POST"])
def crp():
    
    data = request.get_json()
    
    try:
        training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number = get_data_from_json(data)


        predict_vector = rqa(training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number)
    except Exception:
    	return jsonify(error=[0])

    # training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number = get_data_from_json(data)
    # predict_vector = rqa(training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number)

    return jsonify(target=predict_vector)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
