# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
import os
import json
import flask
import pandas as pd
from tensorflow.contrib import predictor
from sklearn.externals import joblib

from gamesbiz.resolve import paths


class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """This class method just checks if the model path is available to us"""

        if os.path.exists(paths.model('exported_model/')):
            cls.model = True
        else:
            cls.model = None

        return cls.model


app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():

    health = ScoringService.get_model() is not None

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """This method reads in the data (json object) sent with the request and returns a prediction
    as response """

    data = None
    export_path = 'exported_model/'

    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')

        data = pd.read_json(data, lines=True)

        X_scaler = joblib.load(os.path.join(paths.model('X_scaler.save')))
        scaled_data = X_scaler.transform(data.values)

        predict_fn = predictor.from_saved_model(paths.model(export_path))
        predictions = predict_fn({'input': scaled_data})
        prediction = predictions['earnings'][0][0]

        Y_scaler = joblib.load(os.path.join(paths.model('Y_scaler.save')))
        true_prediction = {'earnings': round((float(prediction) - Y_scaler.min_[0]) / Y_scaler.scale_[0], 3)}
        true_prediction = json.dumps(true_prediction)
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')

    result = true_prediction

    return flask.Response(response=result, status=200, mimetype='application/json')