from flask import request
from injector import inject

from backend.services.text_extraction.api.controllers.controller import Controller
from services.text_extraction.application.ai.inference.prediction_manager import PredictionManager


class ParamsController(Controller):
    @inject
    def __init__(self, prediction: PredictionManager):
        self.predict = prediction

    def post(self):
        try:
            req_json = request.get_json()
            response = self.predict.run_inference(req_json)
            result = {'response': response}
            self.predict.logger.info('Request processed successfully--!!')
            return self.response_ok(result)
        except BaseException as ex:
            self.predict.logger.error('Error Occurred-- ' + str(ex))
            return self.response_error(str(ex))

    def get(self):
        return {'response': 'This is an API endpoint for text extraction---!!'}
