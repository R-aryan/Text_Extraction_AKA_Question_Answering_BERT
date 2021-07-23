from flask import request
from injector import inject

from backend.services.text_extraction.api.controllers.controller import Controller


class ParamsController(Controller):
    @inject
    def __init__(self):
        self.predict = 'text'

    def post(self):
        return {'response': 'This is an API endpoint for text extraction--!!'}

    def get(self):
        return {'response': 'This is an API endpoint for text extraction---!!'}
