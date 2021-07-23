from flask import request
from injector import inject

from services.text_extraction.api.controllers.controller import Controller


class ParamsController(Controller):
    @inject
    def __init__(self):
        pass

    def get(self):
        return {'response': 'This is an API endpoint for text extraction using BERT---!!'}
