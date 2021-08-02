from flask_injector import FlaskInjector
from backend.services.text_extraction.api.server import server
from backend.services.text_extraction.application.configuration import Configuration
from backend.services.text_extraction.api.controllers.params_controller import ParamsController

api_name = '/text_extraction/api/v1/'

server.api.add_resource(ParamsController, api_name + 'predict', methods=["GET", "POST"])

flask_injector = FlaskInjector(app=server.app, modules=[Configuration])
