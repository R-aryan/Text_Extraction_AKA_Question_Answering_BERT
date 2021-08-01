import torch

from common.logging.console_loger import ConsoleLogger
from services.text_extraction.application.ai.model import BERTBaseUncased
from services.text_extraction.application.ai.training.src.preprocess import Preprocess
from services.text_extraction.settings import Settings


class PredictionManager:
    def __init__(self, preprocess: Preprocess, logger: ConsoleLogger):
        self.preprocess = preprocess
        self.logger = logger
        self.settings = Settings
        self.__model = None
        self.__load_model()

    def __load_model(self):
        try:
            self.logger.info(message="Loading Bert Base Uncased Model for text extraction.")
            self.__model = BERTBaseUncased()
            self.logger.info(message="Bert Base Model Successfully Loaded.")
            self.logger.info(message="Loading Model trained Weights.")
            self.__model.load_state_dict(torch.load(self.settings.WEIGHTS_PATH,
                                                    map_location=torch.device(self.settings.DEVICE)))
            self.__model.to(self.settings.DEVICE)
            self.__model.eval()
            self.logger.info(message="Model Weights loaded Successfully--!!")

        except BaseException as ex:
            self.logger.error(message="Exception Occurred while loading model---!! " + str(ex))