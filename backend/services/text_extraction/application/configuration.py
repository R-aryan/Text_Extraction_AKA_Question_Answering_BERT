from injector import Module, singleton

from common.logging.console_loger import ConsoleLogger
from common.logging.logger import Logger
from services.text_extraction.settings import Settings


class Configuration(Module):
    def configure(self, binder):
        logger = ConsoleLogger(filename=Settings.LOGS_DIRECTORY)
        binder.bind(Logger, to=logger, scope=singleton)