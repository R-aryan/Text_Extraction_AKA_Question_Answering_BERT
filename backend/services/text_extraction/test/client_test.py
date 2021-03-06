from common.logging.console_loger import ConsoleLogger
from services.text_extraction.application.ai.inference.prediction_manager import PredictionManager
from services.text_extraction.application.ai.training.src.preprocess import Preprocess
from services.text_extraction.settings import Settings
import pandas as pd

p1 = PredictionManager(preprocess=Preprocess(), logger=ConsoleLogger(filename=Settings.LOGS_DIRECTORY))
df = pd.read_csv(Settings.TEST_DATA).dropna().reset_index(drop=True)
# df_s = pd.read_csv(Settings.SUBMISSION_DATA).dropna().reset_index(drop=True)
index = 15
tweet = df.text.values
sentiment = df.sentiment.values
# sub = df_s.selected_text.values

data = {
    'sentiment': sentiment[index],
    'sentence': tweet[index]

}

print("Sample Input, ", data)
output = p1.run_inference(data)
print(output)
# print(sub[index])
