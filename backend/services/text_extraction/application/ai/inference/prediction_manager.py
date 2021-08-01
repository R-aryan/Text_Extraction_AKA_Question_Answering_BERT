import numpy as np
import torch

from common.logging.console_loger import ConsoleLogger
from services.text_extraction.application.ai.model import BERTBaseUncased
from services.text_extraction.application.ai.training.src.dataset import TextExtractionDataset
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

    def __predict(self, data):
        try:
            self.logger.info(message="Performing prediction on the given data.")
            test_dataset = TextExtractionDataset(
                tweet=[data['sentence']],
                sentiment=[data['sentiment']],
                selected_text=[data['sentence']]
            )

            with torch.no_grad():
                d = test_dataset[0]
                ids = d["ids"]
                token_type_ids = d["token_type_ids"]
                mask = d["mask"]
                tweet_tokens = d["tweet_tokens"]
                padding_len = d["padding_len"]
                sentiment = d["sentiment"]
                orig_selected = d["orig_selected"]
                orig_sentiment = d["orig_sentiment"]
                orig_tweet = d["orig_tweet"]

                ids = ids.to(self.settings.DEVICE, dtype=torch.long).unsqueeze(0)
                token_type_ids = token_type_ids.to(self.settings.DEVICE, dtype=torch.long).unsqueeze(0)
                mask = mask.to(self.settings.DEVICE, dtype=torch.long).unsqueeze(0)
                sentiment = sentiment.to(self.settings.DEVICE, dtype=torch.float).unsqueeze(0)

                outputs_start, outputs_end = self.__model(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
                )

                return outputs_start, outputs_end, d

        except BaseException as ex:
            self.logger.error(message="Exception Occurred while prediction---!! " + str(ex))

    def __post_process(self, outputs_start, outputs_end, d):
        all_outputs = []
        fin_outputs_start = []
        fin_outputs_end = []
        fin_tweet_tokens = []
        fin_padding_lens = []
        fin_orig_selected = []
        fin_orig_sentiment = []
        fin_orig_tweet = []
        fin_tweet_token_ids = []

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        tweet_tokens = d["tweet_tokens"]
        padding_len = d["padding_len"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_sentiment = d["orig_sentiment"]
        orig_tweet = d["orig_tweet"]

        fin_outputs_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())
        fin_outputs_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())

        fin_padding_lens.append(padding_len.cpu().detach().numpy().tolist())
        fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())

        fin_tweet_tokens.append(tweet_tokens)
        fin_orig_sentiment.append(orig_sentiment)
        fin_orig_selected.append(orig_selected)
        fin_orig_tweet.append(orig_tweet)

        fin_outputs_start = np.vstack(fin_outputs_start)
        fin_outputs_end = np.vstack(fin_outputs_end)

        fin_tweet_token_ids = np.vstack(fin_tweet_token_ids)
        jaccards = []
        threshold = self.settings.threshold

        for j in range(len(fin_tweet_tokens)):
            target_string = fin_orig_selected[j]
            tweet_tokens = fin_tweet_tokens[j]
            padding_len = fin_padding_lens[j]
            original_tweet = fin_orig_tweet[j]
            sentiment_val = fin_orig_sentiment[j]

            if padding_len > 0:
                mask_start = fin_outputs_start[j, 3:-1][:-padding_len] >= threshold
                mask_end = fin_outputs_end[j, 3:-1][:-padding_len] >= threshold
                tweet_token_ids = fin_tweet_token_ids[j, 3:-1][:-padding_len]
            else:
                mask_start = fin_outputs_start[j, 3:-1] >= threshold
                mask_end = fin_outputs_end[j, 3:-1] >= threshold
                tweet_token_ids = fin_tweet_token_ids[j, 3:-1]

            mask = [0] * len(mask_start)
            idx_start = np.nonzero(mask_start)[0]
            idx_end = np.nonzero(mask_end)[0]
            if len(idx_start) > 0:
                idx_start = idx_start[0]
                if len(idx_end) > 0:
                    idx_end = idx_end[0]
                else:
                    idx_end = idx_start
            else:
                idx_start = 0
                idx_end = 0

            for mj in range(idx_start, idx_end + 1):
                mask[mj] = 1

            output_tokens = [x for p, x in enumerate(tweet_token_ids) if mask[p] == 1]

            filtered_output = self.settings.TOKENIZER.decode(output_tokens)
            filtered_output = filtered_output.strip().lower()

            if sentiment_val == "neutral":
                filtered_output = original_tweet

            all_outputs.append(filtered_output.strip())

        return all_outputs

    def run_inference(self, data):
        self.logger.info("Received " + str(data) + " for inference--!!")
        outputs_start, outputs_end, d = self.__predict(data)
        result = self.__post_process(outputs_start, outputs_end, d)
        return result
