import torch
import pandas as pd
import numpy as np

from services.text_extraction.application.ai.training.src.preprocess import Preprocess
from services.text_extraction.settings import Settings


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.settings = Settings
        self.preprocess = Preprocess()
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = self.settings.TOKENIZER
        self.max_len = self.settings.MAX_LEN

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = self.preprocess.process_data(
            tweet=self.tweet[item],
            selected_text=self.selected_text[item],
            sentiment=self.sentiment[item],
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets_start': torch.tensor([x for x, _ in data["offsets"]], dtype=torch.long),
            'offsets_end': torch.tensor([x for _, x in data["offsets"]], dtype=torch.long)
        }

#
# if __name__ == "__main__":
#     df = pd.read_csv(Settings.TRAIN_DATA)
#     df = df.dropna().reset_index(drop=True)
#     dset = TweetDataset(tweet=df.text.values, sentiment=df.sentiment.values, selected_text=df.selected_text.values)
#     print(dset[100])
