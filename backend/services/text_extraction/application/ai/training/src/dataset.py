import torch
import pandas as pd
import numpy as np

from services.text_extraction.application.ai.training.src.preprocess import Preprocess
from services.text_extraction.settings import Settings


class TextExtractionDataset:
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
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())

        len_st = len(selected_text)
        idx0 = -1
        idx1 = -1
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0, idx1 + 1):
                if tweet[j] != " ":
                    char_targets[j] = 1

        tok_tweet = self.tokenizer.encode(sequence=self.sentiment[item], pair=tweet)
        tok_tweet_tokens = tok_tweet.tokens
        tok_tweet_ids = tok_tweet.ids
        tok_tweet_offsets = tok_tweet.offsets[3:-1]
        # print(tok_tweet_tokens)
        # print(tok_tweet.offsets)
        # ['[CLS]', 'spent', 'the', 'entire', 'morning', 'in', 'a', 'meeting', 'w', '/',
        # 'a', 'vendor', ',', 'and', 'my', 'boss', 'was', 'not', 'happy', 'w', '/', 'them',
        # '.', 'lots', 'of', 'fun', '.', 'i', 'had', 'other', 'plans', 'for', 'my', 'morning', '[SEP]']
        targets = [0] * (len(tok_tweet_tokens) - 4)
        if self.sentiment[item] == "positive" or self.sentiment[item] == "negative":
            sub_minus = 8
        else:
            sub_minus = 7

        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1 - sub_minus:offset2 - sub_minus]) > 0:
                targets[j] = 1

        targets = [0] + [0] + [0] + targets + [0]

        # print(tweet)
        # print(selected_text)
        # print([x for i, x in enumerate(tok_tweet_tokens) if targets[i] == 1])
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_end[non_zero[-1]] = 1

        # print(targets_start)
        # print(targets_end)

        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0] * 3 + [1] * (len(tok_tweet_ids) - 3)

        padding_length = self.max_len - len(tok_tweet_ids)
        ids = tok_tweet_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        targets = targets + ([0] * padding_length)
        targets_start = targets_start + ([0] * padding_length)
        targets_end = targets_end + ([0] * padding_length)

        sentiment = [1, 0, 0]
        if self.sentiment[item] == "positive":
            sentiment = [0, 0, 1]
        if self.sentiment[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'tweet_tokens': " ".join(tok_tweet_tokens),
            'targets': torch.tensor(targets, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            'padding_len': torch.tensor(padding_length, dtype=torch.long),
            'orig_tweet': self.tweet[item],
            'orig_selected': self.selected_text[item],
            'sentiment': torch.tensor(sentiment, dtype=torch.float),
            'orig_sentiment': self.sentiment[item]
        }


if __name__ == "__main__":
    df = pd.read_csv(Settings.TRAIN_DATA)
    df = df.dropna().reset_index(drop=True)
    dset = TextExtractionDataset(tweet=df.text.values, sentiment=df.sentiment.values,
                                 selected_text=df.selected_text.values)
    print(dset[100])
