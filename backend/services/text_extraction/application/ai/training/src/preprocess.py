import re
import string
from nltk import word_tokenize

from backend.services.text_extraction.settings import Settings


class Preprocess:
    def __init__(self):
        self.settings = Settings

    def clean_text(self, text):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

    # def process_data(self, tweet, selected_text, sentiment, tokenizer, max_len):
    #     tweet = " ".join(str(tweet).split())
    #     selected_text = " ".join(str(selected_text).split())
    #
    #     len_st = len(selected_text)
    #     idx0 = None
    #     idx1 = None
    #     for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
    #         if tweet[ind: ind + len_st] == selected_text:
    #             idx0 = ind
    #             idx1 = ind + len_st
    #             break
    #
    #     char_targets = [0] * len(tweet)
    #     if idx0 is not None and idx1 is not None:
    #         for ct in range(idx0, idx1):
    #             char_targets[ct] = 1
    #
    #     tok_tweet = tokenizer.encode(tweet)
    #     input_ids_orig = tok_tweet.ids[1:-1]
    #     tweet_offsets = tok_tweet.offsets[1:-1]
    #
    #     target_idx = []
    #     for j, (offset1, offset2) in enumerate(tweet_offsets):
    #         if sum(char_targets[offset1: offset2]) > 0:
    #             target_idx.append(j)
    #
    #     targets_start = target_idx[0]
    #     targets_end = target_idx[-1]
    #
    #     input_ids = [101] + [self.settings.sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
    #     token_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
    #     mask = [1] * len(token_type_ids)
    #     tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
    #     targets_start += 3
    #     targets_end += 3
    #
    #     padding_length = max_len - len(input_ids)
    #     if padding_length > 0:
    #         input_ids = input_ids + ([0] * padding_length)
    #         mask = mask + ([0] * padding_length)
    #         token_type_ids = token_type_ids + ([0] * padding_length)
    #         tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    #
    #     return {
    #         'ids': input_ids,
    #         'mask': mask,
    #         'token_type_ids': token_type_ids,
    #         'targets_start': targets_start,
    #         'targets_end': targets_end,
    #         'orig_tweet': tweet,
    #         'orig_selected': selected_text,
    #         'sentiment': sentiment,
    #         'offsets': tweet_offsets
    #     }
