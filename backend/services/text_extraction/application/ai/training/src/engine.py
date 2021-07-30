import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import string
from services.text_extraction.application.ai.training.src import utils
from services.text_extraction.settings import Settings


class Engine:
    def __init__(self):
        self.settings = Settings

    def loss_fn(self, start_logits, end_logits, start_positions, end_positions):
        l1 = nn.BCEWithLogitsLoss()(start_logits, start_positions)
        l2 = nn.BCEWithLogitsLoss()(end_logits, end_positions)
        total_loss = (l1 + l2)
        return total_loss

    def set_seed(self, seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def calculate_jaccard_score(self,
                                original_tweet,
                                target_string,
                                sentiment_val,
                                idx_start,
                                idx_end,
                                offsets_start,
                                offsets_end,
                                verbose=False):

        offsets = list(zip(offsets_start, offsets_end))

        if idx_end < idx_start:
            idx_end = idx_start

        filtered_output = ""
        original_tweet_sp = " ".join(original_tweet.split())
        for ix in range(idx_start, idx_end + 1):
            if offsets[ix][0] == 0 and offsets[ix][1] == 0:
                continue
            filtered_output += original_tweet_sp[offsets[ix][0]: offsets[ix][1]]
            if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
                filtered_output += " "

        filtered_output = filtered_output.replace(" .", ".")
        filtered_output = filtered_output.replace(" ?", "?")
        filtered_output = filtered_output.replace(" !", "!")
        filtered_output = filtered_output.replace(" ,", ",")
        filtered_output = filtered_output.replace(" ' ", "'")
        filtered_output = filtered_output.replace(" n't", "n't")
        filtered_output = filtered_output.replace(" 'm", "'m")
        filtered_output = filtered_output.replace(" do not", " don't")
        filtered_output = filtered_output.replace(" 's", "'s")
        filtered_output = filtered_output.replace(" 've", "'ve")
        filtered_output = filtered_output.replace(" 're", "'re")

        if sentiment_val == "neutral":
            filtered_output = original_tweet

        if sentiment_val != "neutral" and verbose == True:
            if filtered_output.strip().lower() != target_string.strip().lower():
                print("********************************")
                print(f"Output= {filtered_output.strip()}")
                print(f"Target= {target_string.strip()}")
                print(f"Tweet= {original_tweet.strip()}")
                print("********************************")

        jac = utils.jaccard(target_string.strip(), filtered_output.strip())
        return jac

    def train_fn(self, data_loader, model, optimizer, device, schedular):
        print("Starting training...\n")
        model.train()
        losses = utils.AverageMeter()
        jaccards = utils.AverageMeter()
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]

            # moving tensors to device

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.float)
            targets_end = targets_end.to(device, dtype=torch.float)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            outputs_start, outputs_end = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids,
            )

            loss = self.loss_fn(outputs_start, outputs_end, targets_start, targets_end)

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc
            optimizer.step()
            # Update the learning rate
            schedular.step()

            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)

    def eval_fn(self, data_loader, model, device):
        model.eval()
        losses = utils.AverageMeter()
        jaccards = utils.AverageMeter()
        all_outputs = []
        fin_outputs_start = []
        fin_outputs_end = []
        fin_tweet_tokens = []
        fin_padding_lens = []
        fin_orig_selected = []
        fin_orig_sentiment = []
        fin_orig_tweet = []
        fin_tweet_token_ids = []
        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader))
            for bi, d in enumerate(tk0):
                ids = d["ids"]
                token_type_ids = d["token_type_ids"]
                mask = d["mask"]
                tweet_tokens = d["tweet_tokens"]
                padding_len = d["padding_len"]
                sentiment = d["sentiment"]
                orig_selected = d["orig_selected"]
                orig_sentiment = d["orig_sentiment"]
                orig_tweet = d["orig_tweet"]

                ids = ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets_start = targets_start.to(device, dtype=torch.float)
                targets_end = targets_end.to(device, dtype=torch.float)
                sentiment = sentiment.to(device, dtype=torch.float)

                outputs_start, outputs_end = model(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=token_type_ids
                )
                loss = self.loss_fn(outputs_start, outputs_end, targets_start, targets_end)

                fin_outputs_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())
                fin_outputs_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())

                fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
                fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())

                fin_tweet_tokens.extend(tweet_tokens)
                fin_orig_sentiment.extend(orig_sentiment)
                fin_orig_selected.extend(orig_selected)
                fin_orig_tweet.extend(orig_tweet)

                losses.update(loss.item(), ids.size(0))
                tk0.set_postfix(loss=losses.avg)

            fin_outputs_start = np.vstack(fin_outputs_start)
            fin_outputs_end = np.vstack(fin_outputs_end)
            fin_tweet_token_ids = np.vstack(fin_tweet_token_ids)
            threshold = self.settings.threshold

            jaccard_scores = []

            for j in range(len(fin_tweet_tokens)):
                target_string = fin_orig_selected[j]
                tweet_tokens = fin_tweet_tokens[j]
                padding_len = fin_padding_lens[j]
                original_tweet = fin_orig_tweet[j]
                sentiment_val = fin_orig_sentiment[j]

                if padding_len > 0:
                    mask_start = fin_outputs_start[j, :][:-padding_len] >= threshold
                    mask_end = fin_outputs_end[j, :][:-padding_len] >= threshold

                else:
                    mask_start = fin_outputs_start[j, 3:-1] >= threshold
                    mask_end = fin_outputs_end[j, 3:-1] >= threshold

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

                output_tokens = [x for p, x in enumerate(tweet_tokens.split()) if mask[p] == 1]
                output_tokens = [x for x in output_tokens if x not in self.settings.SPECIAL_TOKENS]

                final_output = ""
                for ot in output_tokens:
                    if ot.startswith("##"):
                        final_output += ot[2:]
                    elif len(ot) == 1 and ot in string.punctuation:
                        final_output += ot
                    else:
                        final_output += " " + ot

                final_output = final_output.strip()

                if sentiment == "neutral" or len(original_tweet.split())<4:
                    final_output = original_tweet

                jac = utils.jaccard(target_string.strip(),final_output.strip())

                jaccard_scores.append(jac)

        mean_jac = np.mean(jaccard_scores)
        # print(f"Jaccard score = {mean_jac}")
        return mean_jac
