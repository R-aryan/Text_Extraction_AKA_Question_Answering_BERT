import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from services.text_extraction.application.ai.training.src import utils


class Engine:
    def __init__(self):
        pass

    def loss_fn(self, start_logits, end_logits, start_positions, end_positions):
        l1 = nn.BCEWithLogitsLoss()(start_logits, start_positions)
        l2 = nn.BCEWithLogitsLoss()(end_logits, end_positions)
        total_loss = l1 + l2
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
            offsets_start = d["offsets_start"].numpy()
            offsets_end = d["offsets_end"].numpy()

            # moving tensors to device

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
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

            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score = self.calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets_start=offsets_start[px, :],
                    offsets_end=offsets_end[px, :]
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

    def eval_fn(self, data_loader, model, device):
        model.eval()
        losses = utils.AverageMeter()
        jaccards = utils.AverageMeter()
        all_outputs = []
        fin_outputs_start = []
        fin_outputs_end = []
        fin_outputs_start2 = []
        fin_outputs_end2 = []
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
                sentiment = d["sentiment"]
                orig_selected = d["orig_selected"]
                orig_tweet = d["orig_tweet"]
                targets_start = d["targets_start"]
                targets_end = d["targets_end"]
                offsets_start = d["offsets_start"].numpy()
                offsets_end = d["offsets_end"].numpy()

                # moving tensors to device
                ids = ids.to(device, dtype=torch.long)
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                mask = mask.to(device, dtype=torch.long)
                targets_start = targets_start.to(device, dtype=torch.long)
                targets_end = targets_end.to(device, dtype=torch.long)

                outputs_start, outputs_end = model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids
                )
                loss = self.loss_fn(outputs_start, outputs_end, targets_start, targets_end)
                outputs_start = torch.sigmoid(outputs_start).cpu().detach().numpy()
                outputs_end = torch.sigmoid(outputs_end).cpu().detach().numpy()
                jaccard_scores = []
                fin_outputs_start.append(torch.sigmoid(outputs_start).cpu().detach().numpy())
                fin_outputs_end.append(torch.sigmoid(outputs_end).cpu().detach().numpy())

                fin_padding_lens.extend(padding_len.cpu().detach().numpy().tolist())
                fin_tweet_token_ids.append(ids.cpu().detach().numpy().tolist())

                fin_tweet_tokens.extend(tweet_tokens)
                fin_orig_sentiment.extend(orig_sentiment)
                fin_orig_selected.extend(orig_selected)
                fin_orig_tweet.extend(orig_tweet)

                # for px, tweet in enumerate(orig_tweet):
                #     selected_tweet = orig_selected[px]
                #     tweet_sentiment = sentiment[px]
                #     jaccard_score = self.calculate_jaccard_score(
                #         original_tweet=tweet,
                #         target_string=selected_tweet,
                #         sentiment_val=tweet_sentiment,
                #         idx_start=np.argmax(outputs_start[px, :]),
                #         idx_end=np.argmax(outputs_end[px, :]),
                #         offsets_start=offsets_start[px, :],
                #         offsets_end=offsets_end[px, :]
                #     )
                #     jaccard_scores.append(jaccard_score)

                jaccards.update(np.mean(jaccard_scores), ids.size(0))
                losses.update(loss.item(), ids.size(0))
                tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

        print(f"Jaccard score = {jaccards.avg}")
        return jaccards.avg
