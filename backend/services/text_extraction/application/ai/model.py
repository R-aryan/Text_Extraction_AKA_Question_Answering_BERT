import torch
import transformers
import torch.nn as nn

from services.text_extraction.settings import Settings


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.settings = Settings
        self.bert = transformers.BertModel.from_pretrained(self.settings.BERT_PATH, return_dict=False)
        self.l0 = nn.Linear(self.settings.input_dim, self.settings.output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # not using sentiment at all
        sequence_output, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # (batch_size, num_tokens, 768)
        logits = self.l0(sequence_output)
        # (batch_size, num_tokens, 2)
        # (batch_size, num_tokens, 1), (batch_size, num_tokens, 1)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        # (batch_size, num_tokens), (batch_size, num_tokens)

        return start_logits, end_logits
