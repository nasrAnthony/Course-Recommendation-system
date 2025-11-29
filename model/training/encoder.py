import torch
import torch.nn as nn
from transformers import AutoModel

# Model: BERT + Projection Head

MODEL_NAME = "bert-base-uncased"
PROJ_DIM = 256


class CourseEncoder(nn.Module):
    def __init__(self, base_model_name: str = MODEL_NAME, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_name)
        hidden = self.bert.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        last_hidden = out.last_hidden_state  # (B, L, H)

        # mean pooling
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts                    # (B, H)

        z = self.proj(pooled)
        z = nn.functional.normalize(z, p=2, dim=-1)
        return z