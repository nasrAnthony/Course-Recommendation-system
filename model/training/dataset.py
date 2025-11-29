from torch.utils.data import Dataset
import torch
from typing import List
from augmentation import make_two_views



class ContrastiveCourseDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, full_codes: List[str], max_len: int):
        """
        texts:  list of course descriptions (len = N)
        labels: list of int faculty IDs aligned with texts (len = N)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.course_codes = full_codes

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        base_text = self.texts[idx]
        view1, view2 = make_two_views(base_text, self.course_codes[idx])

        encoded1 = self.tokenizer(
            view1,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded2 = self.tokenizer(
            view2,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids_a":      encoded1["input_ids"].squeeze(0),
            "attention_mask_a": encoded1["attention_mask"].squeeze(0),
            "input_ids_b":      encoded2["input_ids"].squeeze(0),
            "attention_mask_b": encoded2["attention_mask"].squeeze(0),
            "label":            torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item