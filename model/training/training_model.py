import os
import random
import math
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from dataset import ContrastiveCourseDataset
from encoder import CourseEncoder
from loss import SupervisedNTXentLoss, isotropy_regularizer


# Paths and Parameters -----------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "data", "cleaned_courses.csv")
TEXT_COLUMN = "TextForBERT"

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
PROJ_DIM = 256
TEMPERATURE = 0.2 # weight for similarity in contrastive loss
LAMBDA_ISO = 0.05  # weight for isotropy regularizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def set_seed(seed=42):
    """Seeding data for cleaner runs"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

def train():
    # Load data --------------------------------------------------------------
    df = pd.read_csv(CSV_PATH)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Column {TEXT_COLUMN} not found in CSV.")

    texts = df[TEXT_COLUMN].astype(str).tolist()

    print(f"Loaded {len(texts)} course descriptions.")

    # Build faculty labels ----------------------------------------------------
    if "Faculty" not in df.columns:
        raise ValueError("Expected a 'Faculty' column in the CSV for labels.")

    faculties = df["Faculty"].astype(str).tolist()
    full_codes = df["Faculty"].astype(str) + " " + df["Code"].astype(str)
    unique_faculties = sorted(set(faculties))
    fac2id = {fac: i for i, fac in enumerate(unique_faculties)}
    labels = [fac2id[f] for f in faculties]    # list[int], same length as texts

    print(f"Found {len(unique_faculties)} unique faculties.")

    # Tokenizing course data --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = ContrastiveCourseDataset(texts, labels, tokenizer, full_codes, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Initialization -----------------------------------------------------------
    # Model: BERT
    # Optimizer: Adam 
    # Scheduler: for learning rate
    # Loss: contrastive loss
    model = CourseEncoder().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = SupervisedNTXentLoss(temperature=TEMPERATURE)

    model.train()
    step = 0

    # Training Loop ------------------------------------------------------------
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_ctr = 0.0      # for logging
        running_iso = 0.0      # for logging

        for batch in dataloader:
            step += 1
            # loading a batch of tokenized data
            input_ids_a      = batch["input_ids_a"].to(DEVICE)
            attention_mask_a = batch["attention_mask_a"].to(DEVICE)
            input_ids_b      = batch["input_ids_b"].to(DEVICE)
            attention_mask_b = batch["attention_mask_b"].to(DEVICE)
            labels_batch     = batch["label"].to(DEVICE)

            optimizer.zero_grad() # resetting gradients

            z_i = model(input_ids_a, attention_mask_a)  # (N, D) normalized
            z_j = model(input_ids_b, attention_mask_b)  # (N, D) normalized

            # supervised contrastive loss (faculty labels also considered positive)
            loss_contrastive = criterion(z_i, z_j, labels_batch)

            # isotropy regularizer on all embeddings in the batch
            z_all = torch.cat([z_i, z_j], dim=0)        # (2N, D)
            loss_iso = isotropy_regularizer(z_all)

            # total loss
            loss = loss_contrastive + LAMBDA_ISO * loss_iso

            # backward propagation and updating weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_ctr  += loss_contrastive.item()
            running_iso  += loss_iso.item()

            # batch average data and printing
            if step % 10 == 0:
                avg_loss = running_loss / 10
                avg_ctr  = running_ctr  / 10
                avg_iso  = running_iso  / 10
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | Step {step} | "
                    f"Loss: {avg_loss:.4f} "
                    f"(ctr: {avg_ctr:.4f}, iso: {avg_iso:.4f})"
                )
                running_loss = 0.0
                running_ctr  = 0.0
                running_iso  = 0.0

    # Save model --------------------------------------------------------------
    save_dir = os.path.join(BASE_DIR, "model", "model_v7_0.2")
    os.makedirs(save_dir, exist_ok=True)

    # Save both BERT + projection head
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(save_dir)

    # Also save config for reloading later
    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        f.write(f"MODEL_NAME={MODEL_NAME}\n")
        f.write(f"PROJ_DIM={PROJ_DIM}\n")
        f.write(f"MAX_LEN={MAX_LEN}\n")

    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    train()

