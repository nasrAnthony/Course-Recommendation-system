# Base BERT with some testing and visualization
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import umap
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Initialization -----------------------------------------------
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # eval mode

# Builfing the model -----------------------------------------------
def mean_pooling(model_output, attention_mask):
    '''
    Gets an averaged embedding for an input sentence
    '''
    token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * input_mask_expanded).sum(dim=1)
    counts = input_mask_expanded.sum(dim=1)
    return summed / counts


@torch.no_grad()
def encode_texts(texts, batch_size=16, device="cpu", normalize=True):
    """
    The actual "Vanilla BERT Model"
    texts: list of strings
    returns: numpy array (n_texts, hidden_dim)
    """
    model.to(device)
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        outputs = model(**encoded)
        sentence_embeddings = mean_pooling(outputs, encoded["attention_mask"])  # (B, hidden)
        if normalize:
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        all_embs.append(sentence_embeddings.cpu().numpy())

    return np.vstack(all_embs)

# Running the model -----------------------------------------------

df = pd.read_csv("data/cleaned_courses.csv") # loading bert-ready text
texts = df["TextForBERT"].astype(str).tolist()

# embedding the course data
embeddings = encode_texts(texts, batch_size=16) 
print(embeddings.shape)  # (num_courses, 768)

# Cosine Similarity visualization -----------------------------------------------
n = min(400, embeddings.shape[0])
sub_embs = embeddings[:n]
sim_matrix = cosine_similarity(sub_embs, sub_embs) # similariy of course data with itself

# Take only upper triangle (without diagonal) to avoid duplicates/self-similarity
i_upper = np.triu_indices(n, k=1)
sims = sim_matrix[i_upper]

print("Mean cosine similarity:", sims.mean())
print("Std cosine similarity:", sims.std())

plt.figure(figsize=(6, 4))
plt.hist(sims, bins=50, alpha=0.8)
plt.title("Cosine similarity distribution (Vanilla BERT on courses)")
plt.xlabel("Cosine similarity between random course pairs")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()

# UMAP and PCA visualization -----------------------------------------------
######## UMAP ########
reducer_v = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
)
emb_v_umap = reducer_v.fit_transform(embeddings)

# Build prefixes and for the faculties
prefixes = df["Faculty"].str.extract(r"([A-Za-z]+)")[0]
unique_prefixes = sorted(prefixes.unique())
mapping = {p: i for i, p in enumerate(unique_prefixes)}
colors = prefixes.map(mapping).values
cmap = plt.get_cmap("tab20").resampled(len(unique_prefixes))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(
    emb_v_umap[:, 0],
    emb_v_umap[:, 1],
    s=10,
    c=colors,
    cmap=cmap,
    alpha=0.8,
)
plt.title("Vanilla BERT – UMAP (colored by course prefix)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)

######## PCA ########
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

plt.subplot(1, 2, 2)
plt.scatter(
    emb_2d[:, 0],
    emb_2d[:, 1],
    s=10,
    c=colors,
    cmap=cmap,
    alpha=0.8,
)
plt.title("Vanilla BERT embeddings of uOttawa courses (PCA → 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

# Shared legend
handles = [
    mpatches.Patch(color=cmap(mapping[p]), label=p)
    for p in unique_prefixes
]

plt.legend(
    handles=handles,
    title="Course prefix",
    bbox_to_anchor=(1.05, 1.0),
    loc="upper left",
    borderaxespad=0.0,
)
plt.tight_layout()

plt.show()
