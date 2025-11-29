import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def get_top_n_recommendations(student_emb, course_embs, course_df, top_n=5):
    
    # 2D shape for cosine similarity
    student_emb = np.asarray(student_emb)
    if student_emb.ndim == 1:
        student_emb = student_emb[None, :]

    # cosine similarity calculation for all courses
    sims = cosine_similarity(student_emb, course_embs)[0]

    # get top N course indicies
    top_idx = np.argsort(-sims)[:top_n]

    # make a df with the chosen courses data
    top_df = course_df.iloc[top_idx].copy()
    top_df["similarity"] = sims[top_idx]

    return top_df


def print_recommendations(recs_df):
    ''' Takes pd dataframe and prints the top N courses'''

    print(f"\nTop {len(recs_df)} recommended courses:\n")

    for i, row in recs_df.iterrows():
        fac_code = f"{row['Faculty']} {row['Code']}"
        title = row['Title']
        desc = row['Description']
        sim_str = f" [similarity: {row['similarity']:.4f}]"

        print(f"{i+1}. {fac_code}{sim_str}")
        print(f"   Title: {title}")
        print(f"   Desc : {desc}\n")
    
    
def metrics(top_df, liked_courses_str):
    # @Yhilal02 look at this again and clean it up

    liked_list = [x.strip() for x in liked_courses_str.split(";") if x.strip()]
    liked_set = set(liked_list)

    matches = []
    first_match_rank = None

    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        fac_code = f"{row['Faculty']} {row['Code']}"
        
        if fac_code in liked_set:
            matches.append(fac_code)
            
            if first_match_rank is None:
                first_match_rank = rank

    top_n = len(top_df)
    num_matches = len(matches)
    num_liked = len(liked_list)
    
    # metrics calculations
    hit = 1 if num_matches > 0 else 0
    precision = num_matches / top_n if top_n > 0 else 0
    recall = num_matches / num_liked if num_liked > 0 else 0
    mrr = 1 / first_match_rank if first_match_rank is not None else 0

    return {
        "matches": matches,
        "hit": hit,
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
    }
    
    
def cosine_stats_and_plot(emb, label):
    '''
    Calculates cosine similarity for a set of embedded vectors.
    input -> embeddings
    output -> prints mean and std deviation//displays a histogram with the distribution
    '''
    n = min(600, emb.shape[0])
    sub = emb[:n]

    sims = cosine_similarity(sub, sub)
    iu = np.triu_indices_from(sims, k=1)
    vals = sims[iu]

    print(f"\n{label}")
    print("Mean cosine:", vals.mean())
    print("Std  cosine:", vals.std())

    plt.hist(vals, bins=50, alpha=0.8)
    plt.title(f"Cosine similarity distribution – {label}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()