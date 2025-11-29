import re
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
    ''' 
    takes in the top n recommendation dataframe and liked course string
    returns metrics for the student
    '''
    # format liked courses
    tokens = re.split(r"[;,]", liked_courses_str)
    liked_list = [x.strip() for x in tokens if x.strip()]
    liked_set = set(liked_list)

    matches = []
    first_match_rank = None

    # checking the recieved courses vs liked courses
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        fac_code = f"{row['Faculty']} {row['Code']}"
        
        if fac_code in liked_set:
            matches.append(fac_code)
            
            if first_match_rank is None:
                first_match_rank = rank

    n = len(top_df)
    num_matches = len(matches)
        
    # metrics calculations
    hit = 1 if num_matches > 0 else 0                       # was one of the liked courses recommended
    precision = num_matches / n if n > 0 else 0             # @Yhilal02 revisit this vs recall and see which is better (f1 score? scaling up precision?)
    mrr = 1 / first_match_rank if first_match_rank else 0   # how early was match received

    return {
        "hit": hit,
        "precision": precision,
        "mrr": mrr
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
    
    
def evaluate_many_students(student_embs, course_embs, course_df, student_df, top_n=5):
    ''' gets the metrics for multiple students'''
    results = []

    for i in range(len(student_df)):
        # get the top n for student[i]
        student_emb = student_embs[i]
        liked_str = student_df.loc[i, "LikedCourses"]

        top_df = get_top_n_recommendations(
            student_emb,
            course_embs,
            course_df,
            top_n
        )

        # save their metrics
        m = metrics(top_df, liked_str)
        m["student_index"] = i
        m["num_liked"] = len([x.strip() for x in re.split(r"[;,]", liked_str) if x.strip()])
        results.append(m)

    return pd.DataFrame(results)