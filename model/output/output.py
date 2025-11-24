import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_courses(
    query_emb,
    course_embs,
    course_df,
    top_n
):
    """
    query_emb: (dim,) or (1, dim)
    course_embs: (num_courses, dim)
    course_df: DataFrame with course info
    """
    # Ensure numpy arrays
    query_emb = np.asarray(query_emb)
    course_embs = np.asarray(course_embs)

    # Make query 2D for cosine_similarity
    if query_emb.ndim == 1:
        query_emb = query_emb[None, :]  # (1, dim)

    # Cosine similarity between query and all courses
    sims = cosine_similarity(query_emb, course_embs)[0]  # (num_courses,)

    # Get top-k indices (sorted high → low)
    top_idx = np.argsort(-sims)[:top_n]

    # Slice your DataFrame and attach scores
    results = course_df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]

    # Optional: reset index for neatness
    results = results.reset_index(drop=True)

    return results


# Fake course embeddings: 10 courses, 768-dim
num_courses = 50
dim = 768

rng = np.random.default_rng(0)
fake_course_embs = rng.normal(size=(num_courses, dim))

# Fake query embedding
fake_query_emb = rng.normal(size=(dim,))

# Fake course metadata
fake_course_df = pd.DataFrame({
    "course_code": [f"COURSE{i:03d}" for i in range(num_courses)],
    "course_title": [f"Dummy Course {i}" for i in range(num_courses)]
})

# Run recommender
top_fake = recommend_courses(fake_query_emb, fake_course_embs, fake_course_df, top_n=5)
print(top_fake[["course_code", "course_title", "similarity"]])
