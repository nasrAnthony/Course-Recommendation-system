import random
import os
from typing import List, Tuple
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize

import nltk

for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']:
    nltk.download(pkg, quiet=True)

# Get path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH_AUGMENTED_DATA = os.path.join(BASE_DIR, "data", "students_clean_train.csv")
df_2 = pd.read_csv(CSV_PATH_AUGMENTED_DATA)

# Text Augmentation -----------------------------------------------
# - random deletion
# - synonym replacement
# - random insertion
# - random swap


def get_wordnet_pos(treebank_tag: str):
    """
    Returns tags to words (adjective, verb, noun, adverb)
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def get_synonym(word: str, wn_pos=None):
    """Get a random synonym for a word if it's available"""
    try:
        # get the whole set of synonyms for a specific word
        synsets = wn.synsets(word, pos=wn_pos) if wn_pos else wn.synsets(word)
        if not synsets:
            return None

        # choose a random synonym to retyrn
        synset = random.choice(synsets)
        lemmas = [l.name().replace('_', ' ') for l in synset.lemmas() if l.name().lower() != word.lower()]
        if not lemmas:
            return None
        return random.choice(lemmas)
    except Exception:
        return None


def random_deletion(words: List[str], p: float = 0.1) -> List[str]:
    """Delete a a random word and return the full updated text"""
    if len(words) == 1:
        return words
    kept = [w for w in words if random.random() > p] # 10% chance of removing each word
    if not kept:
        kept = [random.choice(words)]
    return kept


def synonym_replacement(words: List[str], n: int = 1) -> List[str]:
    """Replace up to n words with synonyms and return the full updated text"""
    if len(words) == 0:
        return words

    new_words = words.copy()
    tagged = pos_tag(new_words)
    candidates = list(range(len(new_words)))  # indices

    random.shuffle(candidates)
    num_replaced = 0

    for idx in candidates:
        if num_replaced >= n:
            break

        word = new_words[idx]
        if not word.isalpha():
            continue

        _, tag = tagged[idx]
        wn_pos = get_wordnet_pos(tag)

        # adjectives get more weight
        synonym = get_synonym(word, wn_pos=wn_pos)
        if synonym is None:
            continue

        # replace random candidate index with synonym
        new_words[idx] = synonym
        num_replaced += 1

    return new_words


def random_swap(words: List[str], n: int = 1) -> List[str]:
    """Make n random swaps in the string and return the full updated text"""
    if len(words) < 2:
        return words
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words


def random_insertion(words: List[str], n: int = 1) -> List[str]:
    """Insert n synonyms of random words at random and return the full updated text"""
    new_words = words.copy()
    length = len(new_words)
    if length == 0:
        return new_words

    # choose the index
    for _ in range(n):
        idx = random.randrange(length)
        word = new_words[idx]
        if not word.isalpha():
            continue

        # Try to get synonym (any POS)
        synonym = get_synonym(word)
        if synonym is None:
            continue

        # insert the random synonym in a random place
        insert_pos = random.randrange(len(new_words) + 1)
        new_words.insert(insert_pos, synonym)

    return new_words


def augment_text(text: str, mode: str = "light") -> str:
    """
    Create one augmented view.
    mode = "light" or "heavy"

    light  -> small changes, stays very close to original
    heavy  -> larger changes, but try to keep semantics
    """
    words = word_tokenize(text)

    if len(words) == 0:
        return text

    # augmentation parameters by mode
    if mode == "heavy":
        del_prob = 0.15             # more deletion
        syn_prob = 0.8              # more synonym operations
        syn_n_choices = [1, 2, 3]   # replace up to 3 words
        ins_prob = 0.5              # more insertion
        ins_n_choices = [1, 2]
        swap_prob = 0.4             # higher chance of swap
    else:
        del_prob = 0.05          # lower deletion prob
        syn_prob = 0.4           # less synonym operations
        syn_n_choices = [1]      # replace at most 1 word
        ins_prob = 0.2           # rarely insert
        ins_n_choices = [1]
        swap_prob = 0.2          # low swap chance

    # 1. Random deletion
    if random.random() < 0.8:  # keep same probability of *using* deletion
        words = random_deletion(words, p=del_prob)

    # 2. Synonym replacement
    if random.random() < syn_prob:
        words = synonym_replacement(words, n=random.choice(syn_n_choices))

    # 3. Random insertion
    if random.random() < ins_prob:
        words = random_insertion(words, n=random.choice(ins_n_choices))

    # 4. Random swap
    if random.random() < swap_prob:
        words = random_swap(words, n=1)

    return " ".join(words)


def make_two_views(text: str, course_code:str) -> Tuple[str, str]:
    """
    Return 2 differently augmented views of the same base text.

    view1: light augmentation (close to original)
    view2: heavy augmentation (more aggressively perturbed)
    """
    rows = df_2.loc[df_2["LikedCourses"].str.strip() == course_code, "StudentText"]
    if not rows.empty:
        view1 = rows.iloc[0]
    else:
        view1 = augment_text(text, mode="light")
    
    view2 = augment_text(text, mode="heavy")
    return view1, view2