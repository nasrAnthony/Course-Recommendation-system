import os
import csv
import re

# Files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FP = os.path.join(BASE_DIR, "data", "data.csv")            # previous data
CLEAN_FP = os.path.join(BASE_DIR, "data", "cleaned_courses.csv")  # new clean data

FRENCH_STOPWORDS = {
    "le", "la", "les", "des", "du", "de", "et", "ou", "avec",
    "pour", "dans", "sur", "au", "aux", "une", "un", "ces", "ce",
}

USELESS_TITLE_WORDS = {
    "selected topics",
    "advanced topics",
    "special topics",
    "topics in",
    "directed studies"
    "thesis"
}

MIN_DESC_WORDS = 12

def is_probably_french(text: str) -> bool:
    """
    probably french -> True
    probably not french -> False
    """
    if not text:
        return False

    text_lower = text.lower()

    # accents check
    if re.search(r"[éèàùâêîôûçëïü]", text_lower):
        return True

    # extract words and put into array
    words = re.findall(r"\b\w+\b", text_lower)
    if not words:
        return False
    french_hits = sum(1 for w in words if w in FRENCH_STOPWORDS)

    # check if the number of french words is >3 and there are 20% french words
    if french_hits >= 3 and french_hits / len(words) > 0.2:
        return True

    return False


def is_useless(title: str, desc: str) -> bool:
    """
    useless -> True
    """
    title_lower = (title or "").strip().lower()
    desc_lower = (desc or "").strip().lower()
    
    for word in USELESS_TITLE_WORDS:
        if word in title_lower:
            return True

    if not desc_lower:
        return True

    if len(desc_lower.split()) < MIN_DESC_WORDS:
        return True

    return False

def text_for_bert(row: dict) -> str:
    """
    combining text to make it readable for BERT
    """
    title = (row.get("Title") or "").strip()
    desc = (row.get("Description") or "").strip()
    components = (row.get("Components") or "").strip()
    prereq = (row.get("Prerequisites") or "").strip()

    pieces = []
    if title:
        pieces.append(f"{title}.")
    if desc:
        pieces.append(desc)
    if components:
        pieces.append(components)
    if prereq:
        pieces.append(prereq)

    text = " ".join(p.strip() for p in pieces if p.strip()) #combination of words
    text = " ".join(text.split()) #normalizing spaces
    text = text.lower() #lowercase

    return text


def clean_courses():
    kept = 0
    dropped_french = 0
    dropped_useless = 0
    
    with open(RAW_FP, "r", encoding="utf-8") as infile, \
         open(CLEAN_FP, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)        
        fieldnames = reader.fieldnames + []
        #new column for bert text (if it doesn't exist)
        if "TextForBERT" not in fieldnames:
            fieldnames = fieldnames + ["TextForBERT"]
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()                   

        for row in reader:
            title = row.get("Title", "")
            desc = row.get("Description", "")

            if is_probably_french(title + " " + desc):
                dropped_french += 1
                continue
            
            if is_useless(title, desc):
                dropped_useless += 1
                continue
            
            row["TextForBERT"] = text_for_bert(row)

            writer.writerow(row)
            kept += 1

    print(f"Kept: {kept}")
    print(f"Dropped (French):  {dropped_french}")
    print(f"Dropped (useless): {dropped_useless}")


if __name__ == "__main__":
    clean_courses()
