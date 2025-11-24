import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_STUDENTS_FP = os.path.join(BASE_DIR, "data", "students_raw.csv")
CLEAN_STUDENTS_FP = os.path.join(BASE_DIR, "data", "students_clean.csv")


def clean_student_text(text: str) -> str:
    """ Clean a single student text input """
    
    if not text:
        return ""
    
    # the usual formatting
    cleaned = text.strip()
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.lower()
    
    return cleaned

def clean_students_file():
    """ cleaning student dataset using function above """
    cleaned_rows = []
    
    if not os.path.exists(RAW_STUDENTS_FP):
        print("Run the generator first")
        return

    with open(RAW_STUDENTS_FP, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        
        for row in reader:
            raw_text = row.get("StudentText", "")
            cleaned_text = clean_student_text(raw_text)
            cleaned_rows.append({
                "StudentText": cleaned_text,
                "LikedCourses": row.get("LikedCourses", "").strip()
            })
    
    with open(CLEAN_STUDENTS_FP, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = ["StudentText", "LikedCourses"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"Cleaned {len(cleaned_rows)} student entries")

if __name__ == "__main__":
    clean_students_file()
