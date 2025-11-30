import os
import csv
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COURSES_FP = os.path.join(BASE_DIR, "data", "cleaned_courses.csv")
STUDENTS_TRAIN_FP = os.path.join(BASE_DIR, "data", "students_clean_train.csv")
STUDENTS_TEST_FP = os.path.join(BASE_DIR, "data", "students_clean_test.csv")
N_STUDENTS = 600    # num of student data
TRAIN_SPLIT = 0.8   # 80% for training, 20% for testing


def clean_student_text(text: str) -> str:
    """ Clean a single student text input """
    
    if not text:
        return ""
    
    # the usual formatting
    cleaned = text.strip()
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.lower()
    
    return cleaned

def load_courses():
    """ Load courses from cleaned_courses.csv """
    
    courses = []
    with open(COURSES_FP, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            courses.append(row)
            
    print(f"Loaded {len(courses)} courses")
    return courses

def make_student_text(course_titles):
    """ takes all course titles and randomly makes a positive/negative sentence """   
    
    n = len(course_titles)
    k = 1 if random.random() < 0.5 else 2
    used_indices = random.sample(range(n), k=k) # 1 or 2 course titles to use

    # get the titles for those indices
    titles = [course_titles[i].get("Title", "").strip() for i in used_indices]
    title_list = ", ".join([t for t in titles])  # get 2 random course titles

    # positive sentences
    positive_short = [
        f"I like {titles[0]}.",
        f"I enjoy {titles[0]}."]
    positive_long = [
        f"I am interested in courses like {title_list}.",
        f"I enjoy studying areas such as {title_list}.",]

    # 50% short 50% long sentence
    sentence = []
    if random.random() < 0.5: sentence.append(random.choice(positive_short))
    else: sentence.append(random.choice(positive_long))

    # some negative sentences
    negative = [
        "I am not interested in business courses.",
        "I would rather avoid purely theoretical classes.",
        "I am less interested in software-heavy courses.",
        "I prefer not to take unrelated subjects.",
        "I am not looking for mechanical engineering topics.",]
    
    # 25% chance to get a negative in sentence
    if random.random() < 0.25: sentence.append(random.choice(negative))

    # Early formatting stuff
    text = " ".join(sentence)
    return text, used_indices

def generate_students(courses, n_students):
    """ actually makes the student list with sentence/liked course code(s) """
    
    rows = []
    num_courses = len(courses)

    for i in range(n_students):

        num_liked = random.randint(3, 7)
        chosen = random.sample(courses, k=min(num_liked, num_courses))
        student_text, used_indices = make_student_text(chosen)

        # only get the code(s) for used in student text
        liked_codes = []
        for idx in used_indices:
            c = chosen[idx]
            code_str = f"{c.get('Faculty','').strip()} {c.get('Code','').strip()}"
            liked_codes.append(code_str)
        
        # Clean the student text before adding to rows
        cleaned_text = clean_student_text(student_text)
        
        row = {
            "StudentText": cleaned_text,
            "LikedCourses": ";".join(liked_codes),
        }

        rows.append(row)

    return rows

def main():
    courses = load_courses()
    if not courses:
        print("Run the scraper/cleaner first!!!!!!")
        return
    students = generate_students(courses, N_STUDENTS)

    # Shuffle students before splitting to ensure random distribution
    random.shuffle(students)
    
    # Split into train and test sets
    split_idx = int(len(students) * TRAIN_SPLIT)
    train_students = students[:split_idx]
    test_students = students[split_idx:]
    
    # Write training set
    with open(STUDENTS_TRAIN_FP, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["StudentText", "LikedCourses"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_students)
    
    # Write test set
    with open(STUDENTS_TEST_FP, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["StudentText", "LikedCourses"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_students)
        
    print(f"Generated and cleaned {len(students)} sample student entries")
    print(f"  - Training set: {len(train_students)} entries -> {STUDENTS_TRAIN_FP}")
    print(f"  - Test set: {len(test_students)} entries -> {STUDENTS_TEST_FP}")
    
if __name__ == "__main__":
    main()