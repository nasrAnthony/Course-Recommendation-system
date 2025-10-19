import os
import csv
import requests
from bs4 import BeautifulSoup

#Statics
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FP = os.path.join(BASE_DIR, 'data', 'data.csv')

#Testing on elg catalog only
target_paths = ['https://catalogue.uottawa.ca/en/courses/elg/', #electrical
                'https://catalogue.uottawa.ca/en/courses/ceg/', #computer
                'https://catalogue.uottawa.ca/en/courses/bmg/', #biomedical
                'https://catalogue.uottawa.ca/en/courses/chg/', #chemical
                'https://catalogue.uottawa.ca/en/courses/seg/', #software
                'https://catalogue.uottawa.ca/en/courses/cvg/', #civil
                'https://catalogue.uottawa.ca/en/courses/mcg/', #mechanical
                ]


#helper
def is_french(course_code: str) -> bool:
    """
    if french -> True
    if not french -> False

    Checking via 2nd course digit (5, 7, 9 -> French)
    """
    french_codes = ['5', '7', '9']
    return(course_code[1] in french_codes)

def scraper():
    """
    Main scraper unit
    """
    count = 0
    with open(DATA_FP, "w", newline="", encoding="utf-8") as data_file:
        writer = csv.writer(data_file)
        writer.writerow(["Faculty", "Code", "Title", "Description", 
                         "Lecture", "Laboratory", "Tutorial", "Prerequisites"])
        for url in target_paths:
            print(f"Starting scrape on -> {url}")
            try:
                response = requests.get(url)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url}: {e}")
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            course_blocks = soup.find_all("div", class_="courseblock")
            for block in course_blocks:
                title_tag = block.find("h3", class_="courseblocktitle noindent")

                if not title_tag:
                    continue

                title_text = title_tag.get_text(strip=True)
                gen_info = title_text.split(" ", 1)
                course_faculty, course_code = gen_info[0][0:3], gen_info[0][-4:]

                if is_french(course_code=course_code):
                    continue

                course_title = gen_info[-1].split('(3')[0]
                desc_tag = block.find("p", class_="courseblockdesc noindent")

                if not desc_tag:
                    continue

                course_description = desc_tag.get_text(strip=True)
                
                comp_tags = block.find_all("p", class_="courseblockextra noindent")
                lecture, lab, tutorial = "", "", ""  # initialize empty strings
                if comp_tags:
                    components = comp_tags[0].get_text(strip=True)
                    components = components.removeprefix("Course Component:").strip()
    
                    # Split by comma and assign to correct variable
                    for comp in components.split(','):
                        comp = comp.strip().lower()
                        if 'lecture' in comp:
                            lecture = comp
                        elif 'laboratory' in comp or 'lab' in comp:
                            lab = comp
                        elif 'tutorial' in comp:
                            tutorial = comp

                
                prereq_tags = block.find_all("p", class_="courseblockextra highlight noindent")
                if prereq_tags: 
                    prerequisites = prereq_tags[0].get_text(strip=True)
                    prerequisites = prerequisites.removeprefix("Prerequisite:").removeprefix("Prerequisites:").strip()
                
                writer.writerow([course_faculty, course_code, course_title, 
                                 course_description, lecture, lab, tutorial, prerequisites])
                
                count += 1
        print(f"Successfully scraped a total of {count} courses.")

if __name__ == "__main__":
    scraper()