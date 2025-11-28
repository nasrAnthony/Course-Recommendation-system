import os
import csv
import requests
from bs4 import BeautifulSoup

# Setting file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FP = os.path.join(BASE_DIR, 'data', 'data.csv')

# Testing on engineering courses only
target_paths = ['https://catalogue.uottawa.ca/en/courses/elg/', #electrical
                'https://catalogue.uottawa.ca/en/courses/ceg/', #computer
                'https://catalogue.uottawa.ca/en/courses/bmg/', #biomedical
                'https://catalogue.uottawa.ca/en/courses/chg/', #chemical
                'https://catalogue.uottawa.ca/en/courses/seg/', #software
                'https://catalogue.uottawa.ca/en/courses/cvg/', #civil
                'https://catalogue.uottawa.ca/en/courses/mcg/', #mechanical
                ]


def scraper():
    """
    Main scraper unit
    Gets course data with minimal cleanup. See clean_courses.py for cleaning script
    """
    count = 0
    with open(DATA_FP, "w", newline="", encoding="utf-8") as data_file:
        writer = csv.writer(data_file)
        # setting columns
        writer.writerow(["Faculty", "Code", "Title", "Description", 
                         "Components", "Prerequisites"])
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
                
                # course title and code
                title_tag = block.find("h3", class_="courseblocktitle noindent")
                if not title_tag:
                    continue
                title_text = title_tag.get_text(strip=True)
                gen_info = title_text.split(" ", 1)
                course_faculty, course_code = gen_info[0][0:3], gen_info[0][-4:]
                course_title = gen_info[-1].split('(3')[0]
                
                # course description
                desc_tag = block.find("p", class_="courseblockdesc noindent")
                if not desc_tag:
                    continue
                course_description = desc_tag.get_text(strip=True)
                
                # course components
                comp_tags = block.find_all("p", class_="courseblockextra noindent")
                components = "" 
                if comp_tags:
                    components = comp_tags[0].get_text(strip=True)
                    components = components.removeprefix("Course Component:").strip() # slight cleanup
                    components = " ".join(components.split())
    
                # prerequisites
                prereq_tags = block.find_all("p", class_="courseblockextra highlight noindent")
                prerequisites = ""
                if prereq_tags: 
                    prerequisites = prereq_tags[0].get_text(strip=True)
                
                # writing all the data
                writer.writerow([course_faculty, course_code, course_title, 
                                 course_description, components, prerequisites])
                
                count += 1
        print(f"Successfully scraped a total of {count} courses.")

if __name__ == "__main__":
    scraper()