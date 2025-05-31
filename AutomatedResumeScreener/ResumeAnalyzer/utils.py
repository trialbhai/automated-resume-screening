import spacy
import pdfplumber
import docx

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


nlp = spacy.load("en_core_web_sm")

def extract_resume_details(text):
    doc = nlp(text)
    skills = []
    education = []
    experience = []
    
    for ent in doc.ents:
        if ent.label_ in ["ORG", "EDUCATION"]:
            education.append(ent.text)
        elif ent.label_ in ["WORK_OF_ART", "JOB_TITLE", "PERSON"]:
            experience.append(ent.text)
        elif ent.label_ in ["SKILL", "ABILITY", "TECH"]:
            skills.append(ent.text)
    
    return {"skills": skills, "education": education, "experience": experience}
