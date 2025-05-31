import warnings
warnings.filterwarnings("ignore")
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import spacy
import fitz
import pytesseract
import docx
import re

# Initialize
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Install: python -m spacy download en_core_web_sm")
    nlp = None

def extract_text_pymupdf(file_path):
    """Extract text and images with OCR."""
    try:
        doc = fitz.open(file_path)
        text = ""
        images_processed = set()
        
        for page in doc:
            # Extract text
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
            
            # Extract and OCR images
            for img in page.get_images(full=True):
                xref = img[0]
                if xref not in images_processed:
                    images_processed.add(xref)
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n > 4:  # Convert CMYK to RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    img_file = f"temp_img_{len(images_processed)}.png"
                    pix.save(img_file)
                    pix = None
                    
                    # OCR the image
                    try:
                        ocr_text = pytesseract.image_to_string(img_file, lang='eng')
                        if ocr_text.strip():
                            text += ocr_text + "\n"
                        os.remove(img_file)  # Clean up
                    except Exception:
                        pass
        
        doc.close()
        return text
    except Exception as e:
        print(f"PDF error: {e}")
        return ""

def extract_resume_details(text):
    """Extract key resume information."""
    if not nlp or not text:
        return {}
    
    doc = nlp(text)
    
    # Extract entities
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    
    # Extract contacts
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phones = re.findall(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', text)
    
    # Extract skills
    skills = ['python', 'java', 'javascript', 'html', 'css', 'sql', 'react', 'node',
              'machine learning', 'data analysis', 'project management', 'leadership']
    found_skills = [skill for skill in skills if skill in text.lower()]
    
    return {
        "name": persons[0] if persons else "Not found",
        "email": emails[0] if emails else "Not found", 
        "phone": phones[0] if phones else "Not found",
        "organizations": list(set(orgs))[:3],  # Top 3
        "locations": list(set(locations))[:2],  # Top 2
        "skills": found_skills
    }

def process_resume(file_path):
    """Process resume file."""
    ext = os.path.splitext(file_path)[1].lower()
    texts = ""
    
    if ext == '.pdf':
        texts = extract_text_pymupdf(file_path)
        print(texts)
    elif ext in ['.docx', '.doc']:
        try:
            doc = docx.Document(file_path)
            texts = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            print(f"DOCX error: {e}")
    else:
        print(f"Unsupported: {ext}")
        return None
    
    if len(texts) > 10:
        print(f"✅ Extracted {len(texts)} characters")
        details = extract_resume_details(texts)
        
        # Save results
        with open("resume_report.txt", "w", encoding="utf-8") as f:
            f.write(texts)
        
        return texts, details
    else:
        print("❌ No text extracted")
        return None

# Main execution
if __name__ == "__main__":
    files = ["Tester_Resume_Template.pdf", "Teacher.pdf", "converted_text.pdf"]
    
    for file in files:
        if os.path.exists(file):
            print(f"\n📄 Processing: {file}")
            result = process_resume(file)
            if result:
                text, details = result
                print("📋 Details:")
                for key, value in details.items():
                    print(f"  {key}: {value}")
            print("-" * 40)
        else:
            print(f"❌ Not found: {file}")


def parse_resume_sections(text):
    pass
    # Identify sections like:
    # - Personal Information (name, email, phone)
    # - Experience/Work History
    # - Education
    # - Skills
    # - Certifications
    # - Projects