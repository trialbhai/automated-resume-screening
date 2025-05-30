import spacy
from pdfminer.high_level import extract_text
import fitz  # PyMuPDF
import PIL.Image
import io
import easyocr
import os
import docx
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

from externalHelp.pdf_extract_copy import PDFExtractor

pdf_path = "converted_text.pdf"
#pdf_path = "Tester_Resume_Template.pdf"

try:
    extractor = PDFExtractor(output_base_dir="extracted_pdfs")
    stats = extractor.extract_content(
        pdf_path=pdf_path,
        strategy="hi_res",
        extract_images=True,
        extract_tables=True
    )
    print("Extraction completed. Stats:", stats)
except Exception as e:
    print(f"Error in PDF extraction: {e}")

# ADD EasyOCR-based OCR extraction
def extract_text_easyocr(pdf_path):
    """Extract text using EasyOCR for scanned PDFs."""
    try:
        assert os.path.exists(pdf_path), f"Error: File '{pdf_path}' not found!"
        reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader
        pages = convert_from_path(pdf_path, poppler_path=r"C:\Users\WTI\AppData\Local\Programs\poppler-24.08.0\bin")
        full_text = ""
        for page_num, page in enumerate(pages):
            try:
                img_byte_arr = io.BytesIO()
                page.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                text = reader.readtext(img_byte_arr, detail=0)  # OCR processing
                full_text += f"Page {page_num + 1}:\n{' '.join(text)}\n" + "-" * 40 + "\n"
            except Exception as e:
                print(f"Error processing page {page_num + 1} with EasyOCR: {e}")
        return full_text
    except Exception as e:
        print(f"Error in EasyOCR text extraction: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

nlp = spacy.load("en_core_web_sm")

def extract_resume_details(text):
    """Extract skills, education, and experience using spaCy NLP."""
    try:
        doc = nlp(text)
        skills, education, experience = [], [], []
        for ent in doc.ents:
            if ent.label_ in ["ORG", "EDUCATION"]:
                education.append(ent.text)
            elif ent.label_ in ["WORK_OF_ART", "JOB_TITLE", "PERSON"]:
                experience.append(ent.text)
            elif ent.label_ in ["SKILL", "ABILITY", "TECH"]:
                skills.append(ent.text)
        return {"skills": skills, "education": education, "experience": experience}
    except Exception as e:
        print(f"Error processing resume details with NLP: {e}")
        return {}

def extract_text_pymupdf(file_path):
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])  # Ensures proper text extraction
        return text
    except Exception as e:
        print(f"Error extracting text using PyMuPDF: {e}")
        return ""

def extract_text_pdfplumber(file_path):
    """Extract text from a PDF using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                try:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"
                except Exception as e:
                    print(f"Error extracting text from page {page.page_number} using pdfplumber: {e}")
        return text
    except Exception as e:
        print(f"Error extracting text using pdfplumber: {e}")
        return ""

# ADD new OCR function call
try:
    pdf_text = extract_text_easyocr(pdf_path)
    print("Extracted Text from EasyOCR:\n", pdf_text)
except Exception as e:
    print(f"Error calling EasyOCR function: {e}")

# **Use either PyMuPDF or pdfplumber for text extraction**
try:
    pdf_text = extract_text_pymupdf(pdf_path)  # Using PyMuPDF
    # pdf_text = extract_text_pdfplumber(pdf_path)  # Alternative: Using pdfplumber
    print("Extracted Text:\n", pdf_text)
except Exception as e:
    print(f"Error extracting PDF text: {e}")

# Process extracted text with NLP
try:
    resume_details_pdf = extract_resume_details(pdf_text)
    print("Resume Details from PDF:")
    print(resume_details_pdf)
except Exception as e:
    print(f"Error processing extracted text with NLP: {e}")
