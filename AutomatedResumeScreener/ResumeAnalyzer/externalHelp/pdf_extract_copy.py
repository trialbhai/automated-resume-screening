import os
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, IO
from io import BytesIO, TextIOWrapper, BufferedReader
from tempfile import SpooledTemporaryFile

# Alternative imports - replace unstructured with these
import fitz  # PyMuPDF for PDF processing
import pandas as pd
from PIL import Image as PILImage
import camelot  # For table extraction
import pdfplumber  # Alternative PDF processing


def convert_to_bytes(
    file: Optional[Union[bytes, SpooledTemporaryFile, IO[bytes]]] = None,
) -> bytes:
    """Convert various file-like objects to bytes."""
    if isinstance(file, bytes):
        f_bytes = file
    elif isinstance(file, SpooledTemporaryFile):
        file.seek(0)
        f_bytes = file.read()
        file.seek(0)
    elif isinstance(file, BytesIO):
        f_bytes = file.getvalue()
    elif isinstance(file, (TextIOWrapper, BufferedReader)):
        with open(file.name, "rb") as f:
            f_bytes = f.read()
    else:
        raise ValueError("Invalid file-like object type")
    return f_bytes


class PDFElement:
    """Base class for PDF elements to replace unstructured elements."""
    def __init__(self, text: str, element_type: str, page_number: int = 0, bbox: tuple = None):
        self.text = text
        self.element_type = element_type
        self.page_number = page_number
        self.bbox = bbox  # Bounding box coordinates


class TextElement(PDFElement):
    """Text element replacement."""
    def __init__(self, text: str, page_number: int = 0, bbox: tuple = None):
        super().__init__(text, "Text", page_number, bbox)


class TitleElement(PDFElement):
    """Title element replacement."""
    def __init__(self, text: str, page_number: int = 0, bbox: tuple = None):
        super().__init__(text, "Title", page_number, bbox)


class TableElement(PDFElement):
    """Table element replacement."""
    def __init__(self, text: str, page_number: int = 0, bbox: tuple = None, html_content: str = None):
        super().__init__(text, "Table", page_number, bbox)
        self.html_content = html_content


class ImageElement(PDFElement):
    """Image element replacement."""
    def __init__(self, page_number: int = 0, bbox: tuple = None, image_path: str = None):
        super().__init__("", "Image", page_number, bbox)
        self.image_path = image_path


class PDFExtractor:
    """A class to extract content from PDFs using alternative libraries."""
    
    def __init__(self, output_base_dir: str = "extracted_content"):
        """
        Initialize the PDF extractor.
        
        Args:
            output_base_dir (str): Base directory for extracted content
        """
        self.output_base_dir = output_base_dir

    def _create_output_dirs(self, pdf_name: str) -> tuple[Path, Path, Path, Path]:
        """Create output directories for the extracted content."""
        # Create main directory named after the PDF
        base_dir = Path(self.output_base_dir) / pdf_name
        
        # Create subdirectories for different content types
        text_dir = base_dir / "text"
        images_dir = base_dir / "images"
        tables_dir = base_dir / "tables"
        metadata_dir = base_dir / "metadata"
        
        # Create all directories
        for dir_path in [base_dir, text_dir, images_dir, tables_dir, metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return base_dir, text_dir, images_dir, tables_dir

    def _extract_text_with_pymupdf(self, pdf_path: str) -> List[PDFElement]:
        """Extract text using PyMuPDF."""
        elements = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                # Determine if it's a title based on font size
                                font_size = span.get("size", 12)
                                bbox = span.get("bbox", (0, 0, 0, 0))
                                
                                if font_size > 14:  # Assume larger font is title
                                    element = TitleElement(text, page_num, bbox)
                                else:
                                    element = TextElement(text, page_num, bbox)
                                elements.append(element)
        
        doc.close()
        return elements

    def _extract_images_with_pymupdf(self, pdf_path: str, images_dir: Path) -> List[ImageElement]:
        """Extract images using PyMuPDF."""
        elements = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_name = f"page_{page_num}_img_{img_index}.png"
                        img_path = images_dir / img_name
                        pix.save(str(img_path))
                        
                        element = ImageElement(
                            page_number=page_num,
                            bbox=page.get_image_bbox(img),
                            image_path=str(img_path)
                        )
                        elements.append(element)
                    
                    pix = None
                except Exception as e:
                    print(f"Warning: Could not extract image {img_index} from page {page_num}: {e}")
        
        doc.close()
        return elements

    def _extract_tables_with_camelot(self, pdf_path: str) -> List[TableElement]:
        """Extract tables using Camelot."""
        elements = []
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(tables):
                # Convert table to text
                df = table.df
                text_content = df.to_string(index=False)
                
                # Convert to HTML
                html_content = df.to_html(index=False)
                
                element = TableElement(
                    text=text_content,
                    page_number=table.page - 1,  # Camelot uses 1-based page numbers
                    html_content=html_content
                )
                elements.append(element)
                
        except Exception as e:
            print(f"Warning: Could not extract tables with Camelot: {e}")
            # Fallback to pdfplumber
            try:
                elements.extend(self._extract_tables_with_pdfplumber(pdf_path))
            except Exception as e2:
                print(f"Warning: Could not extract tables with pdfplumber either: {e2}")
        
        return elements

    def _extract_tables_with_pdfplumber(self, pdf_path: str) -> List[TableElement]:
        """Extract tables using pdfplumber as fallback."""
        elements = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    
                    for table_index, table in enumerate(tables):
                        if table:
                            # Convert table to DataFrame then to text
                            df = pd.DataFrame(table[1:], columns=table[0])
                            text_content = df.to_string(index=False)
                            html_content = df.to_html(index=False)
                            
                            element = TableElement(
                                text=text_content,
                                page_number=page_num,
                                html_content=html_content
                            )
                            elements.append(element)
        except Exception as e:
            print(f"Warning: pdfplumber table extraction failed: {e}")
        
        return elements

    def _process_element(self, element: PDFElement, index: int, text_dir: Path, tables_dir: Path, pdf_path: str) -> Dict[str, Any]:
        """Process a single element and extract its metadata."""
        metadata = {
            "element_index": index,
            "element_type": element.element_type,
            "filename": os.path.basename(pdf_path),
            "extraction_date": datetime.now().isoformat(),
            "page_number": element.page_number,
            "bbox": element.bbox
        }

        try:
            # Process element based on its type
            if isinstance(element, (TextElement, TitleElement)):
                output_path = text_dir / f"{element.element_type.lower()}_{index}.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(element.text)
                metadata["content_path"] = str(output_path)
                
            elif isinstance(element, TableElement):
                # Save table content
                txt_path = tables_dir / f"table_{index}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(element.text)
                metadata["content_path"] = str(txt_path)
                
                # Save HTML structure if available
                if element.html_content:
                    html_path = tables_dir / f"table_{index}.html"
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(element.html_content)
                    metadata["html_path"] = str(html_path)
                    
            elif isinstance(element, ImageElement):
                if element.image_path:
                    metadata["image_path"] = element.image_path
                    
        except Exception as e:
            print(f"Warning: Error processing element {index}: {str(e)}")
            
        return metadata

    def extract_content(self, pdf_path: str, strategy: str = "hi_res", extract_images: bool = True, extract_tables: bool = True) -> dict:
        """Extract content from a PDF file using alternative libraries.
        
        Args:
            pdf_path (str): Path to the PDF file
            strategy (str): Extraction strategy - kept for compatibility (not used in alternative implementation)
            extract_images (bool): Whether to extract images
            extract_tables (bool): Whether to extract tables
        """
        # Get PDF name without extension
        pdf_name = Path(pdf_path).stem
        
        # Create output directories
        base_dir, text_dir, images_dir, tables_dir = self._create_output_dirs(pdf_name)
        
        try:
            elements = []
            
            # Extract text and titles
            text_elements = self._extract_text_with_pymupdf(pdf_path)
            elements.extend(text_elements)
            
            # Extract images if requested
            if extract_images:
                image_elements = self._extract_images_with_pymupdf(pdf_path, images_dir)
                elements.extend(image_elements)
            
            # Extract tables if requested
            if extract_tables:
                table_elements = self._extract_tables_with_camelot(pdf_path)
                elements.extend(table_elements)
            
            # Initialize statistics
            stats = {
                "text_blocks": sum(1 for e in elements if isinstance(e, TextElement)),
                "titles": sum(1 for e in elements if isinstance(e, TitleElement)),
                "images": sum(1 for e in elements if isinstance(e, ImageElement)),
                "tables": sum(1 for e in elements if isinstance(e, TableElement))
            }
            
            all_metadata = []
            
            # Process each element
            for idx, element in enumerate(elements):
                metadata = self._process_element(
                    element, idx, text_dir, tables_dir, pdf_path
                )
                all_metadata.append(metadata)
            
            # Save document metadata
            document_metadata = {
                "filename": os.path.basename(pdf_path),
                "extraction_date": datetime.now().isoformat(),
                "statistics": stats,
                "elements_metadata": all_metadata
            }
            
            with open(base_dir / "document_metadata.json", "w", encoding="utf-8") as f:
                json.dump(document_metadata, f, indent=2, ensure_ascii=False)
            
            return stats
            
        except Exception as e:
            print(f"Error extracting content: {str(e)}")
            return {}


def main():
    """Example usage"""
    try:
        extractor = PDFExtractor("extracted_pdfs")
        
        # Extract content from a PDF
        pdf_path = "your_pdf_here"  # Replace with your PDF path
        results = extractor.extract_content(
            pdf_path,
            extract_images=True,
            extract_tables=True
        )
        
        print("\nExtraction completed successfully! 🎉")
        print("\nStatistics:")
        for key, value in results.items():
            if isinstance(value, (int, str, float)):  # Only print simple values
                print(f"- {key}: {value}")
        
        print("\nExtracted content is saved in:", Path("extracted_pdfs") / Path(pdf_path).stem)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()