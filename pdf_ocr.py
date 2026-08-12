import cv2
import numpy as np
import pytesseract

from PIL import Image
from pdf2image import convert_from_path


class PDFOCR:
    """
    OCR utility for extracting Marathi and English text from scanned PDFs.
    """

    def __init__(
        self,
        tesseract_path: str,
        poppler_path: str,
        languages: str = "eng+mar",
        dpi: int = 300,
    ):
        """
        Parameters
        ----------
        tesseract_path : str
            Path to tesseract.exe

        poppler_path : str
            Path to Poppler bin folder

        languages : str
            OCR languages (default: eng+mar)

        dpi : int
            DPI used while converting PDF pages to images.
        """

        self.languages = languages
        self.dpi = dpi
        self.poppler_path = poppler_path

        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    # -----------------------------------------------------
    # Image Preprocessing
    # -----------------------------------------------------

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before OCR.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return thresh

    # -----------------------------------------------------
    # OCR on Single Image
    # -----------------------------------------------------

    def image_to_text(self, image: np.ndarray) -> str:
        """
        Extract text from an OpenCV image.
        """

        processed = self.preprocess_image(image)

        pil_image = Image.fromarray(processed)

        text = pytesseract.image_to_string(
            pil_image,
            lang=self.languages,
            config="--oem 3 --psm 6",
        )

        return text

    # -----------------------------------------------------
    # OCR on PDF
    # -----------------------------------------------------

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from all pages of a PDF.
        """

        pages = convert_from_path(
            pdf_path,
            dpi=self.dpi,
            poppler_path=self.poppler_path,
        )

        all_text = []

        total_pages = len(pages)

        for page_number, page in enumerate(pages, start=1):

            print(f"Processing Page {page_number}/{total_pages}")

            image = np.array(page.convert("RGB"))

            text = self.image_to_text(image)

            all_text.append(
                f"\n\n========== PAGE {page_number} ==========\n\n"
            )

            all_text.append(text)

        return "".join(all_text)

    # -----------------------------------------------------
    # Save OCR Result
    # -----------------------------------------------------

    def extract_to_file(self, pdf_path: str, output_file: str):
        """
        Extract text from PDF and save to a text file.
        """

        text = self.extract_text(pdf_path)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved OCR output to: {output_file}")

        return output_file