# pip install pytesseract pdf2image pillow opencv-python
# if mar.trainneddata not availeble in C:\Program Files\Tesseract-OCR\tessdata\
# go to https://github.com/tesseract-ocr/tessdata/blob/main/mar.traineddata

from pdf_ocr import PDFOCR
import re
import jiwer


FILE = r"C:\Work\GR AI\GR's\Assembly related\GAD Circular 7.9.2019 Handling assembly matters 15 days.pdf"
ocr = PDFOCR(
    tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    poppler_path=r"C:\Program Files\poppler\poppler-26.02.0\Library\bin",
    languages="eng+mar",
    dpi=300,
)

text = ocr.extract_text(FILE)

print(text)

# Or save directly
#ocr.extract_to_file("sample.pdf", "output.txt")N

def evaluate_ocr(ground_truth: str, ocr_output: str) -> dict:
    """
    Evaluates Marathi OCR performance against ground truth text.
    Returns Character Error Rate (CER) and Word Error Rate (WER).
    """
    def clean_marathi_text(text: str) -> str:
        # Remove common punctuation and the Marathi danda '|'
        text = re.sub(r'[।\.,\?!\-_()\"\'\'“”]', '', text)
        # Normalize multiple spaces into a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Preprocess both inputs
    clean_truth = clean_marathi_text(ground_truth)
    clean_ocr = clean_marathi_text(ocr_output)

    # Calculate error rates
    cer = jiwer.cer(clean_truth, clean_ocr)
    wer = jiwer.wer(clean_truth, clean_ocr)

    # Return results as a structured dictionary
    return {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "cer_percentage": round(cer * 100, 2),
        "wer_percentage": round(wer * 100, 2)
    }