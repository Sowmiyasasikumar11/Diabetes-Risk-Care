import io
import logging
import math
import re

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}
MAX_REPORT_BYTES = 10 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


class ReportProcessingError(ValueError):
    """Raised when a report cannot be safely read or processed."""


def is_allowed_filename(filename):
    safe_name = secure_filename(filename or "")
    return bool(safe_name and "." in safe_name and safe_name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)


def extract_report_text(filename, report_bytes):
    extension = secure_filename(filename).rsplit(".", 1)[1].lower()
    if extension == "pdf":
        return _extract_pdf_text(report_bytes)
    return _extract_image_text(report_bytes)


def extract_medical_values(text):
    values = {}
    patterns = {
        "age": r"(?:patient\s+)?age\s*[:=\-]?\s*(\d{1,3}(?:\.\d+)?)\s*(?:years?|yrs?)?\b",
        "gender": r"(?:gender|sex)\s*[:=\-]?\s*(female|male|other)\b",
        "bmi": r"(?:bmi|body\s+mass\s+index)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b",
        "blood_glucose_level": r"(?:blood\s+glucose|blood\s+sugar|glucose\s+level|fasting\s+glucose|random\s+glucose)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b",
        "hba1c_level": r"(?:hba1c(?:\s+level)?|a1c|glycated\s+haemoglobin|glycated\s+hemoglobin)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*%?\b",
        "pregnancies": r"(?:pregnancies|gravida)\s*[:=\-]?\s*(\d{1,2})\b",
        "hypertension": r"hypertension\s*[:=\-]?\s*(yes|no|present|absent|0|1)\b",
        "heart_disease": r"(?:heart\s+disease|cardiac\s+disease)\s*[:=\-]?\s*(yes|no|present|absent|0|1)\b",
        "smoking_history": r"(?:smoking\s+history|smoker)\s*[:=\-]?\s*(current|former|never|non[-\s]?smoker)\b",
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            raw_value = match.group(1).strip()
            values[field] = raw_value

    for field in ("age", "bmi", "blood_glucose_level", "hba1c_level"):
        if field in values:
            try:
                number = float(values[field])
            except ValueError:
                values.pop(field, None)
                continue
            if not math.isfinite(number) or number <= 0:
                values.pop(field, None)
            else:
                values[field] = int(number) if field == "age" and number.is_integer() else number

    if "age" in values and not 1 <= values["age"] <= 120:
        values.pop("age")
    if "pregnancies" in values:
        pregnancies = int(values["pregnancies"])
        if pregnancies < 0:
            values.pop("pregnancies")
        else:
            values["pregnancies"] = pregnancies
    for field in ("hypertension", "heart_disease"):
        if field in values:
            values[field] = {"yes": "1", "present": "1", "no": "0", "absent": "0"}.get(
                values[field].lower(), values[field]
            )
    if "smoking_history" in values:
        values["smoking_history"] = values["smoking_history"].lower()
        if values["smoking_history"] == "non-smoker":
            values["smoking_history"] = "never"
    if "gender" in values:
        values["gender"] = values["gender"].capitalize()
    return values


def _extract_pdf_text(report_bytes):
    document = None
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf

        document = pymupdf.open(stream=report_bytes, filetype="pdf")
        page_text = [page.get_text("text") for page in document]
        text = "\n".join(page_text).strip()
        if text:
            return text
        raise ReportProcessingError("The PDF does not contain readable text.")
    except ReportProcessingError:
        raise
    except ImportError as error:
        LOGGER.exception("PDF extraction requires PyMuPDF or fitz in the active Python environment.")
        raise ReportProcessingError(
            "PDF extraction is unavailable. Install the project requirements and restart the application."
        ) from error
    except Exception as error:
        LOGGER.exception("PDF extraction failed for an uploaded report.")
        raise ReportProcessingError("The PDF could not be read.") from error
    finally:
        if document is not None:
            document.close()


def _extract_image_text(report_bytes):
    try:
        image = Image.open(io.BytesIO(report_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ReportProcessingError("The image could not be read.") from error
    try:
        import pytesseract

        text = pytesseract.image_to_string(image).strip()
    except Exception as error:
        raise ReportProcessingError("The image could not be read by the OCR service.") from error
    if not text:
        raise ReportProcessingError("The image does not contain readable text.")
    return text