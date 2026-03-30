import os
import io
import json
import logging
import asyncio
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import shutil
from dotenv import load_dotenv
from PIL import Image
import requests

# --- CONFIGURATION ---
load_dotenv()
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("Error: GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=GENAI_API_KEY)

app = FastAPI(title="Medical VLM Service")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Supported MIME types — images + PDF
SUPPORTED_MIME_TYPES = {
    # Images (Gemini Vision supported natively)
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
    # Converted server-side before sending to Gemini
    ".tif":  "image/tiff",
    ".tiff": "image/tiff",
    # Documents
    ".pdf":  "application/pdf",
}

# MIME types that Gemini does NOT support — we convert them to PNG first
CONVERT_TO_PNG = {"image/tiff"}

# --- DATA MODELS ---
class Metadata(BaseModel):
    patient_name: Optional[str] = Field(None, description="Name of the patient")
    doctor_name: Optional[str] = Field(None, description="Name of the doctor")
    medical_center: Optional[str] = None
    document_date: Optional[str] = Field(None, description="YYYY-MM-DD")

class ExtractedItem(BaseModel):
    item_name: str = Field(..., description="Name of medicine or test")
    value: Optional[str] = Field(None, description="Dosage or Result value")
    unit_or_frequency: Optional[str] = Field(None, description="e.g. 'mg', 'Twice Daily'")
    category: Optional[str] = Field(None, description="Medication | Lab Test | Vitals")
    confidence_score: str = Field(..., description="High Confidence e.g. '98.5%' or '99.9%'")

class Classification(BaseModel):
    type: str = Field(..., description="Prescription, Lab Report, etc.")
    confidence: str = Field(..., description="High, Medium, Low")

class MedicalDocumentResponse(BaseModel):
    metadata: Metadata
    classification: Classification
    extracted_data: List[ExtractedItem]
    summary: Optional[str] = None
    raw_ocr_output: Optional[str] = None

# --- AI EXTRACTION (Vision OCR → Structured JSON) ---
async def analyze_document_with_vlm(file_path: str, mime_type: str) -> dict:
    try:
        logging.info(f"Sending file to Google's VLM (MIME: {mime_type})...")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Convert unsupported types (e.g. TIFF) to PNG before sending to Gemini
        if mime_type in CONVERT_TO_PNG:
            logging.info(f"Converting {mime_type} → image/png for Gemini compatibility...")
            img = Image.open(io.BytesIO(file_bytes))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            file_bytes = buf.getvalue()
            mime_type = "image/png"

        prompt = """
        You are an expert medical document VLM with OCR capabilities. You will be given a medical document (image or PDF).

        Your job is to:
        1. Extract ALL raw text from the document exactly as it appears — this is the OCR output.
        2. Structure and categorize that data into strict JSON.
        3. Assign a realistic \`confidence_score\` to each extracted item using the following strict rubric (output as percentage string, e.g., "85.5%"):
           - 95-100%: Perfectly legible digital text, no artifacts, standard medical formatting.
           - 85-94%: Clear handwritten text, or digital text with slight compression/blur, easy to read.
           - 70-84%: Messy handwriting, faint print, or moderate noise/shadows over the text.
           - 50-69%: Very messy doctor handwriting, severe blur, crossed-out text, requires guessing context to read.
           - <50%: Nearly illegible, heavy artifacting, extreme guesswork. Do NOT default to 99% if the text is clearly handwritten or slightly blurry. Be honest and penalize for bad handwriting.

        Return ONLY valid JSON with this exact structure:
        {
            "metadata": { "patient_name": "...", "doctor_name": "...", "medical_center": "...", "document_date": "YYYY-MM-DD" },
            "classification": { "type": "Prescription/Lab Report/etc.", "confidence": "High/Medium/Low" },
            "extracted_data": [
                { "item_name": "...", "value": "...", "unit_or_frequency": "...", "category": "Medication|Lab Test|Vitals", "confidence_score": "XX.X%" }
            ],
            "summary": "1 sentence summary of the document",
            "raw_ocr_output": "All raw text extracted from the document exactly as it appears"
        }

        For confidence_score, reflect your actual certainty (e.g. "99.2%" if clear, "84.5%" if ambiguous).
        If a field is not present in the document, use null.
        For PDFs with multiple pages, process all pages and combine the extracted data.
        """

        doc_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model='gemini-2.5-flash',
                contents=[prompt, doc_part]
            )
        except Exception as api_err:
            err_str = str(api_err)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                logging.warning("gemini-2.5-flash is temporarily unavailable, falling back to gemini-2.0-flash...")
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model='gemini-2.0-flash',
                    contents=[prompt, doc_part]
                )
            else:
                raise

        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        parsed_data = json.loads(clean_json)

        logging.info("Google VLM extraction successful.")
        return parsed_data

    except Exception as e:
        logging.error(f"VLM Error: {e}")
        raise


# --- ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/analyze", response_model=MedicalDocumentResponse)
async def analyze_document(file: UploadFile = File(...)):
    # Determine MIME type from extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    mime_type = SUPPORTED_MIME_TYPES.get(ext)

    # Fallback: try content_type from request
    if not mime_type:
        ct = file.content_type or ""
        if ct.startswith("image/") or ct == "application/pdf":
            mime_type = ct

    if not mime_type:
        supported = ", ".join(SUPPORTED_MIME_TYPES.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported formats: {supported}"
        )

    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        structured_data = await analyze_document_with_vlm(temp_filename, mime_type)
        
        # --- SEND DATA TO REGISTRATIONS API ---
        try:
            # Create a copy to avoid modifying the response going back to the frontend
            standard_json_payload = dict(structured_data)

            # Fire it off to the specific port
            response = requests.post(
                "http://172.18.8.57:8081/submitdata", 
                json=standard_json_payload, 
                timeout=5
            )
            logging.info(f"Sent to port 8081! Server responded with status: {response.status_code}")
        except Exception as forward_err:
            logging.error(f"Failed to send JSON to target API: {forward_err}")
        # ---------------------------------------

        return structured_data
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)