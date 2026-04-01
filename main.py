import os
import io
import json
import logging
import asyncio
import shutil
import re
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image

# --- CONFIGURATION ---
load_dotenv()
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GENAI_API_KEY:
    raise ValueError("Error: GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=GENAI_API_KEY)

app = FastAPI(title="Medical VLM Service")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

SUPPORTED_MIME_TYPES = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
    ".tif":  "image/tiff",
    ".tiff": "image/tiff",
    ".pdf":  "application/pdf",
}

CONVERT_TO_PNG = {"image/tiff"}


def extract_json_payload(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return text

    # Remove fenced markdown blocks and any leading labels.
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```', '', text)

    # Preserve only the first top-level JSON object or array.
    first_index = min([i for i in (text.find('{'), text.find('[')) if i != -1], default=-1)
    if first_index > 0:
        text = text[first_index:]

    if not text:
        return text

    opening = '{' if text.startswith('{') else ('[' if text.startswith('[') else None)
    if opening:
        closing = '}' if opening == '{' else ']'
        stack = []
        for idx, ch in enumerate(text):
            if ch == opening:
                stack.append(ch)
            elif ch == closing and stack:
                stack.pop()
                if not stack:
                    return text[:idx + 1].strip()

    return text.strip()


# --- MERGED DATA MODELS ---

class Patient(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = Field(None, description="male | female | other | unknown")
    birthDate: Optional[str] = Field(None, description="YYYY-MM-DD")
    Age: Optional[int] = None

class Organization(BaseModel):
    name: Optional[str] = None

class DocumentInfo(BaseModel):
    doctor_name: Optional[str] = None
    document_date: Optional[str] = Field(None, description="YYYY-MM-DD")

class Classification(BaseModel):
    type: str = Field(..., description="Prescription, Lab Report, etc.")
    confidence: str = Field(..., description="High, Medium, Low")

class Observation(BaseModel):
    name: str = Field(..., description="Name of test or vitals")
    value: Optional[str] = Field(None, description="Result value. Use string to accommodate '<0.5' or '120/80'")
    unit: Optional[str] = None
    dateTime: Optional[str] = Field(None, description="ISO datetime if available")
    confidence_score: str = Field(..., description="e.g., '98.5%'")

class Condition(BaseModel):
    name: str = Field(..., description="Diagnosis or symptom")
    clinicalStatus: Optional[str] = Field(None, description="active | inactive | resolved")
    confidence_score: str = Field(..., description="e.g., '98.5%'")

class Medication(BaseModel):
    name: str = Field(..., description="Name of medicine")
    dosage_or_frequency: Optional[str] = None
    confidence_score: str = Field(..., description="e.g., '98.5%'")

class MedicalDocumentResponse(BaseModel):
    document_info: Optional[DocumentInfo] = None
    patient: Optional[Patient] = None
    organization: Optional[Organization] = None
    classification: Optional[Classification] = None
    observations: Optional[List[Observation]] = []
    conditions: Optional[List[Condition]] = []
    medications: Optional[List[Medication]] = []
    summary: Optional[str] = None
    # raw_ocr_output has been permanently removed


# --- AI EXTRACTION (Vision OCR → Structured JSON) ---
async def analyze_document_with_vlm(file_bytes: bytes, mime_type: str) -> dict:
    try:
        logging.info(f"Sending file to Google's VLM (MIME: {mime_type})...")

        if mime_type in CONVERT_TO_PNG:
            img = Image.open(io.BytesIO(file_bytes))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            file_bytes = buf.getvalue()
            mime_type = "image/png"

        prompt = """
        You are an expert medical document VLM with OCR capabilities. You will be given a medical document (image or PDF).

        Your job is to read the document and extract the core clinical data into strict JSON format.

        CRITICAL EXTRACTION RULES:
        1. Ignore Administrative Bullshit: DO NOT extract sample collection times, registered dates, report generation times, lab technician names, pathologist names, or addresses into the arrays. 
        2. DO extract the Patient demographics (Name, Age, Gender) and Organization/Facility name into their respective objects.
        3. Assign a realistic `confidence_score` to each array item (Observations, Conditions, Medications) using this rubric:
           - 95-100%: Perfectly legible digital text.
           - 85-94%: Clear handwritten text or slightly blurry digital text.
           - 70-84%: Messy handwriting or faint print.
           - <70%: Nearly illegible, requires guessing.

        Return ONLY valid JSON with this exact structure (Omit keys if the data does not exist in the document):
        {
            "document_info": { "doctor_name": "...", "document_date": "YYYY-MM-DD" },
            "classification": { "type": "Prescription/Lab Report/etc.", "confidence": "High/Medium/Low" },
            "patient": { "name": "...", "gender": "male|female|other|unknown", "birthDate": "YYYY-MM-DD", "Age": 21 },
            "organization": { "name": "..." },
            "observations": [
                { "name": "...", "value": "...", "unit": "...", "dateTime": "...", "confidence_score": "XX.X%" }
            ],
            "conditions": [
                { "name": "...", "clinicalStatus": "active|inactive|resolved", "confidence_score": "XX.X%" }
            ],
            "medications": [
                { "name": "...", "dosage_or_frequency": "...", "confidence_score": "XX.X%" }
            ],
            "summary": "1 sentence clinical summary of the document"
        }
        """

        doc_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )

        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model='gemini-2.5-flash',
                contents=[prompt, doc_part],
                config=config
            )
        except Exception as api_err:
            err_str = str(api_err)
            if "503" in err_str or "UNAVAILABLE" in err_str:
                logging.warning("gemini-2.5-flash unavailable, falling back to gemini-2.0-flash...")
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model='gemini-2.0-flash',
                    contents=[prompt, doc_part],
                    config=config
                )
            else:
                raise

        clean_json = extract_json_payload(response.text)

        try:
            parsed_data = json.loads(clean_json)
        except json.JSONDecodeError as parse_err:
            logging.error(f"VLM JSON parse failed. Raw response:\n{response.text}")
            raise ValueError(f"Failed to parse valid JSON from VLM output: {parse_err}")

        logging.info("Google VLM extraction successful.")
        return parsed_data

    except Exception as e:
        logging.error(f"VLM Error: {e}")
        raise


# --- ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    if not os.path.exists("static/index.html"):
        return HTMLResponse("<h1>Medical VLM Service is Running</h1><p>Please place your index.html in the static folder.</p>")
    
    return FileResponse("static/index.html")

# response_model_exclude_none=True ensures that any null values are stripped from the final JSON
@app.post("/analyze", response_model=MedicalDocumentResponse, response_model_exclude_none=True)
async def analyze_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    mime_type = SUPPORTED_MIME_TYPES.get(ext)

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

    file_bytes = await file.read()

    try:
        structured_data = await analyze_document_with_vlm(file_bytes, mime_type)
        return structured_data
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))