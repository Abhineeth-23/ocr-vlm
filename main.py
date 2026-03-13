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
    fhir_bundle: Optional[dict] = Field(None, description="FHIR compliant bundle")

# --- FHIR CONVERTER ---
def generate_fhir_bundle(data: dict) -> dict:
    entries = []
    meta = data.get('metadata', {})

    patient_id = "patient-1"
    if meta.get('patient_name') and meta['patient_name'] != "Not Detected":
        entries.append({
            "fullUrl": f"urn:uuid:{patient_id}",
            "resource": {"resourceType": "Patient", "id": patient_id, "name": [{"text": meta['patient_name']}]}
        })

    practitioner_id = "practitioner-1"
    if meta.get('doctor_name') and meta['doctor_name'] != "Not Detected":
        entries.append({
            "fullUrl": f"urn:uuid:{practitioner_id}",
            "resource": {"resourceType": "Practitioner", "id": practitioner_id, "name": [{"text": meta['doctor_name']}]}
        })

    doc_type = data.get('classification', {}).get('type', '').lower()

    for idx, item in enumerate(data.get('extracted_data', [])):
        item_name = item.get('item_name')
        if not item_name: continue

        value = item.get('value') or ""
        unit = item.get('unit_or_frequency') or ""
        combined_val = f"{value} {unit}".strip()
        category = item.get('category', '').lower() if item.get('category') else ""

        if "prescription" in doc_type or "medication" in category:
            resource = {
                "resourceType": "MedicationRequest",
                "id": f"medreq-{idx}",
                "status": "unknown",
                "intent": "order",
                "medicationCodeableConcept": {"text": item_name},
            }
            if combined_val: resource["dosageInstruction"] = [{"text": combined_val}]
            if meta.get('patient_name'): resource["subject"] = {"reference": f"urn:uuid:{patient_id}"}
            if meta.get('doctor_name'): resource["requester"] = {"reference": f"urn:uuid:{practitioner_id}"}
        else:
            resource = {
                "resourceType": "Observation",
                "id": f"obs-{idx}",
                "status": "final",
                "code": {"text": item_name},
                "valueString": combined_val
            }
            if meta.get('patient_name'): resource["subject"] = {"reference": f"urn:uuid:{patient_id}"}
            if meta.get('doctor_name'): resource["performer"] = [{"reference": f"urn:uuid:{practitioner_id}"}]

        entries.append({"fullUrl": f"urn:uuid:{resource['id']}", "resource": resource})

    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


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
        parsed_data['fhir_bundle'] = generate_fhir_bundle(parsed_data)

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
        return structured_data
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)