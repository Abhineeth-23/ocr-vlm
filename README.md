# OCR-VLM: Medical Document Analyzer

OCR-VLM is a modern web application that leverages Google's advanced **Gemini Vision AI** to perform OCR, structure medical data, and validate clinical entities from medical documents like prescriptions, lab reports, and vitals records. It features a premium, responsive glassmorphism frontend and a lightweight FastAPI/Python backend for secure API interactions and logic handling.

## 🚀 Features

*   **Advanced Vision AI**: Extracts text directly from images (JPG, PNG, WEBP) and PDFs using `gemini-2.5-flash` with dynamic fallback to `gemini-2.0-flash` during high demand.
*   **Structured Medical Data**: Automatically categorizes extracted items into clinical entities, values, units, and confidence scores based on a strict JSON schema.
*   **FHIR Export**: Generates compliant FHIR Release 4 JSON bundles for EHR integrations (including Patient, Practitioner, Observation, and MedicationRequest resources).
*   **Premium Glassmorphism UI**: A completely custom, modern interface with smooth animated gradients, dark/light mode toggling, and interactive data tables.
*   **Multi-Input Ready**: 
    *   Drag and drop document upload
    *   Direct `[ 📸 Camera Capture ]` for mobile devices and webcams to easily scan physical documents.
*   **Lookalike Digital Copy**: View a cleanly formatted text rendition of the document mimicking traditional medical stationery.

## 🛠 Tech Stack

*   **Frontend**: Vanilla HTML5, CSS3 (Custom Glassmorphism), JavaScript (No frontend build step required)
*   **Backend**: Python, FastAPI, Uvicorn (ASGI server)
*   **AI Engine**: Google Gemini API (`google-genai` SDK)

## 📌 Prerequisites
1. **Python 3.9+** installed locally.
2. A **Google Gemini API Key**. You can get one from Google AI Studio.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhineeth-23/ocr-vlm.git
   cd ocr-vlm
   ```

2. **Set up your environment variables:**
   Create a `.env` file in the root directory and add your API Key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Create and activate a virtual environment:**
   *On Windows (PowerShell):*
   ```powershell
   python -m venv win-venv
   .\win-venv\Scripts\Activate.ps1
   ```
   *On macOS/Linux:*
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install backend dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the App

Start the FastAPI application via `uvicorn`:

```bash
uvicorn main:app --reload
```

Then, open your web browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

## 📷 Usage
1. Upload a medical document using **Browse Files**, **Drag & Drop**, or by capturing one directly with your **Camera**.
2. Click **Start AI Analysis**. The file will securely be processed by the VLM.
3. Review the Extracted Data, Classification, Confidence factors, and the Lookalike Digital Copy directly generated from the image.
4. Export the resulting data to a standard JSON format or a **FHIR-compliant bundle**.
