import os
import re
import json
import base64
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import google.generativeai as genai

# Load env variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="The Knowledge Galaxy Backend")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "The Knowledge Galaxy API is running!"}

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY environment variable not found in .env or system environment.")
else:
    genai.configure(api_key=api_key)  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRIDGE_DIR = os.path.join(BASE_DIR, "gemini_bridge")
TEXTBOOK_DIR = os.path.join(BASE_DIR, "textbooks")

# Ensure directories exist
os.makedirs(os.path.join(BRIDGE_DIR, "questions"), exist_ok=True)
os.makedirs(os.path.join(BRIDGE_DIR, "remediation"), exist_ok=True)
os.makedirs(TEXTBOOK_DIR, exist_ok=True)

# Mount static folder
app.mount("/gemini_bridge", StaticFiles(directory=BRIDGE_DIR), name="gemini_bridge")

# --- Helper Functions ---

def slugify(str_val: str) -> str:
    if not str_val:
        return "general"
    val = str_val.lower().strip()
    val = re.sub(r'[^a-z0-9]+', '_', val)
    val = re.sub(r'^_+|_+$', '', val)
    return val if val else "general"

def save_textbook(subject: str, file_name: str, text: str) -> None:
    filename = f"{TEXTBOOK_DIR}/{slugify(subject)}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"file_name": file_name, "text": text}, f, ensure_ascii=False, indent=2)

def load_textbook(subject: str) -> Optional[Dict[str, Any]]:
    filename = f"{TEXTBOOK_DIR}/{slugify(subject)}.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else None
        except Exception:
            return None
    return None

# --- Pydantic Schemas for API Requests/Responses ---

class GradeRequest(BaseModel):
    subject: str
    student_id: Optional[int] = None
    student_name: Optional[str] = None
    notes: str
    image_name: Optional[str] = None
    image_base64: Optional[str] = None
    image_mime_type: Optional[str] = None

class ErrorItem(BaseModel):
    topic: str
    concept: str
    severity: int = Field(..., description="Severity of the misconception from 0 to 100")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0")
    evidence: str = Field(..., description="Short quoted or paraphrased evidence from the work")

class GradeResponse(BaseModel):
    subject: str
    student_name: str
    grade_percent: int = Field(..., description="Final grade score from 0 to 100")
    overall_feedback: str = Field(..., description="Overall summary feedback for the student")
    errors: List[ErrorItem] = Field(default_factory=list)
    assigned_cluster: Optional[str] = Field(None, description="The name of the misconception cluster, or None if grade is 100")
    recommended_interventions: List[str] = Field(..., description="Exactly 3 actionable recommended interventions")

class GenerateQuestionsRequest(BaseModel):
    subject: str
    cluster: str
    type: str # "quiz" or "worksheet"

class QuestionItem(BaseModel):
    question: str
    options: List[str]
    correct_index: int
    explanation: str

class QuestionsResponse(BaseModel):
    cluster: str
    type: str
    generated_by: str = "gemini-3.1-pro-preview"
    questions: List[QuestionItem]

class GenerateRemediationRequest(BaseModel):
    subject: str
    cluster: str

class RemediationStep(BaseModel):
    title: str = Field(..., description="Step title/number, e.g. 'Step 1: Diagnostic Misconception analysis'")
    description: str = Field(..., description="Step detailed description and guidelines")

class RemediationResponse(BaseModel):
    cluster: str
    generated_by: str = "gemini-3.1-pro-preview"
    steps: List[RemediationStep]

class UploadTextbookRequest(BaseModel):
    subject: str
    file_name: str
    text: str

# --- structured output helper ---
def clean_schema(d: Any) -> Any:
    if not isinstance(d, dict):
        return d
    cleaned = {}
    for k, v in d.items():
        if k == 'default':
            continue
        if isinstance(v, dict):
            cleaned[k] = clean_schema(v)
        elif isinstance(v, list):
            cleaned[k] = [clean_schema(item) if isinstance(item, dict) else item for item in v]
        else:
            cleaned[k] = v
    return cleaned

def generate_structured_json(contents: Any, response_schema: Any, model_name: str = "gemini-3.1-pro-preview") -> Any:
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not configured.")
    
    # Generate schema dictionary from Pydantic model
    schema_dict = None
    if hasattr(response_schema, "model_json_schema"):
        schema_dict = response_schema.model_json_schema()
    elif hasattr(response_schema, "schema"):
        schema_dict = response_schema.schema()

    if schema_dict:
        # Strip all 'default' keys because google-generativeai SDK doesn't support them
        schema_dict = clean_schema(schema_dict)

    fallback_models = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-2.5-pro"]
    if model_name not in fallback_models:
        fallback_models.insert(0, model_name)

    last_error = None
    for model_attempt in fallback_models:
        try:
            print(f"Attempting to generate structured JSON using model: {model_attempt}")
            model = genai.GenerativeModel(model_attempt)  # type: ignore
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(  # type: ignore
                    response_mime_type="application/json",
                    response_schema=schema_dict if schema_dict else response_schema
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Failed with model {model_attempt}: {e}")
            last_error = e

    # If all structured schema attempts failed, attempt fallback without schema validation
    print("Attempting fallback generation without schema validation...")
    for model_attempt in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-2.5-pro"]:
        try:
            model = genai.GenerativeModel(model_attempt)  # type: ignore
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(  # type: ignore
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            last_error = e

    raise HTTPException(status_code=500, detail=f"Failed to generate structured content from Gemini API: {str(last_error)}")

# --- Routes ---

@app.post("/grade", response_model=GradeResponse)
async def grade_assignment(req: GradeRequest) -> Any:
    contents: List[Any] = []

    # Prepare base64 image if exists
    if req.image_base64 and req.image_mime_type:
        try:
            image_bytes = base64.b64decode(req.image_base64)
            contents.append({
                "mime_type": req.image_mime_type,
                "data": image_bytes
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")

    # Load textbook if exists for additional grounding context
    textbook_info = load_textbook(req.subject)
    textbook_grounding = ""
    if textbook_info:
        # Excerpt from textbook to assist assessment
        textbook_grounding = f"\nReferenced Textbook content:\n{textbook_info['text'][:12000]}\n"

    # Construct the instruction prompt
    prompt = f"""You are an expert grading assistant assessing student submissions.
Subject: {req.subject}
Student Name: {req.student_name or 'Unknown'}
Student ID: {req.student_id or 'N/A'}
Teacher's Observations/Notes: {req.notes}
{textbook_grounding}

If an image is attached, analyze the handwritten or typed work in the image. Integrate your observations with the teacher's notes.

Requirements:
1. Provide a grade percentage (0-100) based on their mastery.
2. Provide a detailed summary of your feedback in `overall_feedback`.
3. If the student's grade is 100%, they have full mastery. You must set `errors` to an empty list and `assigned_cluster` to null.
4. If the student's grade is less than 100%, identify the specific errors they made. For each error:
   - Identify the specific `topic` (e.g. 'Fraction Addition', 'Loop control').
   - Describe the incorrect `concept` (e.g. 'Adding denominators directly').
   - Assess the `severity` (0-100).
   - Set your `confidence` (0.0-1.0).
   - Quote or paraphrase the `evidence` of this error from the teacher's notes or the image.
5. If the student has errors (grade < 100), identify the primary misconception and label it as `assigned_cluster`. This cluster name should be a concise, descriptive name of the misconception, in the words of a professional teacher (e.g., "Denominator Addition Error", "Recursion Stack Overflow", "Off-by-One Loop Boundary"). Do NOT pick from any predefined list; write a brand-new name that fits the error.
6. Provide exactly 3 actionable, high-quality recommended interventions in `recommended_interventions` to help the student overcome this specific misconception.

You must return a JSON object that strictly adheres to the requested schema.
"""
    contents.append(prompt)

    # Call Gemini
    result_data = generate_structured_json(contents, GradeResponse)
    
    # Standardize result name & subject
    result_data["subject"] = req.subject
    result_data["student_name"] = req.student_name or "Student"
    
    # Handle the null assigned_cluster for 100% grade
    if result_data.get("grade_percent") == 100:
        result_data["assigned_cluster"] = None
        result_data["errors"] = []

    return result_data


@app.post("/generate_questions", response_model=QuestionsResponse)
async def generate_questions(req: GenerateQuestionsRequest) -> Any:
    textbook_info = load_textbook(req.subject)
    textbook_grounding = ""
    if textbook_info:
        textbook_grounding = f"\nTextbook Grounding Context ({textbook_info['file_name']}):\n{textbook_info['text']}\n"
    else:
        textbook_grounding = "\nGenerate questions based on standard curriculum baselines since no textbook is uploaded.\n"

    prompt = f"""You are an expert curriculum developer. Your task is to generate 5 multiple-choice questions for the following concept/misconception.

Subject: {req.subject}
Concept/Misconception Cluster: {req.cluster}
Type of Output: {req.type} (generate multiple-choice questions suitable for a {req.type})
{textbook_grounding}

Requirements:
1. Generate exactly 5 questions.
2. For each question:
   - Provide the question text.
   - Provide exactly 4 options.
   - Provide the `correct_index` (0-3) of the correct option.
   - Provide a clear, educational `explanation` of why that answer is correct.
3. If textbook grounding is provided, you MUST construct the questions using the terminology, definitions, and context from that textbook excerpt.
4. Output must match the requested schema perfectly.
"""
    result_data = generate_structured_json(prompt, QuestionsResponse)
    result_data["cluster"] = req.cluster
    result_data["type"] = req.type
    result_data["generated_by"] = "gemini-3.1-pro-preview"

    # Write to gemini_bridge/questions/<cluster_slug>_<type>.json
    cluster_slug = slugify(req.cluster)
    filename = os.path.join(BRIDGE_DIR, "questions", f"{cluster_slug}_{req.type}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return result_data


@app.post("/generate_remediation", response_model=RemediationResponse)
async def generate_remediation(req: GenerateRemediationRequest) -> Any:
    prompt = f"""You are an expert educational psychologist and teacher. Your task is to design a step-by-step remediation guide for a student struggling with the following misconception.

Subject: {req.subject}
Concept/Misconception Cluster: {req.cluster}

Requirements:
1. Generate between 4 and 6 sequential steps.
2. The plan must include:
   - A diagnostic analysis step to identify the root cause of the misconception.
   - A re-teaching step presenting the concept with a different modality (e.g., visual models, hands-on, tracing).
   - A guided practice step with scaffolding.
   - A peer or individual practice step.
   - A final re-assessment step.
3. Let the steps' titles and descriptions be specific to the subject and the misconception cluster, rather than generic templates.
4. Output must match the requested schema perfectly.
"""
    # Use aliases or dict for mapping to 'num' and 'desc'
    # Pydantic schema will handle parsing it.
    result_data = generate_structured_json(prompt, RemediationResponse)
    result_data["cluster"] = req.cluster
    result_data["generated_by"] = "gemini-3.1-pro-preview"

    # Write to gemini_bridge/remediation/<cluster_slug>.json
    cluster_slug = slugify(req.cluster)
    filename = os.path.join(BRIDGE_DIR, "remediation", f"{cluster_slug}.json")
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return result_data


@app.post("/upload_textbook")
async def upload_textbook(req: UploadTextbookRequest) -> Dict[str, str]:
    save_textbook(req.subject, req.file_name, req.text)
    return {"status": "success", "file_name": req.file_name, "subject": req.subject}
