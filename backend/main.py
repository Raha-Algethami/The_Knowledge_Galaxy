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

def save_textbook(subject: str, file_name: str, text: str, file_uri: Optional[str] = None, gemini_file_id: Optional[str] = None, is_base64: bool = False) -> None:
    filename = f"{TEXTBOOK_DIR}/{slugify(subject)}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"file_name": file_name, "text": text, "file_uri": file_uri, "gemini_file_id": gemini_file_id, "is_base64": is_base64}, f, ensure_ascii=False, indent=2)

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
    existing_gaps: Optional[List[str]] = Field(
        default=None,
        description="Existing misconception galaxy labels already identified for this subject. Reuse one exactly if it matches."
    )

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
    is_base64: bool = False

# --- structured output helper ---
def clean_schema(d: Any, defs: Optional[dict] = None) -> Any:
    """Recursively clean a Pydantic-generated JSON schema for the Gemini SDK.

    The Gemini SDK does NOT support several JSON-Schema keywords:
      - 'default'  : remove
      - '$defs'    : inline & remove
      - 'title'    : remove
      - 'anyOf'    : unwrap — Optional[X] becomes just the non-null type
    """
    if defs is None and isinstance(d, dict) and "$defs" in d:
        defs = d["$defs"]

    if not isinstance(d, dict):
        return d

    if "$ref" in d:
        ref_path = d["$ref"]
        def_name = ref_path.split("/")[-1]
        if defs and def_name in defs:
            return clean_schema(defs[def_name], defs)

    # Unwrap Optional[X] → anyOf: [{type: X}, {type: null}]
    if "anyOf" in d:
        non_null = [s for s in d["anyOf"] if not (isinstance(s, dict) and s.get("type") == "null")]
        if len(non_null) == 1:
            # Merge any sibling keys (e.g. 'description') into the unwrapped schema
            unwrapped = clean_schema(non_null[0], defs)
            for k, v in d.items():
                if k not in ("anyOf", "default", "title") and k not in unwrapped:
                    unwrapped[k] = v
            return unwrapped

    cleaned = {}
    for k, v in d.items():
        if k in ('default', '$defs', 'title'):
            continue
        if isinstance(v, dict):
            cleaned[k] = clean_schema(v, defs)
        elif isinstance(v, list):
            cleaned[k] = [clean_schema(item, defs) if isinstance(item, dict) else item for item in v]
        else:
            cleaned[k] = v
    return cleaned

def generate_structured_json(contents: Any, response_schema: Any, model_name: str = "gemini-3.1-pro-preview") -> tuple:
    """Call Gemini with structured JSON output.

    Returns:
        A tuple of (parsed_json: dict, model_used: str) so callers can record
        which model actually succeeded.
    """
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

    # --- Confirmed, live model IDs (verified via models.list() on 2026-08-11) ---
    # Pro models first — accuracy over speed for grading/classification.
    # Flash models are only a last resort.
    # NOTE: gemini-3-pro-preview does NOT exist in the API; it has been removed.
    PRO_MODELS = [
        "gemini-3.1-pro-preview",   # confirmed via models.list()
        "gemini-2.5-pro",           # confirmed via models.list()
    ]
    FLASH_MODELS = [
        "gemini-3.5-flash",         # confirmed via models.list()
        "gemini-3.6-flash",         # confirmed via models.list()
        "gemini-flash-latest",      # confirmed via models.list()
    ]

    # Build ordered list: requested model first (if not already present), then all Pro, then Flash
    fallback_models: list = []
    if model_name not in PRO_MODELS and model_name not in FLASH_MODELS:
        fallback_models.append(model_name)
    fallback_models.extend(PRO_MODELS)
    fallback_models.extend(FLASH_MODELS)
    # Deduplicate while preserving order
    seen: set = set()
    fallback_models = [m for m in fallback_models if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]

    last_error = None

    # Pass 1: structured schema output
    for model_attempt in fallback_models:
        try:
            print(f"[generate_structured_json] Attempting model: {model_attempt} (with schema)")
            model = genai.GenerativeModel(model_attempt)  # type: ignore
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(  # type: ignore
                    response_mime_type="application/json",
                    response_schema=schema_dict if schema_dict else response_schema
                )
            )
            parsed = json.loads(response.text)
            print(f"[generate_structured_json] SUCCESS with model: {model_attempt}")
            return parsed, model_attempt
        except Exception as e:
            print(f"[generate_structured_json] FAILED with model {model_attempt}: {e}")
            last_error = e

    # Pass 2: fallback without schema validation (same Pro-first order)
    print("[generate_structured_json] All schema attempts failed. Retrying without schema validation...")
    for model_attempt in fallback_models:
        try:
            print(f"[generate_structured_json] Attempting model: {model_attempt} (no schema)")
            model = genai.GenerativeModel(model_attempt)  # type: ignore
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(  # type: ignore
                    response_mime_type="application/json"
                )
            )
            parsed = json.loads(response.text)
            print(f"[generate_structured_json] SUCCESS (no schema) with model: {model_attempt}")
            return parsed, model_attempt
        except Exception as e:
            print(f"[generate_structured_json] FAILED (no schema) with model {model_attempt}: {e}")
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
        if textbook_info.get("gemini_file_id"):
            try:
                gemini_file = genai.get_file(textbook_info["gemini_file_id"])
                contents.append(gemini_file)
                textbook_grounding = f"\nReferenced Textbook content is provided as an attached document ({textbook_info['file_name']}).\n"
            except Exception as e:
                print(f"Warning: Could not fetch gemini file {textbook_info['gemini_file_id']}: {e}")
        else:
            # Excerpt from textbook to assist assessment
            textbook_grounding = f"\nReferenced Textbook content:\n{textbook_info['text'][:12000]}\n"

    # Construct the instruction prompt
    # Build existing-gaps guidance so Gemini reuses exact cluster names instead
    # of inventing a near-duplicate each time the same misconception recurs.
    existing_gaps_guidance = ""
    if req.existing_gaps and len(req.existing_gaps) > 0:
        gaps_list = ", ".join(f'"{g}"' for g in req.existing_gaps)
        existing_gaps_guidance = f"""
Existing misconception categories already identified for {req.subject}:
{gaps_list}

CRITICAL: If the student's primary error clearly matches one of the existing categories above,
you MUST return that EXACT label as `assigned_cluster`, copied character-for-character
(including capitalisation and spacing). Only create a brand-new label if the misconception
is genuinely different from every category in the list above.
"""

    prompt = f"""You are an expert grading assistant assessing student submissions.
Subject: {req.subject}
Student Name: {req.student_name or 'Unknown'}
Student ID: {req.student_id or 'N/A'}
Teacher's Observations/Notes: {req.notes}
{textbook_grounding}
{existing_gaps_guidance}
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
5. If the student has errors (grade < 100), identify the primary misconception and label it as `assigned_cluster`. If it matches an existing category from the list above, use that exact label. Otherwise write a concise, descriptive name in the words of a professional teacher (e.g., "Denominator Addition Error", "Recursion Stack Overflow", "Off-by-One Loop Boundary").
6. Provide exactly 3 actionable, high-quality recommended interventions in `recommended_interventions` to help the student overcome this specific misconception.

You must return a JSON object that strictly adheres to the requested schema.
"""
    contents.append(prompt)

    # Call Gemini — returns (parsed_json, model_used)
    result_data, model_used = generate_structured_json(contents, GradeResponse)
    print(f"[/grade] Grading completed. Model used: {model_used}")

    # Standardize result name & subject
    result_data["subject"] = req.subject
    result_data["student_name"] = req.student_name or "Student"

    # Normalise grade field — Gemini sometimes returns 'grade' or 'grade_percentage'
    # instead of the exact schema name 'grade_percent'. Remap any known aliases.
    for alias in ("grade_percentage", "grade", "score", "grade_score"):
        if alias in result_data and "grade_percent" not in result_data:
            result_data["grade_percent"] = result_data.pop(alias)
            print(f"[/grade] Remapped '{alias}' → 'grade_percent'")
            break

    # Handle the null assigned_cluster for 100% grade
    if result_data.get("grade_percent") == 100:
        result_data["assigned_cluster"] = None
        result_data["errors"] = []

    return result_data


@app.post("/generate_questions", response_model=QuestionsResponse)
async def generate_questions(req: GenerateQuestionsRequest) -> Any:
    contents_list: List[Any] = []
    textbook_info = load_textbook(req.subject)
    textbook_grounding = ""
    if textbook_info:
        if textbook_info.get("gemini_file_id"):
            try:
                gemini_file = genai.get_file(textbook_info["gemini_file_id"])
                contents_list.append(gemini_file)
                textbook_grounding = f"\nTextbook Grounding Context is provided as an attached document ({textbook_info['file_name']}).\n"
            except Exception as e:
                print(f"Warning: Could not fetch gemini file {textbook_info['gemini_file_id']}: {e}")
                textbook_grounding = "\nGenerate questions based on standard curriculum baselines since the attached textbook could not be loaded.\n"
        else:
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
3. Ensure high pedagogical rigor:
   - Use plausible, non-obvious distractors (do NOT use silly or easy-to-eliminate wrong answers).
   - Maintain grade-appropriate difficulty.
   - Ensure high variety in question phrasing, context, and structure across the 5 questions (avoid similar-looking repeats).
4. If textbook grounding is provided, you MUST construct the questions using the terminology, definitions, and context from that textbook excerpt.
5. Output must match the requested schema perfectly.
"""
    contents_list.append(prompt)
    result_data, model_used = generate_structured_json(contents_list, QuestionsResponse)
    print(f"[/generate_questions] Questions generated. Model used: {model_used}")
    result_data["cluster"] = req.cluster
    result_data["type"] = req.type
    result_data["generated_by"] = model_used

    # Write to gemini_bridge/questions/<cluster_slug>_<type>.json
    cluster_slug = slugify(req.cluster)
    filename = os.path.join(BRIDGE_DIR, "questions", f"{cluster_slug}_{req.type}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return result_data


@app.post("/generate_remediation", response_model=RemediationResponse)
async def generate_remediation(req: GenerateRemediationRequest) -> Any:
    contents_list: List[Any] = []
    textbook_info = load_textbook(req.subject)
    textbook_grounding = ""
    if textbook_info:
        if textbook_info.get("gemini_file_id"):
            try:
                gemini_file = genai.get_file(textbook_info["gemini_file_id"])
                contents_list.append(gemini_file)
                textbook_grounding = f"\nTextbook Grounding Context is provided as an attached document ({textbook_info['file_name']}). Please use this textbook to align the remediation steps with the curriculum.\n"
            except Exception as e:
                print(f"Warning: Could not fetch gemini file {textbook_info['gemini_file_id']}: {e}")
        else:
            textbook_grounding = f"\nTextbook Grounding Context ({textbook_info['file_name']}):\n{textbook_info['text']}\n"

    prompt = f"""You are an expert educational psychologist and teacher. Your task is to design a step-by-step remediation guide for a student struggling with the following misconception.

Subject: {req.subject}
Concept/Misconception Cluster: {req.cluster}
{textbook_grounding}
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
    contents_list.append(prompt)
    # Use aliases or dict for mapping to 'num' and 'desc'
    # Pydantic schema will handle parsing it.
    result_data, model_used = generate_structured_json(contents_list, RemediationResponse)
    print(f"[/generate_remediation] Remediation generated. Model used: {model_used}")
    result_data["cluster"] = req.cluster
    result_data["generated_by"] = model_used

    # Write to gemini_bridge/remediation/<cluster_slug>.json
    cluster_slug = slugify(req.cluster)
    filename = os.path.join(BRIDGE_DIR, "remediation", f"{cluster_slug}.json")
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return result_data


@app.post("/upload_textbook")
async def upload_textbook(req: UploadTextbookRequest) -> Dict[str, str]:
    if req.is_base64:
        import uuid
        data_parts = req.text.split(",")
        b64_data = data_parts[1] if len(data_parts) > 1 else req.text
        
        ext = os.path.splitext(req.file_name)[1]
        temp_path = f"temp_upload_{uuid.uuid4().hex}{ext}"
        
        try:
            with open(temp_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            
            # Decode the file locally to bypass the broken Gemini File API!
            decoded_text = base64.b64decode(b64_data).decode('utf-8', errors='ignore')
            save_textbook(req.subject, req.file_name, decoded_text, is_base64=False)
            return {"status": "success", "file_name": req.file_name, "subject": req.subject}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Gemini upload failed: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        save_textbook(req.subject, req.file_name, req.text, is_base64=False)
        return {"status": "success", "file_name": req.file_name, "subject": req.subject}
