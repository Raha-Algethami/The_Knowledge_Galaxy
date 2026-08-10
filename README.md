# The Knowledge Galaxy — powered by the Gemini Model

A real-time, physics-based classroom dashboard that takes a student's graded assignment and instantly turns it into a living star field, where students who share the same knowledge gap are pulled together into glowing "Concept Galaxies." This tool is designed to be reliable, scalable, and easy to read.

With implemented Gemini-powered grading, classification, and content generation to automatically evaluate coursework, discover concept clusters, and route each student's star into the correct galaxy in real time.

## How to Use

1. **Installation:** Ensure you have the required libraries listed in `requirements.txt`. (You can install them by running `pip install -r requirements.txt` inside the `backend` folder.) Then add your Gemini API key to `backend/.env` as `GEMINI_API_KEY=your_key_here`.
2. **Running the program:** In your terminal, start the backend by typing:
   `uvicorn main:app --reload --port 8001`
   Then, in a second terminal, serve the frontend by typing:
   `python -m http.server 5500`
   Open `http://localhost:5500` in your browser to launch the app.

## Features

- **[Dynamic Concept Clustering]** Gemini classifies each graded assignment without being given a predefined list of topics — it freely names the exact misconception it finds, and a matching galaxy is created automatically if one doesn't already exist.
- **[Built to Scale]** Designed with a decoupled architecture: the Python backend handles all Gemini API calls and grading logic, while the frontend engine stays exactly the same regardless of how many students, subjects, or galaxies are added.
- **[Self-Organizing Physics Engine]** Real-time orbital simulation pulls each student's star toward its assigned galaxy, with automatic collision-avoidance so galaxies never overlap — and full drag-and-drop control if a teacher wants to manually reposition one.
- **[Easy to Understand]** Every Gemini integration point in the frontend has a documented local fallback, so the app keeps working even before the backend is fully wired up, making it easy for other developers to read, debug, and contribute to the codebase.
- **[AI-Generated Quizzes, Worksheets & Remediation Plans]** Gemini generates cluster-specific quizzes and worksheets — grounded in an uploaded textbook when one is provided — plus a step-by-step remediation plan tailored to each galaxy's specific misconception.
- **[Automated PDF Export]** Integrates directly with jsPDF to export a complete, paginated report for the currently open subject, including every galaxy's roster, confusion levels, and full remediation plan.

### Programming Language
- Python
- JavaScript
- HTML / CSS

### Development Tools
- Google Antigravity IDE
- Google Gemini Model (via the `google-generativeai` SDK)
- FastAPI
- Uvicorn
- Pydantic
- python-dotenv
- Python virtual environments (`venv`)
- HTML5 Canvas API
- jsPDF
- Git
- GitHub
- Google Cloud Run
- Google Cloud CLI (`gcloud`)
- pip / `requirements.txt`

---

## 📂 Project Structure
```text
The_Knowloadge_Galaxy
│
├── backend
│   ├── main.py
│   └── requirements.txt
├── frontend
│   └── index.html
├── gemini_bridge
│   ├── questions
│   └── remediation
└── README.md
```
---

##  Future Upgrades

- **Persistent Storage:** Replace in-memory roster/textbook storage with a real database (e.g. SQLite or Firestore) so class data survives a server restart.
- **Authentication:** Add teacher login so each account only sees their own classes, subjects, and student data.

---

## Author

**Raha Algethami**<br>
Computer Engineering Student<br>
Taif University<br>
Saudi Arabia