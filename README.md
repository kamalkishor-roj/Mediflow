# 🚑 MediFlow: Intelligent Patient Triage & Clinical Assistant

MediFlow is an end-to-end, production-grade clinical AI system that combines **machine learning risk prediction** with a **real-time web application** to assist clinicians in early sepsis detection.

The system predicts a patient's sepsis risk from vital signs and presents results via an interactive dashboard.

---

## 🧠 Problem Statement

Hospitals are overwhelmed with high patient loads and time-critical decisions.  
Clinicians need:
- **Predictive intelligence** to identify high-risk patients early
- **Simple, fast tools** that integrate into clinical workflows

MediFlow addresses this by delivering real-time sepsis risk scores from patient vitals.

---

## 🏗️ System Architecture

```

┌────────────────────┐
│   Streamlit UI     │
│ (Clinician Input)  │
└─────────┬──────────┘
          │ HTTP (JSON)
          ▼
┌────────────────────┐
│   FastAPI Backend  │
│  /predict endpoint │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ XGBoost ML Model   │
│ (Sepsis Risk %)    │
└─────────┬──────────┘
          │ 
          ▼
┌────────────────────┐
│   Response Output  │
│ Risk Score (0–100) │
└────────────────────┘

````

---

## 🧪 Machine Learning Pipeline

- Dataset: ICU patient vital signs & lab measurements
- Data Processing: **Polars** (high-performance alternative to Pandas)
- Model: **XGBoost (Binary Classification)**
- Target: `SepsisLabel`
- Evaluation:
  - ROC-AUC ≈ **0.89**
  - PR-AUC ≈ **0.22** (expected due to class imbalance)
- Experiment Tracking: **MLflow**

---

## ⚙️ Tech Stack

| Layer | Technology |
|------|------------|
| Data Processing | Polars |
| Model | XGBoost |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker & Docker Compose |

---

## 🚀 How to Run the Project (Docker)

### Prerequisites
- Docker Desktop (Windows/Linux/Mac)

### 1️⃣ Clone the Repository
```bash
git clone <your-github-repo-url>
cd Mediflow
````

### 2️⃣ Build & Run

```bash
docker compose up --build
```

### 3️⃣ Access the Application

* **FastAPI (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Streamlit Dashboard:** [http://localhost:8501](http://localhost:8501)

---

## 🧑‍⚕️ Usage Example

1. Open the Streamlit dashboard
2. Enter patient vitals (HR, O₂Sat, BP, Temp, etc.)
3. Click **Predict Sepsis Risk**
4. View estimated sepsis risk percentage

---

## 📌 Key Highlights

* End-to-end ML system (training → deployment → UI)
* Production-ready API serving trained ML models
* Fully containerized for reproducible deployment
* Designed with real clinical constraints in mind

---

## 📄 Disclaimer

This project is for **educational and research purposes only** and should not be used for real clinical decision-making.

---

## 👤 Author

**Kamalkishor Roj**
Machine Learning & Data Science Enthusiast

```
