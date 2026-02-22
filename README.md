# Solaritivity - AI Driven Solar Defect Intelligence Platform

An end to end governed AI system for Solar PV defect detection, impact estimation, and carbon intelligence. 
Built to power India’s march toward Viksit Bharat 2047 with reliable, efficient, and safe solar infrastructure.

---

## **Problem Statement**

As India races toward Viksit Bharat 2047, solar energy is positioned to become the backbone of sustainable industrial growth. Yet, Solar PV modules silently lose 5-30% energy due to micro-cracks, manufacturing defects, and thermal hotspots. These degradations shorten module lifespan by 3-8 years, increase O&M costs, and elevate fire risks  directly impacting national grid reliability, industrial productivity, and carbon efficiency. Despite 800+ GW of installed capacity worldwide, inspections remain manual, inconsistent, and non governed.

---

## **Project Idea**

Solaritivity is built on five core pillars - Integrity, Detectability, Interpretability, Sustainability, and Explainability forming a unified, future ready ecosystem for governed solar defect intelligence.

- **Multi modal defect detection & localization**
- **Energy loss + carbon impact analysis**
- **RAG Based Personalized Chatbot** with sentiment-driven response behavior
- **Digital Twin Layer** for future simulation modules
- **Explainable AI (XAI)** visualizations
- **Pipeline Automation** (Ingestion → Detection → Insights)
- **Real time monitoring layer** for enterprise ready deployment

---

## **Project Structure**

```
HACKATHON_KARUR/
│
├── auth/                    # login + authentication logic
├── chroma_db/               # vector DB for RAG
├── data/                    # raw + processed datasets
├── docs/                    # documentation and reference files
├── explanations/            # XAI output maps (gradcam, saliency)
├── fyp_pycell/              # extra scripts / experimental modules
├── invalid_test_images/     # test images for invalid-input handling
├── model/                   # ML/DL models, weights, checkpoints
├── modules/                 # all core backend modules
├── results/                 # predictions, summaries, analytics results
├── static/        
│ │ ├── script.js
│ │ ├── styles.css
├── templates/
│ ├── partials/              # reusable UI components
│ │ ├── css.html
│ │ ├── js.html
│ │ ├── nav.html
│ │ ├── panel_carbon.html
│ │ ├── panel_chat.html
│ │ ├── panel_detect.html
│ │ ├── panel_detect_thermal.html
│ │ ├── panel_summary.html
│ │ ├── panel_thermal.html
│ │ ├── panel_xai.html
│ │ ├── panel_xai_thermal.html
│ │ └── panel_xai1.html
│ ├── dashboard.html
│ └── login.html
│
├── thermal_uploads/         # thermal image uploads
├── uploads/                 # visual image uploads
├── valid_test_images/       # valid input test samples
│
├── app.py                   # main app runner
├── config.py                # configuration + constants
├── gemini_check.py          # Gemini API related script
├── invalid_gen.py           # script for invalid image generation
├── training_vgg19.ipynb     # model training notebook
├── requirements.txt         # dependencies
└── .env                     # environment variables
```

---

## **System Architecture**

<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/700b97d2-22d8-4e6c-9c00-9f97317573db" />

---

## **How to Use (Quick Start Guide)**

Follow these steps to run Solaritivity.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/paranthagan78/SolaritivityPlus
cd SolaritivityPlus
```

### 2️⃣ Create a Virtual Environment & Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Ingest the Document Dataset
```bash
python ingest.py
```

### 4️⃣ Start the Application
```bash
python app.py
```
