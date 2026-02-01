# OP-ECOM: Online Shoppers Purchase Prediction API

A fast CPU inference ML service predicting **purchase intent** using the UCI Online Shoppers dataset.

## 🎯 Features

- **Model-as-an-API**: POST `/predict` → returns label, probability, latency
- **Fast Inference**: ONNX Runtime optimized for <10ms CPU inference
- **Beautiful Demo**: Next.js frontend with لاجوردی (Lapis Lazuli) theme

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python, FastAPI, ONNX Runtime |
| Frontend | Next.js, React, Tailwind CSS |
| ML Models | Logistic Regression, XGBoost, TabM |

## 🎨 Theme Colors (لاجوردی)

- Primary: `#1E4FA8`
- Dark: `#163B7A`
- Light: `#E8F0FF`

## 📁 Project Structure

| Component | Path | Technology | Port |
|-----------|------|------------|------|
| **AI Brain** | `backend/` | FastAPI + ONNX | 8000 |
| **Tracker API** | `tracker/` | FastAPI + MariaDB | 8001 |
| **Predictor UI** | `frontend/` | Next.js | 3000 |
| **E-com Demo** | `tracker-demo/` | Vite + Vue | 5173 |

## 🚀 Running the Project (Developer Shortcut)

The easiest way to work on the project in VS Code is to use **one single virtual environment** at the root.

### 1. Create & Activate Environment
```powershell
# Run from D:\op_ecom
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install All Dependencies
```powershell
pip install fastapi uvicorn pandas numpy scikit-learn xgboost torch onnxruntime sqlalchemy mysqlclient requests python-multipart
```

### 3. VS Code Integration
*   Press `Ctrl + Shift + P`.
*   Select **Python: Select Interpreter**.
*   Choose the root `.\venv\Scripts\python.exe`. (This clears all red squiggles in your code!)

### 4. Start the Servers (Different Terminals)
*   **AI Brain**: `python -m uvicorn backend.app.main:app --port 8000`
*   **Tracker**: `python -m uvicorn tracker.app.main:app --port 8001`
*   **Predictor UI**: `cd frontend; npm run dev`
*   **E-com Demo**: `cd tracker-demo; npm run dev`

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Get prediction |

## 📈 Dataset

UCI Online Shoppers Purchasing Intention
- 12,330 sessions
- 18 features
- Target: Revenue (Yes/No)
