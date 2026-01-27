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

```
op_ecom/
├── backend/          # FastAPI + ML service
├── frontend/         # Next.js demo website
├── notebooks/        # Training & experiments
├── data/             # Dataset
└── reports/          # Metrics & figures
```

## 🚀 Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

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
