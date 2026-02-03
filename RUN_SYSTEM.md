# OP-ECOM: Full Ecosystem Guide

This document explains how to run the complete integrated system, from the raw data tracking to the AI-powered intervention.

## 🏗️ Architecture
1. **Prediction API** (Port 8000): Host the optimized TabM model.
2. **Analytics Tracker** (Port 8001): Collects real-time events and decides when to intervene.
3. **Shop Demo** (Port 5173): The test website where you can see the AI in action.

## 🚀 How to Run

### Automatic (Recommended)
Run the PowerShell script in the root directory:
```powershell
./run_all.ps1
```

### Manual
If you prefer separate terminals:

1. **Start Prediction API**:
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload --port 8000
   ```

2. **Start Analytics Tracker**:
   ```bash
   cd tracker
   python -m uvicorn app.main:app --reload --port 8001
   ```

3. **Start Shop Demo**:
   ```bash
   cd tracker-demo
   npm run dev
   ```

## 🧪 Testing the AI
1. Open `http://localhost:5173` in your browser.
2. **Act like a buyer**: Click on products, view details, and add items to your cart.
3. **The Exit Intent**: Move your mouse towards the top of the browser window (near the address bar).
4. **The Result**: An AI-powered popup should appear, offering a special discount calculated by your **TabM model** based on your session behavior!
