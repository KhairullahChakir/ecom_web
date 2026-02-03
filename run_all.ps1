# OP-ECOM: Multi-Service Launcher
# Start-Process -NoNewWindow would keep everything in one terminal, but separate windows are better for debugging.

Write-Host "🚀 Launching OP-ECOM Ecosystem..." -ForegroundColor Cyan

# 1. Prediction API
Write-Host "Starting Prediction API on Port 8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; python -m uvicorn app.main:app --reload --port 8000"

# 2. Analytics Tracker
Write-Host "Starting Analytics Tracker on Port 8001..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd tracker; python -m uvicorn app.main:app --reload --port 8001"

# 3. Shop Demo
Write-Host "Starting Shop Demo (Vite)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd tracker-demo; npm run dev"

Write-Host "Done! All services are starting in separate windows." -ForegroundColor Cyan
Write-Host "Check http://localhost:5173 to see the demo." -ForegroundColor White
