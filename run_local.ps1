# Unified Startup Script for OP-ECOM
# This script starts the AI Backend, Analytics Tracker, and Shop Demo in separate windows.

Write-Host "🚀 Starting OP-ECOM Local Ecosystem..." -ForegroundColor Cyan

# 1. Start AI Backend (Port 8000)
Write-Host "Starting AI Backend on Port 8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd d:\op_ecom\backend; python -m uvicorn app.main:app --port 8000"

# 2. Start Analytics Tracker (Port 8002)
Write-Host " Starting Analytics Tracker on Port 8002..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd d:\op_ecom\tracker; python -m uvicorn app.main:app --port 8002"

# 3. Start Shop Demo Website (Port 5173)
Write-Host "Starting Shop Demo Website on Port 5173..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd d:\op_ecom\tracker-demo; npm run dev"

Write-Host "All services initiated. Check the new windows for status!" -ForegroundColor Yellow
Write-Host "Website: http://localhost:5173" -ForegroundColor White
Write-Host "Backend: http://localhost:8000/health" -ForegroundColor White
Write-Host "Tracker: http://localhost:8002/health" -ForegroundColor White
