#!/bin/bash
# OP-ECOM Production Deployment Script (for Ubuntu VPS)

echo "🚀 Starting OP-ECOM Production Stack..."

# 1. Install Docker and Docker Compose if not found
if ! [ -x "$(command -v docker)" ]; then
  echo "Installing Docker..."
  curl -fsSL https://get.docker.com -o get-docker.sh
  sh get-docker.sh
fi

# 2. Stop and remove existing containers
echo "Stopping existing services..."
docker compose down

# 3. Build and launch the stack
echo "Building services (this may take a few minutes)..."
docker compose up -d --build

echo "✅ Ecosystem is now LIVE!"
echo "------------------------------------------------"
echo "👉 Shop Demo: http://185.197.31.25:5173"
echo "👉 Admin Dashboard: http://185.197.31.25:3000"
echo "👉 Prediction API: http://185.197.31.25:8000/health"
echo "👉 Analytics Tracker: http://185.197.31.25:8002/health"
echo "------------------------------------------------"
echo "Use 'docker compose logs -f' to monitor traffic."
