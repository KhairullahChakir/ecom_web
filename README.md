# E-Commerce Analytics & Machine Learning Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)

## 📖 Overview
The **E-Commerce Analytics & Machine Learning Platform** (`ecom_web`) is an advanced data science repository dedicated to analyzing e-commerce behavior and optimizing business metrics using machine learning. 

This repository encapsulates the full data lifecycle: from raw dataset analysis and exploratory data analysis (EDA) in Jupyter Notebooks, to training multiple predictive models, and ultimately deploying these models via a backend service. It also includes comprehensive reports detailing the model architectures and findings.

## ✨ Features
* **Extensive Data Analysis**: `analyze_dataset.py` and Jupyter notebooks to uncover trends in e-commerce purchasing behavior.
* **Predictive Modeling**: Multiple machine learning models tracked and evaluated against business KPIs.
* **Comprehensive Documentation**: Detailed research reports, model performance metrics, and architectural breakdowns (located in the `/reports` directory).
* **Containerized Deployment**: Ready-to-deploy backend architecture using Docker and `docker-compose`.
* **Cross-Platform Scripts**: Includes PowerShell (`.ps1`) and Bash (`.sh`) scripts for seamless local execution and deployment.

## 🚀 Technologies Used
* **Data Science**: Python, Pandas, Scikit-Learn, Jupyter Notebook
* **Backend**: Python (Flask/FastAPI context)
* **Infrastructure**: Docker, Docker Compose
* **Scripting**: PowerShell, Bash

## 📁 Project Structure
```text
ecom_web/
├── backend/                  # API server for model inference
├── data/                     # Raw and processed datasets
├── notebooks/                # Jupyter Notebooks for EDA and Model Training
├── reports/                  # Markdown research reports and thesis chapters
│   ├── COMPREHENSIVE_FINAL_REPORT.md
│   ├── FULL_PROJECT_REPORT.md
│   ├── MODEL_1_REPORT.md
│   ├── MODEL_2_REPORT.md
│   └── thesis_chapter_4_architecture.md
├── scripts/                  # Helper automation scripts
├── tracker/                  # Model tracking services
├── tracker-demo/             # Demo application for the tracking service
├── analyze_dataset.py        # Core dataset analysis script
├── deploy.sh                 # Unix deployment script
├── docker-compose.yml        # Docker orchestration file
├── run_all.ps1               # Windows execution script
├── run_local.ps1             # Windows local testing script
├── .gitignore                # Git ignores
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

## 📸 Screenshots
*(Coming soon - Screenshots of the analytics dashboard and model metrics)*

## 🛠️ Installation & Setup

### 1. Data Science Environment
If you wish to run the analysis scripts or notebooks:
```bash
git clone https://github.com/KhairullahChakir/ecom_web.git
cd ecom_web

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements (ensure you have jupyter, pandas, scikit-learn installed)
pip install -r requirements.txt # (If available in scripts)
python analyze_dataset.py
```

### 2. Docker Deployment
To spin up the backend and tracking services using Docker:
```bash
docker-compose up --build -d
```

### 3. Local Script Execution (Windows)
```powershell
.\run_local.ps1
```

## 🔮 Future Improvements
* **Automated Data Pipelines**: Integrate Apache Airflow or Luigi for automated daily data ingestion.
* **Web Dashboard**: Implement a Streamlit or Dash frontend to visualize model predictions for non-technical users.
* **CI/CD**: Add GitHub Actions for automated linting and Docker image builds.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
