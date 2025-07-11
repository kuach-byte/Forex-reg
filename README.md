Forex-reg

A modular machine learning system for forecasting and classifying Forex market behavior using Python, FastAPI, and Docker. This repository contains:
Machine learning models for trend and volatility detection
A FastAPI-based REST API to serve predictions
Docker-ready structure for deployment
Cleanly organized components for scalability and maintainability


Getting Started

1. Clone the repository

git clone https://github.com/kuach-byte/Forex-reg.git
cd Forex-reg


2. Set up a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r docked-api/forex-api/requirements.txt

3. Run the API

cd docked-api/forex-api
uvicorn main:app --reload

Visit http://127.0.0.1:8000/docs to access the Swagger UI.
