# Trading Opportunity Rating (TOR) API

**Student D** | Group 02 | 36120 Advanced Machine Learning

## Overview

FastAPI microservice that predicts the Trading Opportunity Rating (TOR) for cryptocurrencies 7 days into the future using a LightGBM multiclass classifier.

**Target classes:** Strong Sell, Sell, Hold, Buy, Strong Buy

## Student Info

- **Name:** Kunal Mistry
- **Student ID:** 25734346

## Setup

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn app.main:app --reload --port 8000
```

## Docker

```bash
docker build -t tor-api .
docker run -p 8000:8000 tor-api
```

## Endpoints

| Method | Path             | Description                          |
|--------|------------------|--------------------------------------|
| GET    | /                | API metadata and documentation       |
| GET    | /health          | Health check                         |
| POST   | /features        | Engineered features (SMA20, RelVol)  |
| POST   | /predict         | Generate TOR prediction              |
| GET    | /model-metadata  | Model info, features, importance     |

## Environment Variables

| Variable           | Description                  |
|--------------------|------------------------------|
| STUDENT_A_API_URL  | Student A (API) service URL  |
| STUDENT_B_API_URL  | Student B (MRI) service URL  |
| STUDENT_C_API_URL  | Student C (TC) service URL   |

## Deployment

- **GitHub:** https://github.com/KunalMistry000/36120-26SP-Group_2-25734346-API
- **Render:** (to be added after deployment)
