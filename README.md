# Demo Actions - Customer Churn Prediction API

> FastAPI-based ML service for customer churn prediction

## Overview

A machine learning API service built with FastAPI that predicts customer churn using a Logistic Regression model. Based on customer data, it predicts the likelihood of customer attrition.

## Key Features

- **Churn Prediction**: Logistic Regression model-based prediction
- **RESTful API**: FastAPI prediction endpoint
- **Auto Documentation**: Swagger UI auto-generated
- **GitHub Actions**: CI/CD pipeline example

## Tech Stack

- Python 3.8+
- FastAPI
- scikit-learn
- joblib
- Pydantic

## Quick Start

```bash
pip install fastapi uvicorn scikit-learn joblib pandas
python app.py
# or: uvicorn app:app --reload
```

## API

### POST /predict

**Request:**
```json
{
  "tenure": 12,
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "TechSupport": "No",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check"
}
```

**Response:**
```json
{"prediction": 1}
```

## Project Structure

```
demo_actions/
├── app.py              # FastAPI main server
├── PreProcessing.py    # Data preprocessing
└── logreg_model.joblib # Trained model
```
