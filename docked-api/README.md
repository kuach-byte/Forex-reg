
Forex Prediction API

This is a FastAPI-based REST API for predicting **trend direction** and **volatility levels** of 9 USD-based Forex currency pairs using 18 pre-trained machine learning models (9 for trend, 9 for volatility).

---

Project Structure

```
forex-api/
├── app/
│   ├── app_main.py                
│   ├── requirements.txt      
│   ├── models/                
│   └── utils/               
│       ├── __init__.py
│       
├── Dockerfile                
├── .dockerignore             
└── README.md               


Supported Currency Pairs

- `AUDUSD`
- `EURUSD`
- `GBPUSD`
- `NZDUSD`
- `USDCAD`
- `USDCHF`
- `USDHKD`
- `USDNOK`
- `USDSEK`



Features

- Predicts trend class: Uptrend, Ranging, Downtrend
- Predicts volatility class: Low, Medium, High
- Built on FastAPI with auto docs
- Fully Dockerized
- Accepts already-processed features (model-ready)



Setup Instructions

1. Clone the Repository

2. Place Model Files
Place the 18 `.joblib` files inside:

```
app/models/
├── EURUSD_model.joblib
├── EURUSD_vol_model.joblib
├── ...
```

> **Naming convention:** `{PAIR}_model.joblib` and `{PAIR}_vol_model.joblib`

---

### 3. Docker Build
```bash
docker build -t forex-api .
```

### 4. Run the API Container
```bash
docker run -d -p 8000:8000 --name forex_container forex-api
```
## Or build and run with
docker compose up --build -d


Test the API

### Swagger UI:
Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

### Example Request:
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data "@C:\\Users\\Kuach\\Desktop\\"preprocessed data.json"

```

### Example Response:
```json
{
  "pair": "EURUSD",
  "trend_class": 1,
  "trend_label": "Uptrend",
  "vol_class": 2,
  "vol_label": "High"
}
```




Development (No Docker)

### Prerequisites
- Python 3.8.10
- pip

#] ## Install and run locally:
```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload
```



Future Enhancements

-  Accept raw OHLCV data and compute features inside the API
-  Integrate live MT5 data
-  Add SQLite logging or metrics
-  Secure endpoints with API key or OAuth
-  Add database or cronjob for batch inference



This project is for educational and research purposes only. Not financial advice.