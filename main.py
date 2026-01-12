from fastapi import FastAPI
from pydantic import BaseModel
from database import SessionLocal, News
from sqlalchemy import func

import joblib
import re

app = FastAPI(title="AI News Credibility API")

# Load ML assets
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class NewsRequest(BaseModel):
    title: str
    content: str

def clean(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

@app.get("/")
def home():
    return {"message": "AI News Credibility API is running"}

@app.post("/analyze")
def analyze_news(news: NewsRequest):
    text = clean(news.title + " " + news.content)
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    confidence = float(model.predict_proba(vec).max())

    label = "REAL" if prediction == 1 else "FAKE"

    db = SessionLocal()
    record = News(
        title=news.title,
        prediction=label,
        confidence=confidence
    )
    db.add(record)
    db.commit()
    db.close()

    return {
        "prediction": label,
        "confidence": round(confidence, 3)
    }
@app.get("/history")
def get_history():
    db = SessionLocal()
    records = db.query(News).all()
    db.close()
    return records
@app.get("/stats")
def get_stats():
    db = SessionLocal()

    total = db.query(News).count()
    fake = db.query(News).filter(News.prediction == "FAKE").count()
    real = db.query(News).filter(News.prediction == "REAL").count()
    avg_conf = db.query(func.avg(News.confidence)).scalar()

    db.close()

    return {
        "total_predictions": total,
        "fake": fake,
        "real": real,
        "fake_ratio": round(fake / total, 2) if total else 0,
        "avg_confidence": round(float(avg_conf), 3) if avg_conf else 0
    }