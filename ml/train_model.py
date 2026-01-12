import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

fake = pd.read_csv("../data/Fake.csv")
real = pd.read_csv("../data/True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real]).sample(frac=1)

def clean(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

df["text"] = (df["title"] + " " + df["text"]).apply(clean)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1,2),
    max_features=8000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print(classification_report(y_test, model.predict(X_test_vec)))

joblib.dump(model, "../model.pkl")
joblib.dump(vectorizer, "../vectorizer.pkl")

print("Improved model saved!")
