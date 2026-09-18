# train.py
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

os.makedirs("model", exist_ok=True)

def load_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines

real = load_lines("dataset/real.txt")
fake = load_lines("dataset/fake.txt")

if len(real) == 0 or len(fake) == 0:
    raise SystemExit("ERROR: dataset/real.txt or dataset/fake.txt is empty. Add lines and try again.")

messages = real + fake
labels = [0] * len(real) + [1] * len(fake)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print(f"Model trained on {len(messages)} messages. Saved to model/")
