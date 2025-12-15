import pandas as pd
import re
import pickle

print("🔁 Loading model and vectorizer...")
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

print("📥 Loading dataset...")
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

print("🧹 Cleaning text...")
df["clean"] = df["text"].astype(str).apply(clean_text)

print("🔡 Vectorizing...")
X_vec = tfidf.transform(df["clean"])

print("🤖 Predicting...")
preds = model.predict(X_vec)
probas = model.predict_proba(X_vec)

# pick only rows where model predicts REAL (1)
real_rows = df[preds == 1].copy()
real_rows["real_prob"] = probas[preds == 1][:, 1]

# sort by highest REAL probability
real_rows = real_rows.sort_values("real_prob", ascending=False)

print("\n✅ Top 5 texts that your model is VERY sure are REAL:\n")
for i, row in real_rows.head(5).iterrows():
    print("=" * 80)
    print(f"REAL probability: {row['real_prob']:.4f}")
    print(row["text"][:1000])  # print first 1000 chars
    print()
