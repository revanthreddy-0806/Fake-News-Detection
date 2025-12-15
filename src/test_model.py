import pickle
import re

# load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

samples = [
    # This one should *tend* to look REAL to the model
    """The government today released official economic data showing a steady
    increase in employment over the last quarter, according to a report
    published by the Ministry of Finance.""",

    # This one should *tend* to look FAKE to the model
    """Scientists announced that a secret base was discovered on the dark side
    of the Moon where world leaders meet aliens to decide global policy,
    according to an anonymous social media post."""
]

for i, s in enumerate(samples, start=1):
    cleaned = clean(s)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    print(f"\nSample {i}:")
    print("Prediction:", "REAL (1)" if pred == 1 else "FAKE (0)")
    print("Probabilities [FAKE, REAL]:", proba)
