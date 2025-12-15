# batch_predict.py
"""
Command-line batch prediction:
python batch_predict.py --input myfile.csv --output out.csv
"""
import argparse, pickle, re, pandas as pd

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-zA-Z ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--tfidf", default="tfidf.pkl")
    args = parser.parse_args()

    model = pickle.load(open(args.model, "rb"))
    tfidf = pickle.load(open(args.tfidf, "rb"))

    df = pd.read_csv(args.input)
    if "text" not in df.columns:
        raise SystemExit("Input CSV must have a 'text' column")
    df['clean'] = df['text'].apply(clean_text)
    X = tfidf.transform(df['clean'])
    preds = model.predict(X)
    df['predicted_label'] = preds
    if hasattr(model, "predict_proba"):
        df['predicted_prob'] = model.predict_proba(X).max(axis=1)
    df.to_csv(args.output, index=False)
    print("Saved predictions to", args.output)

if __name__ == "__main__":
    main()
