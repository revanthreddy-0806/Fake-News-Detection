# interactive_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle, re, os
from io import StringIO, BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------- Utilities ----------
@st.cache_resource(show_spinner=False)
def load_model_and_vectorizer(model_path="model.pkl", tfidf_path="tfidf.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

def clean_text(text: str):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_one(model, tfidf, text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
    return pred, proba

def explain_tfidf_contrib(tfidf: TfidfVectorizer, vec):
    """
    Return top tfidf terms for a single sample vector (sparse).
    """
    # vec: sparse row matrix
    try:
        feature_array = np.array(tfidf.get_feature_names_out())
    except:
        feature_array = np.array(tfidf.get_feature_names())
    nz = vec.nonzero()
    if len(nz[1]) == 0:
        return []
    values = vec.toarray().flatten()
    nonzero_idx = np.where(values > 0)[0]
    sorted_idx = nonzero_idx[np.argsort(values[nonzero_idx])[::-1]]
    terms = [(feature_array[i], float(values[i])) for i in sorted_idx[:15]]
    return terms

# ---------- App Layout ----------
st.set_page_config(page_title="Interactive Fake News Lab", layout="wide")
st.title("🧪 Interactive Fake News Lab")

# left sidebar - load or change models/datasets
with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model file", "model.pkl")
    tfidf_path = st.text_input("TF-IDF file", "tfidf.pkl")
    data_path = st.text_input("Dataset CSV (optional)", "Fake.csv")
    st.markdown("---")
    show_top_reals = st.checkbox("Show: Top 'very real' examples (from dataset)", value=True)
    retrain_confirm = st.checkbox("Enable in-app retrain button", value=True)
    st.markdown("Files loaded from project root. Use full paths if outside folder.")
    st.markdown("---")
    st.info("This app builds on your original Streamlit UI and helpers. See original `app.py` and `find_real_examples.py` for reference.")
    # cite: original app + find_real_examples
    st.caption("Original simple app and helper scripts used: app.py, find_real_examples.py, test_model.py, train_model.py.")
    st.caption("Citations: original app and helpers are in your project. :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}")

# Try load
try:
    model, tfidf = load_model_and_vectorizer(model_path, tfidf_path)
    st.sidebar.success("Loaded model & TF-IDF")
except Exception as e:
    st.sidebar.error(f"Error loading model/vectorizer: {e}")
    st.stop()

tabs = st.tabs(["Predict (single)", "Batch CSV", "Explore dataset", "Admin / Retrain"])

# ---------- Tab: Single Prediction ----------
with tabs[0]:
    st.subheader("Single-text prediction")
    text = st.text_area("Paste article / snippet here", height=220)
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Predict"):
            if not text.strip():
                st.warning("Enter some text first.")
            else:
                pred, proba = predict_one(model, tfidf, text)
                label_str = "REAL" if pred == 1 else "FAKE"
                if proba is not None:
                    conf = proba.max()
                    st.metric("Prediction", f"{label_str} (p={conf:.3f})")
                else:
                    st.metric("Prediction", label_str)
                # show top contributing tf-idf terms
                cleaned = clean_text(text)
                vec = tfidf.transform([cleaned])
                terms = explain_tfidf_contrib(tfidf, vec)
                if terms:
                    st.markdown("**Top contributing TF-IDF terms:**")
                    for t, v in terms[:10]:
                        st.write(f"- {t} — {v:.4f}")
                # feedback mechanism
                st.markdown("---")
                st.markdown("### If prediction is wrong, correct it:")
                corrected = st.radio("Correct label for this text:", options=["Leave as is", "FAKE (0)", "REAL (1)"])
                if st.button("Submit feedback (save)"):
                    if corrected != "Leave as is":
                        label_val = 1 if corrected.endswith("(1)") else 0
                        fb = {"text": text, "label": label_val}
                        fb_df = pd.DataFrame([fb])
                        if os.path.exists("feedback.csv"):
                            fb_df.to_csv("feedback.csv", mode="a", header=False, index=False)
                        else:
                            fb_df.to_csv("feedback.csv", index=False)
                        st.success("Feedback saved to feedback.csv")

# ---------- Tab: Batch CSV ----------
with tabs[1]:
    st.subheader("Batch predict from CSV")
    st.markdown("Upload a CSV with a column named `text`. Result will add `predicted_label` and `predicted_prob`.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must contain a `text` column.")
        else:
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            if st.button("Run batch prediction"):
                cleaned = df['text'].astype(str).apply(clean_text)
                X_vec = tfidf.transform(cleaned)
                preds = model.predict(X_vec)
                probs = model.predict_proba(X_vec) if hasattr(model, "predict_proba") else None
                df['predicted_label'] = preds
                if probs is not None:
                    df['predicted_prob'] = probs.max(axis=1)
                st.success("Done — see preview here:")
                st.dataframe(df.head(20))
                # download button
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
                # optionally save to file in workspace
                if st.checkbox("Also save file to workspace as last_predictions.csv"):
                    df.to_csv("last_predictions.csv", index=False)
                    st.toast("Saved to last_predictions.csv")

# ---------- Tab: Explore dataset ----------
with tabs[2]:
    st.subheader("Dataset explorer")
    if os.path.exists(data_path):
        try:
            ds = pd.read_csv(data_path)
            st.write(f"Loaded dataset: {data_path} — {len(ds)} rows")
            if "text" not in ds.columns:
                st.error("Dataset must have a `text` column to explore.")
            else:
                st.markdown("### Class distribution (if `label` present)")
                if "label" in ds.columns:
                    counts = ds['label'].value_counts().rename({0:"FAKE",1:"REAL"})
                    chart = alt.Chart(pd.DataFrame({"label":counts.index,"count":counts.values})).mark_bar().encode(x="label", y="count")
                    st.altair_chart(chart, use_container_width=True)
                st.markdown("### Random samples")
                sample_n = st.slider("Number of random samples to show", 1, 20, 5)
                st.write(ds['text'].sample(min(sample_n, len(ds))).tolist())
                # Option: show top real examples from your helper script logic
                if show_top_reals:
                    st.markdown("### Top confident REAL examples (by your model)")
                    # re-use logic from find_real_examples.py: vectorize all cleaned and find preds==1 sorted by prob
                    df_local = ds.copy()
                    if 'label' in df_local.columns:
                        df_local = df_local[['text', 'label']]
                    df_local['clean'] = df_local['text'].astype(str).apply(clean_text)
                    X_vec = tfidf.transform(df_local['clean'])
                    preds_all = model.predict(X_vec)
                    probs_all = model.predict_proba(X_vec) if hasattr(model, "predict_proba") else None
                    real_rows = df_local[preds_all == 1].copy()
                    if probs_all is not None:
                        real_rows['real_prob'] = probs_all[preds_all == 1][:,1]
                        real_rows = real_rows.sort_values('real_prob', ascending=False)
                        st.write(real_rows[['real_prob','text']].head(10))
                    else:
                        st.write(real_rows['text'].head(10))
                    st.caption("This section uses the same idea as your find_real_examples.py helper. :contentReference[oaicite:9]{index=9}")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    else:
        st.info(f"No dataset found at {data_path}. Put your CSV in project root or set the path in sidebar.")

# ---------- Tab: Admin / Retrain ----------
with tabs[3]:
    st.subheader("Admin, feedback & retrain")
    # show feedback
    if os.path.exists("feedback.csv"):
        fb = pd.read_csv("feedback.csv")
        st.write("Saved feedback (corrections):", fb.tail(20))
        if st.button("Clear feedback file (delete)"):
            os.remove("feedback.csv")
            st.success("feedback.csv removed")
    else:
        st.info("No feedback.csv found (users can submit corrections from Predict tab).")

    st.markdown("---")
    st.write("Retrain the model using your dataset + feedback (this will overwrite model.pkl & tfidf.pkl).")
    st.caption("This retrain behavior is intentionally similar to your `train_model.py` script but runs inside the app (small datasets only). :contentReference[oaicite:10]{index=10}")
    retrain_btn = st.button("Retrain model now (use dataset + feedback)")
    if retrain_btn:
        # load dataset
        if not os.path.exists(data_path):
            st.error("Dataset file not found: cannot retrain.")
        else:
            ds = pd.read_csv(data_path)
            if "text" not in ds.columns or "label" not in ds.columns:
                st.error("Dataset must have `text` and `label` columns to retrain.")
            else:
                st.info("Preparing training data...")
                # merge feedback
                if os.path.exists("feedback.csv"):
                    fb = pd.read_csv("feedback.csv")
                    fb = fb[['text','label']]
                    ds_merge = pd.concat([ds[['text','label']], fb], ignore_index=True)
                else:
                    ds_merge = ds[['text','label']]
                ds_merge['clean'] = ds_merge['text'].astype(str).apply(clean_text)
                # simple train/test split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(ds_merge['clean'], ds_merge['label'], test_size=0.2, random_state=42, stratify=ds_merge['label'])
                st.info("Vectorizing & fitting TF-IDF + LogisticRegression (this may take a few seconds)...")
                tfidf_new = TfidfVectorizer(stop_words='english', max_features=5000)
                X_train_vec = tfidf_new.fit_transform(X_train)
                X_test_vec = tfidf_new.transform(X_test)
                from sklearn.linear_model import LogisticRegression
                model_new = LogisticRegression(max_iter=300)
                model_new.fit(X_train_vec, y_train)
                preds = model_new.predict(X_test_vec)
                st.write("Retrain Test accuracy:", accuracy_score(y_test, preds))
                st.text("Classification report:\n" + classification_report(y_test, preds))
                # save model
                pickle.dump(model_new, open("model.pkl","wb"))
                pickle.dump(tfidf_new, open("tfidf.pkl","wb"))
                st.success("New model saved to model.pkl and tfidf.pkl — reload app to use it.")
                st.balloons()

st.markdown("---")
st.caption("Original lightweight `streamlit` app that this replaces: app.py. :contentReference[oaicite:11]{index=11}")
# streamlit run interactive_app.py

