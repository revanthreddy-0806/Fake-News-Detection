# feedback_utils.py
import pandas as pd
import os

FEEDBACK_FILE = "feedback.csv"

def save_feedback(text, label):
    row = {"text": text, "label": int(label)}
    df = pd.DataFrame([row])
    if os.path.exists(FEEDBACK_FILE):
        df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(FEEDBACK_FILE, index=False)

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    return pd.DataFrame(columns=["text","label"])
