import pandas as pd
import re
import os

def preprocess_text(text: str) -> str:
    """
    Clean a single email string:
      1. Lowercase
      2. Remove punctuation and special characters (keep alphanumeric + spaces)
    """
    if not isinstance(text, str):
        text = str(text)
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation / special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # 3. Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Load a CSV dataset, apply text preprocessing, and optionally save the result.

    Supports both dataset formats found in this project:
      - spam_clean.csv          → columns: text, label
      - spam ham dataset.csv    → columns: (index), label, text, label_num
    
    Returns a DataFrame with columns: text, label, label_num, cleaned_text
    """
    df = pd.read_csv(input_path)

    # ── Normalise column names ────────────────────────────────────────────────
    # Drop unnamed index column if present (from 'spam ham dataset.csv')
    unnamed = [c for c in df.columns if c.lower().startswith('unnamed')]
    if unnamed:
        df.drop(columns=unnamed, inplace=True)

    # Ensure we have a 'text' and a numeric label column
    if 'text' not in df.columns:
        raise ValueError(f"Expected a 'text' column. Found: {list(df.columns)}")

    # Build label_num if missing (spam_clean.csv only has 'label' as 0/1)
    if 'label_num' not in df.columns:
        if 'label' in df.columns:
            df['label_num'] = pd.to_numeric(df['label'], errors='coerce')
        else:
            raise ValueError("No label column found.")

    # Build text 'label' (ham/spam) if missing
    if df['label'].dtype == object and df['label'].str.lower().isin(['ham', 'spam']).any():
        pass  # already string labels
    else:
        df['label'] = df['label_num'].map({0: 'ham', 1: 'spam'})

    # ── Apply preprocessing ───────────────────────────────────────────────────
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # ── Optionally persist ────────────────────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to: {output_path}")

    print(f"Loaded {len(df)} rows | spam: {(df['label_num']==1).sum()} | ham: {(df['label_num']==0).sum()}")
    return df


# ── Quick smoke-test when run directly ───────────────────────────────────────
if __name__ == '__main__':
    # Adjust paths to match your repo layout
    INPUT  = 'data/input/spam_clean.csv'          # or 'spam ham dataset.csv'
    OUTPUT = 'data/processed/spam_preprocessed.csv'

    df = load_and_preprocess(INPUT, OUTPUT)
    print(df[['text', 'cleaned_text', 'label', 'label_num']].head(3).to_string())
