import pandas as pd
import os
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Add the project root to path so we can import preprocess
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
from preprocess import load_and_preprocess


def vectorize(
    df: pd.DataFrame,
    method: str = 'tfidf',
    max_features: int = 5000,
    save_dir: str = None
):
    """
    Convert the 'cleaned_text' column into a numerical feature matrix.

    Parameters
    ----------
    df          : DataFrame with 'cleaned_text' and 'label_num' columns
    method      : 'tfidf'  → TfidfVectorizer  (recommended, default)
                  'count'  → CountVectorizer
    max_features: vocabulary size cap (top N words by frequency)
    save_dir    : if provided, saves X (sparse .npz) and y (.csv) here

    Returns
    -------
    X           : sparse matrix  (n_samples × max_features)
    y           : pandas Series  (n_samples,) with 0/1 labels
    vectorizer  : fitted vectorizer object (reuse for inference)
    """
    if 'cleaned_text' not in df.columns:
        raise ValueError("DataFrame must have a 'cleaned_text' column. Run preprocess first.")

    # ── Choose vectorizer ─────────────────────────────────────────────────────
    method = method.lower().strip()
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
        print(f"Using TF-IDF vectorizer (max_features={max_features})")
    elif method == 'count':
        vectorizer = CountVectorizer(max_features=max_features)
        print(f"Using CountVectorizer (max_features={max_features})")
    else:
        raise ValueError(f"method must be 'tfidf' or 'count', got '{method}'")

    # ── Fit & transform ───────────────────────────────────────────────────────
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label_num'].reset_index(drop=True)

    print(f"Feature matrix shape : {X.shape}")
    print(f"Label distribution   : spam={int((y==1).sum())}  ham={int((y==0).sum())}")

    # ── Optionally save ───────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        sp.save_npz(os.path.join(save_dir, 'X_features.npz'), X)
        y.to_csv(os.path.join(save_dir, 'y_labels.csv'), index=False)
        print(f"Features saved to   : {save_dir}")

    return X, y, vectorizer


# ── Quick smoke-test when run directly ───────────────────────────────────────
if __name__ == '__main__':
    INPUT     = 'data/input/spam_clean.csv'       # or 'spam ham dataset.csv'
    SAVE_DIR  = 'data/processed/'

    # Step 1 – preprocess
    df = load_and_preprocess(INPUT)

    # Step 2 – vectorize  (swap method='count' to use CountVectorizer instead)
    X, y, vectorizer = vectorize(df, method='tfidf', max_features=5000, save_dir=SAVE_DIR)

    # Preview top 10 feature names
    feature_names = vectorizer.get_feature_names_out()
    print(f"\nSample features: {feature_names[:10]}")
