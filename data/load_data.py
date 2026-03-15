"""
MovieLens 100K data loader.
Downloads and preprocesses the dataset into train/test splits.
"""

import os
import zipfile
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(DATA_DIR, "ml-100k.zip")
EXTRACT_PATH = os.path.join(DATA_DIR, "ml-100k")


def download_movielens():
    if not os.path.exists(EXTRACT_PATH):
        print("Downloading MovieLens 100K...")
        r = requests.get(MOVIELENS_URL, stream=True)
        with open(ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        print("Download complete.")
    else:
        print("Dataset already exists.")


def load_ratings():
    """Returns a DataFrame with columns: user_id, item_id, rating, timestamp."""
    download_movielens()
    path = os.path.join(EXTRACT_PATH, "u.data")
    df = pd.read_csv(path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    # Zero-index users and items
    df["user_id"] = df["user_id"] - 1
    df["item_id"] = df["item_id"] - 1
    return df


def get_train_test_split(test_size=0.2, random_state=42):
    df = load_ratings()
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    n_users = df["user_id"].max() + 1
    n_items = df["item_id"].max() + 1
    return train, test, n_users, n_items


def build_interaction_matrix(df, n_users, n_items):
    """Builds a dense user-item rating matrix (0 = unobserved)."""
    R = np.zeros((n_users, n_items), dtype=np.float32)
    for row in df.itertuples(index=False):
        R[row.user_id, row.item_id] = row.rating
    return R
