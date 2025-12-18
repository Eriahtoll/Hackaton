# =========================
# PREPROCESSING OFFLINE
# =========================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib

# -------------------------
# Paramètres
# -------------------------
N_SAMPLES = None
PCA_DIM = 512          # 🔴 1024 est inutile, voir plus bas
OUT_DIR = "../ressources/npyDS/"

# -------------------------
# Chargement
# -------------------------
df_train = pd.read_csv("../ressources/train.csv")
df_test = pd.read_csv("../ressources/test.csv")

# -------------------------
# Nettoyage
# -------------------------
def preprocess(df, drop_columns=None, n_samples=None):
    colonnes_constantes = df.columns[df.nunique(dropna=False) == 1]
    if drop_columns is None:
        drop_columns = []
    cols_to_drop = list(colonnes_constantes) + drop_columns
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")

    if n_samples:
        df_clean = df_clean.iloc[
            np.linspace(0, len(df_clean)-1, n_samples, dtype=int)
        ]

    return df_clean

df_train_clean = preprocess(df_train, drop_columns=["id"], n_samples=N_SAMPLES)
df_test_clean  = preprocess(df_test, drop_columns=["id"])

# -------------------------
# Split X / y
# -------------------------
y = df_train_clean["satisfaction"]
X = df_train_clean.drop(columns=["wip", "investissement", "satisfaction"])

# -------------------------
# Train / Val split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Feature engineering
# -------------------------
selector = VarianceThreshold(threshold=0.01)
scaler   = StandardScaler()
pca      = PCA(n_components=PCA_DIM, random_state=42)

X_train = selector.fit_transform(X_train)
X_val   = selector.transform(X_val)
X_test  = selector.transform(df_test_clean[X.columns])

X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

X_train = pca.fit_transform(X_train)
X_val   = pca.transform(X_val)
X_test  = pca.transform(X_test)

print("Final shapes:", X_train.shape, X_val.shape, X_test.shape)

# -------------------------
# Sauvegarde CSV
# -------------------------
pd.DataFrame(
    np.c_[X_train, y_train.to_numpy()],
    columns=[f"f{i}" for i in range(PCA_DIM)] + ["y"]
).to_csv(OUT_DIR + "train512.csv", index=False)

pd.DataFrame(
    np.c_[X_val, y_val.to_numpy()],
    columns=[f"f{i}" for i in range(PCA_DIM)] + ["y"]
).to_csv(OUT_DIR + "val512.csv", index=False)

pd.DataFrame(
    X_test,
    columns=[f"f{i}" for i in range(PCA_DIM)]
).to_csv(OUT_DIR + "test512.csv", index=False)

# -------------------------
# Sauvegarde des objets
# -------------------------
joblib.dump(selector, OUT_DIR + "variance_selector.joblib")
joblib.dump(scaler,   OUT_DIR + "scaler.joblib")
joblib.dump(pca,      OUT_DIR + "pca.joblib")

print("✅ Preprocessing terminé et sauvegardé")