import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import os

# -------------------------
# Chemins
# -------------------------
TRAIN_FILE_ORIG = "../ressources/train.csv"
TEST_FILE_ORIG  = "../ressources/test.csv"
OUTPUT_DIR      = "../ressources/npyDS/DataSetCluster"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Chargement
# -------------------------
train_df = pd.read_csv(TRAIN_FILE_ORIG)
test_df  = pd.read_csv(TEST_FILE_ORIG)

target_column = "satisfaction"

drop_cols_train = ["id", "wip", "investissement", "satisfaction"]
X_train_full = train_df.drop(columns=drop_cols_train)
y_train_full = train_df[target_column]

X_test = test_df.drop(columns=["id"])

# -------------------------
# Types de colonnes
# -------------------------
binary_cols = [c for c in X_train_full.columns if set(X_train_full[c].dropna().unique()).issubset({0,1})]
continuous_cols = [c for c in X_train_full.columns if c not in binary_cols]

# -------------------------
# Normalisation
# -------------------------
scaler = StandardScaler()
X_train_full[continuous_cols] = scaler.fit_transform(X_train_full[continuous_cols])
X_test[continuous_cols]       = scaler.transform(X_test[continuous_cols])

# -------------------------
# Nettoyage rapide
# -------------------------
selector = VarianceThreshold(threshold=0.0001)
X_train_f = selector.fit_transform(X_train_full)
kept_cols = X_train_full.columns[selector.get_support()]

X_train_f = pd.DataFrame(X_train_f, columns=kept_cols)
X_test_f  = X_test[kept_cols]

print(f"[INFO] Features après VarianceThreshold : {X_train_f.shape[1]}")

# -------------------------
# Lasso (sur échantillon)
# -------------------------
sample_size = min(20000, len(X_train_f))
X_sample = X_train_f.sample(n=sample_size, random_state=42)
y_sample = y_train_full.loc[X_sample.index]

lasso = LassoCV(cv=5, n_jobs=-1, max_iter=5000, random_state=42)
lasso.fit(X_sample, y_sample)

selected_features = X_train_f.columns[lasso.coef_ != 0]
print(f"[INFO] Features après Lasso : {len(selected_features)}")

X_train_sel = X_train_f[selected_features].copy()
X_test_sel  = X_test_f[selected_features].copy()

# =========================================================
# 🔥 FEATURE ENGINEERING (APRES REDUCTION)
# =========================================================

# --- 1. Compteurs globaux ---
X_train_sel["nb_ones"] = X_train_sel.sum(axis=1)
X_test_sel["nb_ones"]  = X_test_sel.sum(axis=1)

X_train_sel["ratio_ones"] = X_train_sel["nb_ones"] / X_train_sel.shape[1]
X_test_sel["ratio_ones"]  = X_test_sel["nb_ones"] / X_test_sel.shape[1]

# --- 2. Rareté ---
freq = X_train_sel.mean(axis=0)
rare_cols   = freq[freq < 0.05].index
common_cols = freq[freq > 0.5].index

X_train_sel["rare_activations"]   = X_train_sel[rare_cols].sum(axis=1)
X_train_sel["common_activations"] = X_train_sel[common_cols].sum(axis=1)

X_test_sel["rare_activations"]   = X_test_sel[rare_cols].sum(axis=1)
X_test_sel["common_activations"] = X_test_sel[common_cols].sum(axis=1)

# --- 3. Clustering de features ---
corr = X_train_sel.corr().abs().fillna(0)
n_clusters = min(10, corr.shape[0] // 5)

clusterer = AgglomerativeClustering(
    n_clusters=n_clusters,
    metric="precomputed",
    linkage="average"
)

labels = clusterer.fit_predict(1 - corr)

for i in range(n_clusters):
    cols = corr.columns[labels == i]
    X_train_sel[f"cluster_{i}_sum"] = X_train_sel[cols].sum(axis=1)
    X_test_sel[f"cluster_{i}_sum"]  = X_test_sel[cols].sum(axis=1)

# --- 4. PCA léger ---
pca = PCA(n_components=5, random_state=42)
X_train_pca = pca.fit_transform(X_train_sel)
X_test_pca  = pca.transform(X_test_sel)

for i in range(5):
    X_train_sel[f"pca_{i}"] = X_train_pca[:, i]
    X_test_sel[f"pca_{i}"]  = X_test_pca[:, i]

print(f"[INFO] Features finales : {X_train_sel.shape[1]}")

# -------------------------
# Split train / val
# -------------------------
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_sel, y_train_full, test_size=0.2, random_state=42
)

train_output = X_train_final.copy()
train_output["y"] = y_train_final.values

val_output = X_val_final.copy()
val_output["y"] = y_val_final.values

# -------------------------
# Sauvegarde
# -------------------------
train_output.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val_output.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
X_test_sel.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

print("✅ Datasets enrichis générés avec succès")