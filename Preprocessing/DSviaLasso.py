import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import os

# --- Chemins ---
TRAIN_FILE_ORIG = "../ressources/train.csv"
TEST_FILE_ORIG  = "../ressources/test.csv"
OUTPUT_DIR      = "../ressources/npyDS/DataSetLasso"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Chargement des CSV ---
train_df = pd.read_csv(TRAIN_FILE_ORIG)
test_df  = pd.read_csv(TEST_FILE_ORIG)

target_column = "satisfaction"

# --- Séparer features et cible ---
drop_cols_train = ["id", "wip", "investissement", "satisfaction"]
X_train_full = train_df.drop(columns=drop_cols_train)
y_train_full = train_df[target_column]

X_test = test_df.drop(columns=["id"])

# --- Identifier colonnes continues et binaires ---
binary_cols = [col for col in X_train_full.columns if set(X_train_full[col].dropna().unique()).issubset({0,1})]
continuous_cols = [col for col in X_train_full.columns if col not in binary_cols]

# --- Normaliser colonnes continues ---
scaler = StandardScaler()
X_train_full[continuous_cols] = scaler.fit_transform(X_train_full[continuous_cols])
X_test[continuous_cols]       = scaler.transform(X_test[continuous_cols])

# --- Pré-filtrage : retirer colonnes quasi-constantes ---
selector = VarianceThreshold(threshold=0.0001)
X_train_filtered = selector.fit_transform(X_train_full)
X_test_filtered  = X_test[X_train_full.columns[selector.get_support()]]

X_train_filtered = pd.DataFrame(X_train_filtered, columns=X_train_full.columns[selector.get_support()])

print(f"Nombre de features après pré-filtrage : {X_train_filtered.shape[1]}")

# --- Lasso sur un échantillon pour réduire mémoire ---
sample_size = min(20000, X_train_filtered.shape[0])
X_sample = X_train_filtered.sample(n=sample_size, random_state=42)
y_sample = y_train_full.loc[X_sample.index]

lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=5000)
lasso.fit(X_sample, y_sample)

selected_features = X_train_filtered.columns[lasso.coef_ != 0]
print(f"Nombre de features sélectionnées après Lasso : {len(selected_features)}")

X_train_selected = X_train_filtered[selected_features]
X_test_selected  = X_test_filtered[selected_features]

# --- Split train/validation ---
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_selected, y_train_full, test_size=0.2, random_state=42
)

# --- Ajouter colonne 'y' pour correspondre à ton format ---
train_output = X_train_final.copy()
train_output["y"] = y_train_final.values

val_output = X_val_final.copy()
val_output["y"] = y_val_final.values

# --- Sauvegarde ---
train_output.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val_output.to_csv(f"{OUTPUT_DIR}/val.csv", index=False)
X_test_selected.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

print("Fichiers train/val/test enregistrés avec la colonne 'y' pour train et val.")