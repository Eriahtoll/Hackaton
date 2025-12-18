import pandas as pd
from sklearn.model_selection import train_test_split
import os
#Ce code a pour but de transformer un dataset train en dataset train (80% du dataset train) et un val (20% du dataset)
# =========================
# PARAMÈTRES
# =========================
TRAIN_INPUT = "./ressources/npyDS/train_independant.csv"
TEST_INPUT  = "./ressources/npyDS/test_independant.csv"

OUTPUT_DIR = "../ressources/npyDS/Independant"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_OUTPUT = f"{OUTPUT_DIR}/train_ind.csv"
VAL_OUTPUT   = f"{OUTPUT_DIR}/val_ind.csv"
TEST_OUTPUT  = f"{OUTPUT_DIR}/test_ind.csv"

TARGET_COL = "satisfaction"      # adapte si nécessaire
ID_COL = "id"

RANDOM_STATE = 42
TEST_SIZE = 0.2       # 20% pour la validation

# =========================
# 1. CHARGEMENT
# =========================
train_df = pd.read_csv(TRAIN_INPUT)
test_df  = pd.read_csv(TEST_INPUT)

# =========================
# 2. SPLIT TRAIN / VAL
# =========================
train_split, val_split = train_test_split(
    train_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

# =========================
# 3. NETTOYAGE TEST
# =========================
if ID_COL in test_df.columns:
    test_df = test_df.drop(columns=[ID_COL])

# =========================
# 4. SAUVEGARDE
# =========================
train_split.to_csv(TRAIN_OUTPUT, index=False)
val_split.to_csv(VAL_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)

print("✅ Fichiers générés avec succès :")
print(f" - {TRAIN_OUTPUT} ({len(train_split)} lignes)")
print(f" - {VAL_OUTPUT}   ({len(val_split)} lignes)")
print(f" - {TEST_OUTPUT}  ({len(test_df)} lignes)")