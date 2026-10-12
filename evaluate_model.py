import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv(
    "dataset/disease_symptoms.csv"
)

data.columns = data.columns.str.strip()

data["Disease"] = (
    data["Disease"]
    .astype(str)
    .str.strip()
)


# ==========================================
# 2. GET SYMPTOM COLUMNS
# ==========================================

symptom_columns = [
    column
    for column in data.columns
    if column.startswith("Symptom_")
]


# ==========================================
# 3. GET ALL UNIQUE SYMPTOMS
# ==========================================

all_symptoms = set()

for column in symptom_columns:

    for symptom in data[column].dropna():

        symptom = str(symptom).strip().lower()

        if symptom:
            all_symptoms.add(symptom)


all_symptoms = sorted(all_symptoms)


# ==========================================
# 4. CREATE FEATURE MATRIX
# ==========================================

X = pd.DataFrame(
    0,
    index=data.index,
    columns=all_symptoms
)


for index, row in data.iterrows():

    for column in symptom_columns:

        symptom = row[column]

        if pd.notna(symptom):

            symptom = str(symptom).strip().lower()

            if symptom in X.columns:

                X.loc[index, symptom] = 1


y = data["Disease"]


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 7. PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 8. METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n================================")
print("MODEL EVALUATION")
print("================================")

print(f"Accuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================
# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

labels = sorted(
    set(y_test) | set(y_pred)
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(
    figsize=(12, 10)
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d"
)

plt.title(
    "Medical Disease Prediction - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "model/confusion_matrix.png"
)

plt.close()

print("\nConfusion matrix saved:")
print("model/confusion_matrix.png")
