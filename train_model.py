import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. Load disease symptom dataset
# ==========================================

DATASET_PATH = "dataset/disease_symptoms.csv"

df = pd.read_csv(DATASET_PATH, encoding="latin1")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. Clean column names
# ==========================================

df.columns = df.columns.str.strip()


# ==========================================
# 3. Separate disease and symptoms
# ==========================================

disease_column = "Disease"

symptom_columns = [
    column
    for column in df.columns
    if column != disease_column
]


# ==========================================
# 4. Create complete symptom list
# ==========================================

all_symptoms = set()

for column in symptom_columns:
    symptoms = df[column].dropna().astype(str)

    for symptom in symptoms:
        symptom = symptom.strip().lower()

        if symptom:
            all_symptoms.add(symptom)


all_symptoms = sorted(all_symptoms)

print("Total unique symptoms:", len(all_symptoms))


# ==========================================
# 5. Create feature matrix
# ==========================================

X = pd.DataFrame(0, index=df.index, columns=all_symptoms)

for index, row in df.iterrows():

    for column in symptom_columns:

        value = row[column]

        if pd.notna(value):

            symptom = str(value).strip().lower()

            if symptom in X.columns:
                X.loc[index, symptom] = 1


# ==========================================
# 6. Target variable
# ==========================================

y = df[disease_column].astype(str).str.strip()


# ==========================================
# 7. Train Random Forest
# ==========================================

print("Training model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)


# ==========================================
# 8. Save model and symptom list
# ==========================================

model_data = {
    "model": model,
    "symptoms": all_symptoms
}

joblib.dump(model_data, "model.pkl")


print("====================================")
print("Model trained successfully!")
print("Model saved as: model.pkl")
print("Diseases:", len(y.unique()))
print("Symptoms:", len(all_symptoms))
print("====================================")