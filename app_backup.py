from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import re
import json
from datetime import datetime

# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = os.path.join(
    "model",
    "medical_model.pkl"
)
HISTORY_FILE = "prediction_history.json"

SYMPTOM_DATASET = os.path.join(
    "dataset",
    "disease_symptoms.csv"
)

PRECAUTION_DATASET = os.path.join(
    "dataset",
    "disease_precaution.csv"
)

RISK_DATASET = os.path.join(
    "dataset",
    "disease_riskFactors.csv"
)

MEDICINE_DATASET = os.path.join(
    "dataset",
    "disease_medicine.csv"
)


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================



model_data = joblib.load(MODEL_PATH)

if isinstance(model_data, dict):
    model = model_data["model"]
else:
    model = model_data


# =========================================================
# LOAD DATASETS
# =========================================================

try:

    symptom_df = pd.read_csv(
        SYMPTOM_DATASET,
        encoding="latin1"
    )

    precaution_df = pd.read_csv(
        PRECAUTION_DATASET,
        encoding="latin1"
    )

    risk_df = pd.read_csv(
        RISK_DATASET,
        encoding="latin1"
    )

    medicine_df = pd.read_csv(
        MEDICINE_DATASET,
        encoding="latin1"
    )

    print("Datasets loaded successfully!")

except Exception as e:

    print("ERROR: Could not load datasets.")
    print(e)

    symptom_df = pd.DataFrame()
    precaution_df = pd.DataFrame()
    risk_df = pd.DataFrame()
    medicine_df = pd.DataFrame()


# =========================================================
# NORMALIZE DISEASE NAME
# =========================================================

def normalize_disease_name(name):

    if pd.isna(name):

        return ""

    name = str(name).strip().lower()

    name = name.replace("_", " ")

    name = name.replace("-", " ")

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name.strip()


# =========================================================
# CREATE NORMALIZED DISEASE COLUMNS
# =========================================================

if not precaution_df.empty:

    precaution_df["Disease_Normalized"] = (
        precaution_df["Disease"]
        .apply(normalize_disease_name)
    )


if not risk_df.empty:

    risk_df["DNAME_Normalized"] = (
        risk_df["DNAME"]
        .apply(normalize_disease_name)
    )


# =========================================================
# GET SYMPTOMS
# =========================================================

def get_symptoms():

    symptoms = []

    if not symptom_df.empty:

        for column in symptom_df.columns:

            if column.lower().startswith("symptom"):

                values = symptom_df[column].dropna()

                for value in values:

                    value = str(value).strip()

                    if value:

                        symptoms.append(value)

    # Remove duplicates
    symptoms = list(
        dict.fromkeys(symptoms)
    )

    symptoms.sort()

    return symptoms


# =========================================================
# HOME PAGE
# =========================================================
def save_prediction_history(
    selected_symptoms,
    prediction,
    confidence
):

    history = []

    if os.path.exists(HISTORY_FILE):

        try:

            with open(
                HISTORY_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                history = json.load(file)

        except (
            json.JSONDecodeError,
            OSError
        ):

            history = []


    new_record = {
        "date_time": datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        ),
        "symptoms": selected_symptoms,
        "prediction": prediction,
        "confidence": confidence
    }


    history.append(new_record)


    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )
@app.route("/")
def home():

    symptoms = get_symptoms()

    return render_template(
        "index.html",
        symptoms=symptoms
    )


@app.route("/history")
def history():

    history_data = []

    if os.path.exists(HISTORY_FILE):

        try:

            with open(
                HISTORY_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                history_data = json.load(file)

        except (
            json.JSONDecodeError,
            OSError
        ):

            history_data = []

    history_data.reverse()

    return render_template(
        "history.html",
        history=history_data
    )
@app.route("/dashboard")
def dashboard():

    history_data = []

    if os.path.exists(HISTORY_FILE):

        try:
            with open(
                HISTORY_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                history_data = json.load(file)

        except (
            json.JSONDecodeError,
            OSError
        ):

            history_data = []

    total_predictions = len(history_data)

    latest_prediction = "No predictions yet"

    most_predicted = "No predictions yet"

    prediction_counts = {}

    if history_data:

        latest_prediction = history_data[-1].get(
            "prediction",
            "Unknown"
        )

        for record in history_data:

            disease = record.get(
                "prediction",
                "Unknown"
            )

            prediction_counts[disease] = (
                prediction_counts.get(disease, 0) + 1
            )

        most_predicted = max(
            prediction_counts,
            key=prediction_counts.get
        )

    return render_template(
    "dashboard.html",
    total_predictions=total_predictions,
    latest_prediction=latest_prediction,
    most_predicted=most_predicted,
    prediction_counts=prediction_counts,
    recent_predictions=history_data[-5:]
)
# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if model is None:

        return """
        <h2>ML model could not be loaded.</h2>
        <p>Please check model/medical_model.pkl</p>
        """


    # -----------------------------------------------------
    # Get symptoms
    # -----------------------------------------------------

    symptoms = get_symptoms()

    selected_symptoms = []

    for symptom in symptoms:

        if request.form.get(symptom):

            selected_symptoms.append(symptom)


    # -----------------------------------------------------
    # Check whether symptoms were selected
    # -----------------------------------------------------

    if len(selected_symptoms) == 0:

        return render_template(
            "result.html",
            prediction=None,
            confidence=None,
            top_predictions=[],
            selected_symptoms=[],
            precautions=[],
            risk_factors=[],
            medicines=[]
        )


    # -----------------------------------------------------
    # Create input dataframe
    # -----------------------------------------------------

    input_data = {}

    for symptom in symptoms:

        if symptom in selected_symptoms:

            input_data[symptom] = 1

        else:

            input_data[symptom] = 0


    input_df = pd.DataFrame(
        [input_data]
    )


    # -----------------------------------------------------
    # Make sure feature order matches model
    # -----------------------------------------------------

    try:

        if hasattr(
            model,
            "feature_names_in_"
        ):

            model_features = list(
                model.feature_names_in_
            )

            input_df = input_df.reindex(
                columns=model_features,
                fill_value=0
            )

    except Exception as e:

        print(
            "Feature alignment warning:",
            e
        )


    # =====================================================
    # MACHINE LEARNING PREDICTION
    # =====================================================

    try:

        prediction = model.predict(
            input_df
        )[0]

        prediction = str(
            prediction
        ).strip()

    except Exception as e:

        return f"""
        <h2>Prediction Error</h2>
        <p>{e}</p>
        """


    # =====================================================
    # TOP 3 PREDICTIONS
    # =====================================================

    top_predictions = []

    confidence = None


    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = model.predict_proba(
                input_df
            )[0]

            classes = model.classes_

            top_indexes = probabilities.argsort()[
                -3:
            ][::-1]

            for index in top_indexes:

                disease_name = str(
                    classes[index]
                ).strip()

                probability = round(
                    float(
                        probabilities[index]
                    ) * 100,
                    2
                )

                top_predictions.append(
                    {
                        "disease": disease_name,
                        "probability": probability
                    }
                )

            if top_predictions:

                confidence = top_predictions[0][
                    "probability"
                ]

        except Exception as e:

            print(
                "Probability error:",
                e
            )

            top_predictions = [
                {
                    "disease": prediction,
                    "probability": None
                }
            ]

    else:

        top_predictions = [
            {
                "disease": prediction,
                "probability": None
            }
        ]


    # =====================================================
    # NORMALIZED PREDICTION NAME
    # =====================================================

    normalized_prediction = (
        normalize_disease_name(prediction)
        if prediction is not None and str(prediction).strip()
        else ""
    )

    # =====================================================
    # PRECAUTIONS
    # =====================================================

    precautions = []

    if not precaution_df.empty:

        try:

            precaution_row = precaution_df[
                precaution_df[
                    "Disease_Normalized"
                ]==normalized_prediction
            ]

            if not precaution_row.empty:

                row = precaution_row.iloc[0]

                for column in [
                    "Precaution_1",
                    "Precaution_2",
                    "Precaution_3",
                    "Precaution_4"
                ]:

                    if column in row.index:

                        value = row[column]

                        if pd.notna(value):

                            value = str(
                                value
                            ).strip()

                            if value:

                                precautions.append(
                                    value
                                )

        except Exception as e:

            print(
                "Precaution error:",
                e
            )


    # =====================================================
    # RISK FACTORS
    # =====================================================

    risk_factors = []

    if not risk_df.empty:

        try:

            risk_row = risk_df[
                risk_df[
                    "DNAME_Normalized"
                ]== normalized_prediction
            ]

            if not risk_row.empty:

                row = risk_row.iloc[0]

                if "RISKFAC" in row.index:

                    value = row["RISKFAC"]

                    if pd.notna(value):

                        value = str(
                            value
                        ).strip()

                        if value:

                            risk_factors.append(
                                value
                            )

        except Exception as e:

            print(
                "Risk factor error:",
                e
            )


    # =====================================================
    # MEDICINE INFORMATION
    # =====================================================

    medicines = []

    if not medicine_df.empty:

        try:

            # Try to find the disease ID
            # using the risk-factor dataset.

            disease_id = None

            if not risk_df.empty:

                disease_row = risk_df[
                    risk_df[
                        "DNAME_Normalized"
                    ] == normalized_prediction
                ]

                if not disease_row.empty:

                    disease_id = (
                        disease_row.iloc[0]["DID"]
                    )


            # Match medicines using Disease_ID

            if disease_id is not None:

                medicine_rows = medicine_df[
                    medicine_df["Disease_ID"]
                    == disease_id
                ]

            else:

                medicine_rows = pd.DataFrame()


            # Extract medicine information

            if not medicine_rows.empty:

                for _, row in medicine_rows.iterrows():

                    medicine = {}

                    if "Medicine_Name" in row.index:

                        medicine[
                            "name"
                        ] = str(
                            row["Medicine_Name"]
                        ).strip()

                    else:

                        medicine[
                            "name"
                        ] = "Unknown"


                    if "Medicine_Composition" in row.index:

                        medicine[
                            "composition"
                        ] = str(
                            row[
                                "Medicine_Composition"
                            ]
                        ).strip()

                    else:

                        medicine[
                            "composition"
                        ] = ""


                    if "Medicine_Description" in row.index:

                        medicine[
                            "description"
                        ] = str(
                            row[
                                "Medicine_Description"
                            ]
                        ).strip()

                    else:

                        medicine[
                            "description"
                        ] = ""


                    medicines.append(
                        medicine
                    )

        except Exception as e:

            print(
                "Medicine error:",
                e
            )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================
    save_prediction_history(
        selected_symptoms,
        prediction,
        confidence
    )
    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        top_predictions=top_predictions,
        selected_symptoms=selected_symptoms,
        precautions=precautions,
        risk_factors=risk_factors,
        medicines=medicines
    )


# =========================================================
# RUN APPLICATION 
# =========================================================
if __name__ == "__main__":
    print("Starting AI Medical Diagnosis System...")
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )