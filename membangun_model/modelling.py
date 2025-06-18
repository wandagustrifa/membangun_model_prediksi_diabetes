import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os
import dagshub

# mlflow.set_tracking_uri('http://127.0.0.1:5000/')
dagshub.init(repo_owner='wandagustrifa', repo_name='diabetes-mlops-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/wandagustrifa/diabetes-mlops-project.mlflow")

def train_model(input_filepath):
    """
    Memuat data yang sudah diproses, melatih model Logistic Regression,
    dan melacak metrik serta artefak menggunakan MLflow.

    Args:
        input_filepath (str): Path ke file dataset yang sudah diproses.
    """
    print(f"Memuat data yang sudah diproses dari: {input_filepath}")
    df_preprocessed = pd.read_csv(input_filepath)

    # Pisahkan fitur (X) dan target (y)
    X = df_preprocessed.drop('Diagnosis', axis=1)
    y = df_preprocessed['Diagnosis']

    # Pembagian Data Pelatihan dan Pengujian
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="LogisticRegression_Baseline"):
        mlflow.autolog(disable=True)
        # Definisikan model
        model = LogisticRegression(solver='liblinear', random_state=42)

        # Latih model
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Hitung metrik evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print("\nMetrik Model:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")

        # Log parameter
        mlflow.log_param("solver", 'liblinear')
        mlflow.log_param("random_state", 42)

        # Log metrik secara manual (sesuai kriteria Skilled)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)

        # Log model sebagai artefak MLflow
        mlflow.sklearn.log_model(
            model, 
            "logistic_regression_model",
            input_example=X_train.head(2), 
            signature=mlflow.models.infer_signature(X_train, model.predict(X_train)))

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"MLflow UI URL: {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    preprocessed_data_path = 'namadataset_preprocessing/preprocessed_diabetes_data.csv' 
    train_model(preprocessed_data_path)