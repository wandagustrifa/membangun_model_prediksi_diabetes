import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from mlflow.models.signature import infer_signature
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil 
import json
import dagshub

# --- MLflow Tracking Configuration ---
# mlflow.set_tracking_uri('http://127.0.0.1:5000/')
dagshub.init(repo_owner='wandagustrifa', repo_name='diabetes-mlops-project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/wandagustrifa/diabetes-mlops-project.mlflow")

MODEL_REGISTRY_NAME="DiabetesPredictionLogisticRegression"

# --- FUNGSI UTAMA: tune_and_train_model ---
def tune_and_train_model(input_filepath):
    print(f"Memuat data yang sudah diproses dari: {input_filepath}")
    df_preprocessed = pd.read_csv(input_filepath)

    X = df_preprocessed.drop('Diagnosis', axis=1)
    y = df_preprocessed['Diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Buat contoh input untuk logging model
    input_example = X_train.sample(n=5, random_state=42)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"LogisticRegression_Tuning_{timestamp}"

    # Tentukan nama folder artefak utama untuk run ini
    # Semua artefak (model, plot, json, html) akan masuk ke sini
    model_folder_name = "model"

    with mlflow.start_run(run_name=run_name):
        mlflow.autolog(disable=True) 

        # Define the pipeline with LogisticRegression
        pipeline = Pipeline([
            ('lr', LogisticRegression())
        ])

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'lr__C': [0.001, 0.01, 0.1, 1, 10], 
            'lr__solver': ['liblinear', 'lbfgs', 'saga'], 
            'lr__max_iter': [1000] 
        }

        grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                                cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        
        print("\nMemulai Hyperparameter Tuning dengan GridSearchCV...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] 

        # Hitung metrik evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision_0 = precision_score(y_test, y_pred, pos_label=0)
        recall_0 = recall_score(y_test, y_pred, pos_label=0)
        f1_0 = f1_score(y_test, y_pred, pos_label=0)
        precision_1 = precision_score(y_test, y_pred, pos_label=1)
        recall_1 = recall_score(y_test, y_pred, pos_label=1)
        f1_1 = f1_score(y_test, y_pred, pos_label=1)
        roc_auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) != 0 else 0

        print("\nMetrik Model Terbaik (pada Test Set):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision Class 0: {precision_0:.4f}")
        print(f"Recall Class 0: {recall_0:.4f}")
        print(f"F1 Score Class 0: {f1_0:.4f}")
        print(f"Precision Class 1: {precision_1:.4f}")
        print(f"Recall Class 1: {recall_1:.4f}")
        print(f"F1 Score Class 1: {f1_1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")

        # Log parameter terbaik
        mlflow.log_params(grid_search.best_params_)

        # Log metrik secara manual 
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_class_0", precision_0)
        mlflow.log_metric("recall_class_0", recall_0)
        mlflow.log_metric("f1_score_class_0", f1_0)
        mlflow.log_metric("precision_class_1", precision_1)
        mlflow.log_metric("recall_class_1", recall_1)
        mlflow.log_metric("f1_score_class_1", f1_1)
        mlflow.log_metric("roc_auc_score", roc_auc)
        mlflow.log_metric("best_cv_roc_auc", grid_search.best_score_) 
        mlflow.log_metric("specificity", specificity) 
        mlflow.log_metric("false_positive_rate", false_positive_rate) 

        # --- ARTEFAK TAMBAHAN ---
        temp_artifacts_dir = "temp_mlflow_artifacts" 
        os.makedirs(temp_artifacts_dir, exist_ok=True)

         # Simpan Confusion Matrix sebagai gambar
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Tidak Diabetes (0)', 'Diabetes (1)'],
                    yticklabels=['Tidak Diabetes (0)', 'Diabetes (1)'])
        plt.title('KNN Confusion Matrix (Tuned)')
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        cm_path = os.path.join(temp_artifacts_dir, "training_confussion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="")
        print(f"Confusion Matrix disimpan sebagai artifak: {cm_path}")

        # Simpan kurva ROC sebagai gambar
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        roc_path = os.path.join(temp_artifacts_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path, artifact_path="")
        print(f"ROC Curve disimpan sebagai artifak: {roc_path}")

        # Simpan classification_report sebagai JSON
        report_dict = {
            "accuracy": accuracy,
            "precision_class_0": precision_0,
            "recall_class_0": recall_0,
            "f1_score_class_0": f1_0,
            "precision_class_1": precision_1,
            "recall_class_1": recall_1,
            "f1_score_class_1": f1_1,
            "roc_auc_score": roc_auc,
            "specifity": specificity,
            "false_positif_rate": false_positive_rate,
            "best_params": grid_search.best_params_
        }

        metric_info_json_path = os.path.join(temp_artifacts_dir, "metric_info.json")
        with open(metric_info_json_path, "w") as f:
            json.dump(report_dict, f, indent=4)
        mlflow.log_artifact(metric_info_json_path, artifact_path="")
        print("Detailed metrics report disimpan sebagai artifak: detailed_metrics_report.json")

        # Simpan informasi estimator terbaik dalam HTML
        estimator_html_path = os.path.join(temp_artifacts_dir, "estimator.html")
        with open(estimator_html_path, "w") as f:
            f.write("<html><body>")
            f.write("<h1>Best Estimator Parameters</h1>")
            f.write(f"<pre>{json.dumps(grid_search.best_params_, indent=4)}</pre>")
            f.write("<h1>Full Estimator String</h1>")
            f.write(f"<pre>{str(best_model)}</pre>")
            f.write("</body></html>")
        mlflow.log_artifact(estimator_html_path, artifact_path="")
        print("Best estimator info disimpan sebagai artifak: best_estimator.html")
        
        # Infer signature untuk model
        signature = infer_signature(X_train, y_pred)

        # Log Model sebagai Artefak MLflow
        mlflow.sklearn.log_model(
            best_model,
            artifact_path=model_folder_name, 
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_REGISTRY_NAME,  
            await_registration_for=300
        )
        
        # Bersihkan folder sementara setelah dilog
        shutil.rmtree(temp_artifacts_dir)
        print("File artefak sementara telah dihapus.")
        
        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"MLflow UI URL: {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    preprocessed_data_path = 'namadataset_preprocessing/preprocessed_diabetes_data.csv'
    tune_and_train_model(preprocessed_data_path)