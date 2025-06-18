import requests
import json
import logging
import os
import time

# Konfigurasi logging untuk output yang lebih jelas
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Definisi URL Endpoint Model Serving ---
# URL ini adalah alamat di mana model Anda sedang di-serve.
# Pastikan model Anda sudah berjalan (misalnya, dengan `mlflow models serve` di localhost:5001).
MODEL_SERVING_URL = "http://localhost:5001/invocations"

# --- 2. Path ke File serving_input_example.json (yaitu input_example.json dari artefak MLflow) ---
# File ini secara otomatis dihasilkan oleh MLflow saat Anda melog model dengan 'input_example'.
# Anda harus mengganti path di bawah ini dengan lokasi aktual file 'input_example.json'
# yang ada di dalam folder artefak model yang Anda download.
SERVING_INPUT_EXAMPLE_PATH = "model_artifacts/best_knn_model/input_example.json"


def load_inference_payload(file_path):
    """
    Memuat payload inferensi dari file JSON.
    Data di file ini sudah dalam format yang diproses oleh preprocessor.
    """
    try:
        with open(file_path, 'r') as f:
            payload = json.load(f)
        logging.info(f"Successfully loaded inference payload from: {file_path}")
        
        # Opsional: Tampilkan beberapa detail dari payload
        if "dataframe_split" in payload and "columns" in payload["dataframe_split"] and "data" in payload["dataframe_split"]:
            logging.info(f"Payload contains {len(payload['dataframe_split']['columns'])} columns and {len(payload['dataframe_split']['data'])} row(s) of data.")
        return payload
    except FileNotFoundError:
        logging.error(f"Error: Inference example JSON not found at '{file_path}'.")
        logging.error("Pastikan Anda telah menyalin file 'input_example.json' dari artefak model MLflow Anda ke lokasi yang benar.")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from '{file_path}'. Is it a valid JSON file?")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading JSON: {e}", exc_info=True)
        return None

def send_prediction_request(payload_data):
    """
    Mengirim permintaan prediksi ke model serving endpoint.
    Payload data diasumsikan sudah dalam format yang benar (sudah diproses).
    """
    headers = {"Content-Type": "application/json"}
    
    try:
        logging.info("Sending prediction request using loaded JSON payload...")
        start_time = time.time()
        response = requests.post(MODEL_SERVING_URL, headers=headers, data=json.dumps(payload_data), timeout=10)
        end_time = time.time()
        response.raise_for_status() # Akan memunculkan HTTPError untuk status 4xx/5xx

        prediction_result = response.json()
        latency = (end_time - start_time) * 1000 # dalam milidetik
        
        logging.info(f"Prediction successful. Latency: {latency:.2f} ms")
        logging.info(f"Prediction response: {prediction_result}")
        return prediction_result

    except requests.exceptions.Timeout:
        logging.error("Prediction request timed out. Model might be overloaded or down.")
        return None
    except requests.exceptions.ConnectionError:
        logging.error("Could not connect to model serving endpoint. Please ensure the model is running at the specified URL.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Error occurred: {e}. Status Code: {e.response.status_code if e.response else 'N/A'}. Response: {getattr(e.response, 'text', 'No response text')}", exc_info=True)
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON response from model. Response text: {response.text}", exc_info=True)
        return None
    except KeyError as e:
        logging.error(f"Key error in response: Missing '{e}' in JSON. Response: {prediction_result}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # --- 3. Memuat Payload JSON ---
    # Memuat payload yang sudah diproses dari file 'input_example.json'
    inference_payload = load_inference_payload(SERVING_INPUT_EXAMPLE_PATH)

    if inference_payload:
        # --- 4. Mengirim Permintaan Prediksi ---
        send_prediction_request(inference_payload)
        logging.info("\nSingle prediction request sent successfully.")
    else:
        logging.error("Could not send prediction request as the inference payload could not be loaded.")
