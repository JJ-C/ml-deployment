import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5001")
client = MlflowClient()

try:
    exp = client.get_experiment_by_name("fraud_detection_poc")
    if exp:
        print(f"Experiment ID: {exp.experiment_id}")
        print(f"Artifact Location: {exp.artifact_location}")
    else:
        print("Experiment 'fraud_detection_poc' not found.")
except Exception as e:
    print(f"Error: {e}")
