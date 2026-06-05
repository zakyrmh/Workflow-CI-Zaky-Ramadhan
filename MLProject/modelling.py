import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    # Menggunakan local tracking khusus untuk environment runner CI
    mlflow.set_experiment("CI_Automated_Retraining")
    mlflow.autolog()
    
    # Membaca data yang terikat di dalam package project
    df = pd.read_csv("./namadataset_preprocessing/data_clean.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="CI_Execution"):
        # Model dilatih dengan parameter optimal hasil tuning sebelumnya
        model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
        model.fit(X_train, y_train)
        print("=== Retraining Sukses! Artefak Model Lokal Berhasil Terbentuk ===")

if __name__ == "__main__":
    main()