import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    # Aktifkan autolog otomatis.
    # Karena skrip ini dipanggil via 'mlflow run', MLflow sudah menyediakan 
    # konteks Run aktif di latar belakang secara otomatis.
    mlflow.sklearn.autolog()
    
    # Membaca data yang terikat di dalam package project
    df = pd.read_csv("./namadataset_preprocessing/data_clean.csv")
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Proses training langsung dieksekusi (autolog akan otomatis menempel pada Run aktif)
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    print("=== Retraining Sukses! Artefak Model Lokal Berhasil Terbentuk ===")

if __name__ == "__main__":
    main()