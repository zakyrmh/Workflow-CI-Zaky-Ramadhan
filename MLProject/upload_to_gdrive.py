import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def main():
    print("=== Memulai Sinkronisasi Artefak ke Google Drive ===")
    
    # 1. Mengambil variabel lingkungan dari GitHub Actions
    creds_raw = os.environ.get("GDRIVE_CREDENTIALS")
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    run_id = os.environ.get("LATEST_RUN_ID")
    
    if not creds_raw or not folder_id or not run_id:
        print("❌ Kesalahan: Variabel GDRIVE_CREDENTIALS, GDRIVE_FOLDER_ID, atau LATEST_RUN_ID tidak ditemukan.")
        return

    # 2. Menentukan jalur fisik file model.pkl hasil retraining otomatis di runner
    # Menyesuaikan dengan pohon direktori mlruns lokal Anda
    model_source_path = f"MLProject/mlruns/0/{run_id}/artifacts/model/model.pkl"
    
    if not os.path.exists(model_source_path):
        print(f"❌ Gagal: File model.pkl tidak ditemukan di jalur {model_source_path}")
        return

    print(f"Membaca berkas model dari: {model_source_path}")
    
    # 3. Autentikasi menggunakan kredensial Google Service Account
    creds_dict = json.loads(creds_raw)
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    service = build('drive', 'v3', credentials=credentials)
    
    # 4. Mengonfigurasi metadata file untuk diunggah ke folder target
    file_metadata = {
        'name': f'model_aqsoldb_run_{run_id}.pkl',
        'parents': [folder_id]
    }
    
    media = MediaFileUpload(model_source_path, mimetype='application/octet-stream', resumable=True)
    
    # 5. Eksekusi unggah file ke Google Drive
    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    
    print(f"✅ Sukses! Berkas model berhasil terunggah ke Google Drive dengan ID: {uploaded_file.get('id')}")

if __name__ == "__main__":
    main()