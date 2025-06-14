import os
import requests
from pathlib import Path
import streamlit as st

def download_model(url, filename):
    """Download model from URL if not exists"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / filename
    
    if model_path.exists():
        return str(model_path)
    
    st.info(f"Downloading {filename}... This may take a few minutes.")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        st.success(f"✅ {filename} downloaded successfully!")
        return str(model_path)
        
    except Exception as e:
        st.error(f"❌ Error downloading {filename}: {str(e)}")
        return None

# Model URLs (you would upload your models to Google Drive/Dropbox/etc.)
MODEL_URLS = {
    'FINAL_VisionTransformer_99.92acc_20250613_114100.pth': 'YOUR_GOOGLE_DRIVE_LINK_HERE',
    'FINAL_Xception_99.26acc_20250614_021320.pth': 'YOUR_GOOGLE_DRIVE_LINK_HERE',
    'FINAL_EfficientNet_B3_99.11acc_20250612_121115.pth': 'YOUR_GOOGLE_DRIVE_LINK_HERE'
}

def ensure_models_downloaded():
    """Ensure all models are downloaded"""
    for filename, url in MODEL_URLS.items():
        if url != 'YOUR_GOOGLE_DRIVE_LINK_HERE':
            download_model(url, filename) 