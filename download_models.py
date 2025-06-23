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
                if chunk:
                    f.write(chunk)
        
        # Verify the downloaded file is valid (should be larger than 1MB for model files)
        if model_path.stat().st_size < 1024*1024:  # Less than 1MB
            st.error(f"❌ Downloaded file {filename} seems too small. Please check the download URL.")
            model_path.unlink()  # Delete the invalid file
            return None
        
        st.success(f"✅ {filename} downloaded successfully!")
        return str(model_path)
        
    except Exception as e:
        st.error(f"❌ Error downloading {filename}: {str(e)}")
        if model_path.exists():
            model_path.unlink()  # Clean up partial download
        return None

# Model URLs - GitHub Releases (reliable for large files)
MODEL_URLS = {
    'FINAL_VisionTransformer_99.92acc_20250613_114100.pth': 'https://github.com/komal2918/deepfake-detection-project/releases/download/v1.0.0/FINAL_VisionTransformer_99.92acc_20250613_114100.pth',
    'FINAL_Xception_99.26acc_20250614_021320.pth': 'https://github.com/komal2918/deepfake-detection-project/releases/download/v1.0.0/FINAL_Xception_99.26acc_20250614_021320.pth',
    'FINAL_EfficientNet_B3_99.11acc_20250612_121115.pth': 'https://github.com/komal2918/deepfake-detection-project/releases/download/v1.0.0/FINAL_EfficientNet_B3_99.11acc_20250612_121115.pth'
}

def ensure_models_downloaded():
    """Ensure all models are downloaded"""
    for filename, url in MODEL_URLS.items():
        if url != 'YOUR_GOOGLE_DRIVE_LINK_HERE':
            download_model(url, filename) 