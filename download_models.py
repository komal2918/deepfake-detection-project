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
        # First attempt: Direct download
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Check if we got redirected to Google Drive's virus scan warning
        if 'accounts.google.com' in response.url or 'drive.google.com' in response.url:
            # Extract file ID from original URL
            file_id = url.split('id=')[1] if 'id=' in url else None
            if file_id:
                # Try alternative download URL that bypasses virus scan
                alt_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                response = session.get(alt_url, stream=True)
        
        response.raise_for_status()
        
        # Check if we're still getting HTML instead of binary data
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            st.error(f"❌ Google Drive returned HTML instead of model file. Try re-sharing the file or use a different cloud storage.")
            return None
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify the downloaded file is valid (should be larger than 1MB for model files)
        if model_path.stat().st_size < 1024*1024:  # Less than 1MB
            st.error(f"❌ Downloaded file {filename} seems too small. Please check the Google Drive link.")
            model_path.unlink()  # Delete the invalid file
            return None
        
        st.success(f"✅ {filename} downloaded successfully!")
        return str(model_path)
        
    except Exception as e:
        st.error(f"❌ Error downloading {filename}: {str(e)}")
        if model_path.exists():
            model_path.unlink()  # Clean up partial download
        return None

# Model URLs - Alternative Google Drive format to bypass virus scan
MODEL_URLS = {
    'FINAL_VisionTransformer_99.92acc_20250613_114100.pth': 'https://drive.google.com/u/0/uc?id=1bi39ctJ-Yv2XGfr6eN7PRGwUbGGmjjg5&export=download&confirm=t',
    'FINAL_Xception_99.26acc_20250614_021320.pth': 'https://drive.google.com/u/0/uc?id=1f1ynbqJSqWaszLz42xi0rvGgG_k8YKFX&export=download&confirm=t',
    'FINAL_EfficientNet_B3_99.11acc_20250612_121115.pth': 'https://drive.google.com/u/0/uc?id=1DpXYknk-7DDK5-v4ZKcde8uVFrI0VuEJ&export=download&confirm=t'
}

def ensure_models_downloaded():
    """Ensure all models are downloaded"""
    for filename, url in MODEL_URLS.items():
        if url != 'YOUR_GOOGLE_DRIVE_LINK_HERE':
            download_model(url, filename) 