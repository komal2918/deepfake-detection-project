#!/usr/bin/env python3
"""
Deployment helper script for Deepfake Detection Hub
Helps prepare the application for various deployment platforms
"""

import os
import shutil
import json
import zipfile
from pathlib import Path

def create_deployment_files():
    """Create necessary deployment files for different platforms"""
    
    # Create Procfile for Heroku
    procfile_content = "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
    with open("Procfile", "w") as f:
        f.write(procfile_content)
    
    # Create railway.json for Railway
    railway_config = {
        "build": {
            "builder": "NIXPACKS"
        },
        "deploy": {
            "startCommand": "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0",
            "healthcheckPath": "/healthz"
        }
    }
    
    with open("railway.json", "w") as f:
        json.dump(railway_config, f, indent=2)
    
    # Create packages.txt for Streamlit Cloud (if needed for system packages)
    packages_content = """# System packages for Streamlit Cloud
# Add any required system packages here
# libgl1-mesa-glx
# libglib2.0-0
"""
    with open("packages.txt", "w") as f:
        f.write(packages_content)
    
    print("✅ Deployment files created:")
    print("  - Procfile (Heroku)")
    print("  - railway.json (Railway)")
    print("  - packages.txt (Streamlit Cloud)")

def check_model_files():
    """Check if model files exist and their sizes"""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return False
    
    model_files = [
        "FINAL_VisionTransformer_99.92acc_20250613_114100.pth",
        "FINAL_Xception_99.26acc_20250614_021320.pth", 
        "FINAL_EfficientNet_B3_99.11acc_20250612_121115.pth"
    ]
    
    print("\n📊 Model Files Status:")
    total_size = 0
    
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"  ✅ {model_file}: {size_mb:.1f} MB")
        else:
            print(f"  ❌ {model_file}: NOT FOUND")
    
    print(f"\n📦 Total Model Size: {total_size:.1f} MB")
    
    if total_size > 100:
        print("\n⚠️  WARNING: Large model files detected!")
        print("   Some deployment platforms have size limits:")
        print("   - Streamlit Cloud: 1GB limit")
        print("   - Heroku: 500MB slug limit")
        print("   - Consider using Git LFS or cloud storage")
    
    return True

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv/

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch models (optional - comment out if you want to include them)
# *.pth
# *.pt

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Logs
*.log

# Temporary files
tmp/
temp/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore created")

def create_docker_files():
    """Create Docker files for containerized deployment"""
    
    # Dockerfile
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Docker Compose
    docker_compose_content = """version: '3.8'

services:
  deepfake-detection:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("✅ Docker files created:")
    print("  - Dockerfile")
    print("  - docker-compose.yml")

def optimize_for_deployment():
    """Optimize application for deployment"""
    
    # Create optimized requirements.txt (without version conflicts)
    optimized_requirements = """streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
Pillow>=9.5.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.15.0
scikit-learn>=1.3.0
"""
    
    with open("requirements-optimized.txt", "w") as f:
        f.write(optimized_requirements)
    
    print("✅ Optimized requirements created: requirements-optimized.txt")

def show_deployment_guide():
    """Display deployment guide"""
    
    guide = """
🚀 DEPLOYMENT GUIDE

1. STREAMLIT CLOUD (Recommended for beginners)
   - Push code to GitHub
   - Visit: https://share.streamlit.io
   - Connect GitHub repo
   - Deploy automatically
   
2. HUGGING FACE SPACES (Great for ML projects)
   - Create account: https://huggingface.co
   - Create new Space (Streamlit)
   - Upload files or connect Git repo
   - Good for large model files
   
3. RAILWAY (Simple deployment)
   - Visit: https://railway.app
   - Connect GitHub repo
   - Deploy with one click
   
4. HEROKU (Traditional PaaS)
   - Install Heroku CLI
   - heroku create your-app-name
   - git push heroku main
   
5. DOCKER (Any platform)
   - docker build -t deepfake-detection .
   - docker run -p 8501:8501 deepfake-detection

💡 TIPS:
- For large models, consider Git LFS or cloud storage
- Test locally first: streamlit run app.py
- Monitor memory usage during deployment
- Use optimized requirements for faster builds

⚠️  IMPORTANT:
- Some platforms have file size limits
- GPU acceleration may not be available on free tiers
- Consider model compression for production
"""
    
    print(guide)

def main():
    """Main deployment preparation function"""
    
    print("🚀 Deepfake Detection Hub - Deployment Preparation")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found!")
        print("   Please run this script from the project root directory")
        return
    
    # Create deployment files
    create_deployment_files()
    create_gitignore()
    create_docker_files()
    optimize_for_deployment()
    
    # Check model files
    check_model_files()
    
    # Show deployment guide
    show_deployment_guide()
    
    print("\n✅ Deployment preparation complete!")
    print("   Your project is ready for deployment!")

if __name__ == "__main__":
    main() 