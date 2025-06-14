# 🕵️ Deepfake Detection Hub

A comprehensive web application for detecting deepfake images using state-of-the-art deep learning models. This project implements three different architectures: Vision Transformer, Xception, and EfficientNet B3, achieving up to **99.92% accuracy**.

## 🌟 Features

- **🔍 Real-time Detection**: Upload single or multiple images for instant deepfake analysis
- **🤖 Multiple Models**: Choose between 3 high-performance models:
  - Vision Transformer (99.92% accuracy)
  - Xception (99.26% accuracy) 
  - EfficientNet B3 (99.11% accuracy)
- **📊 Interactive Statistics**: Comprehensive model performance metrics and comparisons
- **📈 Visualization**: Confusion matrices, performance charts, and training curves
- **🚀 Batch Processing**: Analyze multiple images simultaneously
- **💡 Confidence Scoring**: Get detailed confidence breakdowns for each prediction
- **🎨 Modern UI**: Beautiful, responsive interface built with Streamlit

## 🏗️ Architecture

### Models Used

1. **Vision Transformer (ViT)**
   - Architecture: `vit_base_patch16_224`
   - Parameters: 86.6M
   - Inference Time: ~45ms
   - Best overall accuracy: 99.92%

2. **Xception**
   - Architecture: Depthwise separable convolutions
   - Parameters: 22.9M  
   - Inference Time: ~23ms
   - Excellent efficiency: 99.26%

3. **EfficientNet B3**
   - Architecture: Compound scaling
   - Parameters: 12.2M
   - Inference Time: ~18ms
   - Lightweight: 99.11%

### Dataset
- **140,000+ images** from real-vs-fake dataset
- Balanced distribution of real and fake samples
- High-quality preprocessing with data augmentation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- At least 4GB RAM

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd deepfake-detection-project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify model files**
Ensure your trained model files are in the `models/` directory:
```
models/
├── FINAL_VisionTransformer_99.92acc_20250613_114100.pth
├── FINAL_Xception_99.26acc_20250614_021320.pth
└── FINAL_EfficientNet_B3_99.11acc_20250612_121115.pth
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 📱 Usage Guide

### Single Image Detection
1. Select "Single Image" upload option
2. Choose your model from the sidebar
3. Upload an image (PNG, JPG, JPEG)
4. Click "🔍 Analyze Image"
5. View results with confidence scores

### Batch Processing
1. Select "Multiple Images" upload option
2. Upload multiple images
3. Click "🔍 Analyze All Images"
4. Review batch results in the table format

### Model Comparison
- Navigate to "📊 Statistics" tab for model comparisons
- View "📈 Performance" tab for detailed metrics
- Check "ℹ️ About" for project information

## 🌐 Deployment Options

### 1. Streamlit Cloud (FREE) ⭐ **Recommended**

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy with one click!

**Pros:**
- Completely free
- Easy setup
- Automatic updates
- Built for Streamlit apps

**Note:** Large model files (>100MB) may require Git LFS or alternative storage.

### 2. Hugging Face Spaces (FREE)

**Steps:**
1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space with Streamlit
3. Upload your files
4. Configure app settings

**Pros:**
- Free hosting
- ML-focused platform
- Good for showcasing models
- Supports large files

### 3. Railway (FREE tier)

**Steps:**
1. Visit [railway.app](https://railway.app)
2. Connect GitHub repo
3. Deploy automatically

**Pros:**
- Simple deployment
- Good free tier
- Automatic builds

### 4. Heroku (Limited free tier)

**Steps:**
1. Install Heroku CLI
2. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```
3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### 5. Local Network Sharing

For quick local demos:
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| Vision Transformer | 99.92% | 99.92% | 99.92% | 99.92% | 86.6M |
| Xception | 99.26% | 99.25% | 99.27% | 99.26% | 22.9M |
| EfficientNet B3 | 99.11% | 99.10% | 99.12% | 99.11% | 12.2M |

## 🛠️ Technical Stack

- **Backend**: PyTorch, timm
- **Frontend**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Image Processing**: PIL, OpenCV
- **Data Handling**: Pandas, NumPy

## 📁 Project Structure

```
deepfake-detection-project/
├── app.py                          # Main Streamlit application
├── model_utils.py                  # Model utilities and inference functions
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── models/                         # Trained model files
│   ├── FINAL_VisionTransformer_*.pth
│   ├── FINAL_Xception_*.pth
│   └── FINAL_EfficientNet_B3_*.pth
├── deepfake_detection.py          # Original training script
└── Deepfake_detection.ipynb       # Jupyter notebook
```

## 🔧 Customization

### Adding New Models
1. Create model class in `model_utils.py`
2. Add configuration in `MODEL_CONFIGS` in `app.py`
3. Update model loading logic

### Modifying UI
- Edit CSS in the `st.markdown()` sections
- Customize colors, fonts, and layout
- Add new tabs or sections

### Performance Optimization
- Implement model caching with `@st.cache_resource`
- Use GPU acceleration when available
- Optimize image preprocessing

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check file paths in `MODEL_CONFIGS`
   - Ensure model files exist and are not corrupted
   - Verify CUDA/CPU compatibility

2. **Memory Issues**
   - Reduce batch size for multiple images
   - Use CPU if GPU memory is insufficient
   - Clear cache with `st.cache_resource.clear()`

3. **Dependency Conflicts**
   - Use virtual environment
   - Update pip: `pip install --upgrade pip`
   - Install exact versions from requirements.txt

4. **Streamlit Issues**
   - Clear browser cache
   - Restart Streamlit server
   - Check port availability

## 📋 TODO / Future Enhancements

- [ ] Add video deepfake detection
- [ ] Implement model ensemble voting
- [ ] Add LIME/SHAP explainability
- [ ] Support for more image formats
- [ ] API endpoint creation
- [ ] Mobile app version
- [ ] Real-time webcam detection
- [ ] Advanced preprocessing options

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Academic Use

This project is designed for educational and research purposes. If you use this code in your research, please cite:

```bibtex
@misc{deepfake-detection-hub,
  title={Deepfake Detection Hub: A Comprehensive Web Application},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/deepfake-detection-project}}
}
```

## 🙏 Acknowledgments

- **timm library** for pre-trained model architectures
- **Streamlit** for the amazing web framework
- **PyTorch** for the deep learning foundation
- **Real-vs-Fake dataset** creators
- Open source community for inspiration and support

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/deepfake-detection-project/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/deepfake-detection-project/discussions)

---

**Built with ❤️ for academic research and education**

🌟 **Star this repo if it helped you!** 🌟 
