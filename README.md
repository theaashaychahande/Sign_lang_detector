# Sign_lang_detector

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-green?logo=opencv)](https://opencv.org)

A real-time American Sign Language (ASL) alphabet detection system using computer vision and deep learning.

## Features 
- Real-time ASL alphabet detection (A-Z)
- Hand landmark detection using MediaPipe
- Lightweight CNN model for classification
- Simple and beginner-friendly implementation
- Webcam integration with OpenCV

### Prerequisites
- Python 3.8 or higher
- Webcam

## 🛠️ Technical Stack
| Component | Technology |
|-----------|------------|
| **Core Framework** | TensorFlow 2.12 |
| **Computer Vision** | OpenCV 4.7 + MediaPipe |
| **Model Architecture** | Custom CNN with optimized depthwise convolutions |
| **Inference Engine** | TensorFlow Lite (for mobile deployment) |
| **Data Pipeline** | TensorFlow Data API with real-time augmentation |

## 📊 Performance Metrics
| Metric | Score |
|--------|-------|
| Training Accuracy | 96.4% |
| Validation Accuracy | 92.1% |
| Inference Latency (CPU) | <50ms |
