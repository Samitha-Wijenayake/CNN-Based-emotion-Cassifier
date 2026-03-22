# 🧠 CNN-Based Emotion Classifier

An end-to-end Machine Learning project showcasing a **Custom Deep Convolutional Neural Network (CNN)** for binary image classification. 
The project allows users to classify images of faces into two distinct emotions: **Happy** or **Sad**.

![App Screenshot](https://img.shields.io/badge/Status-Live%20Application-success)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)

---

## 🚀 Project Overview

This project goes through the entire deep learning lifecycle—from data collection and processing up to deploying a fully functioning web application. The core engine is a custom CNN architecture, trained completely from scratch without transfer learning.

### ✨ Key Features
* **Custom Architecture**: Built iteratively from the ground up using Keras Sequential API.
* **Interactive UI**: A beautiful, user-friendly frontend built with **Streamlit**.
* **Real-time Inference**: Drag and drop any `.jpg` or `.png` face and get instant predictions with confidence scores.
* **Automatic Image Reformatting**: Handles missing alpha channels and resizes user input to exactly match the `256x256x3` required shape.

## 🛠️ Technology Stack
* **Deep Learning Framework:** TensorFlow 2 / Keras
* **Interactive Web Application:** Streamlit
* **Image Processing:** PIL (Python Imaging Library) / NumPy
* **Training Hardware:** Google Colab / GPU (T4)

---

## 💻 How to Run Locally

To test this project on your own machine, follow these simple directions:

1. **Clone the repository**
```bash
git clone https://github.com/Samitha-Wijenayake/CNN-Based-emotion-Cassifier.git
cd CNN-Based-emotion-Cassifier
```

2. **Install the dependencies**
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

3. **Launch the Streamlit Web Application**
```bash
streamlit run app.py
```

4. **Test the Model**
Open your browser to `http://localhost:8501`. Try uploading one of the files from the `data/happy` or `data/sad` training directories to check the model's accuracy!

---

## 📈 Model Performance & Limitations

This prototype classifies images into two states effectively, but as it was trained on a relatively small curated dataset (**~200 images**), it may struggle with:
* Cartoons or drawings (trained strictly on human faces)
* Extreme lighting, shadows, or heavily crowded photos.

*Built with ❤️ taking concepts from Computer Vision and Deep Learning.*
