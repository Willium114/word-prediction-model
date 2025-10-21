# 🧠 Word Prediction Model (GRU-based)

This project is a **Word Prediction App** built using **TensorFlow** and **Streamlit**.  
It uses a **Gated Recurrent Unit (GRU)** neural network to predict the **next possible words** in a given text sequence.

---

## 📌 Project Overview

This model is trained on sample text data using a **GRU (Gated Recurrent Unit)** network — a type of recurrent neural network (RNN) known for handling sequential data efficiently.  
The model learns language patterns and predicts what words are likely to come next based on previous context.

---

## 🚀 Features

✅ Predicts next words based on input text  
✅ Simple and interactive Streamlit web app  
✅ Uses TensorFlow Keras GRU architecture  
✅ Easily extendable for larger datasets  
✅ Ready for deployment on **Streamlit Cloud** or **Heroku**

---

## 🏗️ Model Architecture

```text
Embedding Layer → GRU Layer (128 units) → Dense Layer (Softmax Activation)
word-prediction-model/
│
├── app.py                      # Streamlit web app
├── gru_word_prediction_model.h5 # Trained GRU model (or downloaded externally)
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Ignored files/folders
| Tool                   | Description                  |
| ---------------------- | ---------------------------- |
| **Python 3.10+**       | Programming language         |
| **TensorFlow / Keras** | Model training and inference |
| **Streamlit**          | Web UI framework             |
| **NumPy & Pandas**     | Data preprocessing           |
| **Matplotlib**         | Visualization                |
🧑‍💻 Developer
Name: M. Aqib Javed
GitHub: Willium114
Project: Word Prediction Model using GRU
