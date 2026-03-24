# Team-3: Handwritten Math Calculator

A Streamlit web app that recognizes handwritten math symbols and equations drawn live on a canvas, then evaluates them to give the correct answer.  
This project combines a CNN model trained on handwritten symbols (`+, -, *, /, =, x`) and digits (0–9) with calculator logic.

## 👩‍💻 Team
- Surabhi Sarda - 2410030123 
- Rishika Tirumala - 2410030122
- Srirama Bharath - 2410030129

## 🚀 Features
- Draw math symbols or full equations directly in the browser.
- Recognizes handwritten symbols (`+, -, *, /, =, x`) and digits (0–9).
- Evaluates basic arithmetic operations and returns the result.
- Built with **TensorFlow/Keras**, **OpenCV**, and **Streamlit**.

## 📂 Project Structure
handwritten-equation-solver/
```
│
├── app.py                   # Streamlit app (main entry point)
├── symbol_model.keras       # Trained CNN for symbols
├── digit_model.keras        # Trained CNN for digits (optional)
├── training_notebook.ipynb  # Google Colab notebook (exported)
├── README.md                # About the Project
└── .gitignore               
```
## 📒 Training Notebook
The CNN model was trained in Google Colab.  
- View the notebook here: Google Colab Link ("https://colab.research.google.com/drive/1jNrkQYZQ4eJD0DWat5Fx8hsu7X-bv_T3?usp=sharing")
- The exported notebook is included as `training_notebook.ipynb`.

## 🧩 Dependencies
- `streamlit`
- `streamlit-drawable-canvas`
- `tensorflow`
- `opencv-python`
- `numpy`
