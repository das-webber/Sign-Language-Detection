# Sign Language Detection

This project detects sign language gestures using a machine learning model.  
Features:

- Image upload prediction
- Real-time webcam prediction (6 PM – 10 PM)
- Tkinter GUI

## Folder Structure

- `data/raw/` - Raw dataset (downloaded from Kaggle)
- `data/processed/` - Preprocessed images
- `models/` - Trained model files
- `src/` - Training & preprocessing scripts
- `app.py` - GUI application

## How to Run

### 1. Preprocess data
```bash
python src/preprocess.py

### 2. Train model
python src/train.py --data_dir data/processed --labels data/labels.csv --save_path models/saved_model.h5

### 3. Run gui
run app.ipynb
