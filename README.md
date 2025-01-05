
# 🎥 Review Sentiment Classifier

This project uses a Recurrent Neural Network (RNN) model to classify movie reviews as **Positive** or **Negative**. The model is trained on the IMDB dataset and deployed using Streamlit for an interactive web interface.

Deployed at : [Streamlit](https://review-classify.streamlit.app)

## 🛠 Features

- **Model Training**: A simple RNN model is trained on the IMDB dataset (`simplernn.ipynb`).
- **Prediction Script**: A script to test the trained model with new reviews (`prediction.ipynb`).
- **Web Interface**: A user-friendly Streamlit app to classify movie reviews (`app.py`).
- **Pre-trained Model**: Includes a pre-trained RNN model (`rnn_model_imdb.h5`).

## 📁 Project Structure

```
.
├── app.py                 # Streamlit app for user interaction
├── simplernn.ipynb        # Notebook for training the RNN model
├── prediction.ipynb       # Notebook for testing predictions
├── rnn_model_imdb.h5      # Pre-trained RNN model
├── requirements.txt       # Dependencies required to run the project
└── README.md              # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VanshajR/RNN-Review-Classification.git
   cd RNN-Review-Classification
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the `simplernn.ipynb` notebook to train and save the model
   
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Usage

1. Open the app in your browser (usually at `http://localhost:8501`).
2. Enter a movie review in the text box.
3. Click the **Classify Sentiment** button to get the prediction result.

## 📊 Model Training

The RNN model was trained using TensorFlow on the IMDB dataset. The training process is documented in `simplernn.ipynb`.


