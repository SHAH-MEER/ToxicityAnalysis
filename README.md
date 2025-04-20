# 🛡️ ToxicityAnalysis

## 📋 Project Overview
ToxicityAnalysis is a machine learning project that analyzes and classifies text comments into multiple toxicity categories using a sequential neural network with LSTM architecture. The model is built with Keras and TensorFlow and is designed to identify different types of toxic content in text, including toxic, severe_toxic, obscene, threat, insult, and identity_hate categories.

## 📁 Project Structure
```
├── app/                  # Application code and deployment files
│   ├── app.py            # Main application code
│   ├── requirements.txt  # Required dependencies
│   └── toxicity.h5       # Compiled model file
├── model(s)/             # Trained models and model-related utilities
│   └── toxicity.h5       # Saved model file
├── notebook(s)/          # Jupyter notebooks documenting the process
│   └── toxicity.ipynb    # Main notebook with data preprocessing and model development
├── visualisations/       # Data and model visualizations
│   ├── model_architecture.png  # Visual representation of the model architecture
│   └── training_history.png    # Training metrics visualization
└── .gitignore            # Git ignore file
```

## 📊 Dataset
The project uses the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle, which contains text comments labeled with six toxicity categories:
- 🚫 toxic
- ⚠️ severe_toxic
- 🔞 obscene
- 🔪 threat
- 🗯️ insult
- 👥 identity_hate

Each comment can belong to multiple categories or none at all.

## 🧠 Model Architecture
The neural network model uses a sequential architecture with the following layers:
- Embedding layer (1800, 32) - 6,400,032 parameters
- Bidirectional LSTM layer (64) - 16,640 parameters
- Dense layer (128) - 8,320 parameters
- Dense layer 1 (256) - 33,024 parameters
- Dense layer 2 (128) - 32,896 parameters
- Dense layer 3 (6) - 774 parameters

Total trainable parameters: 6,491,686 (24.76 MB)

## 🏋️ Training Process
The model was trained for 10 epochs with the following configuration:
- Batch size: 872/872
- Training time: ~94-106s per epoch
- Final training loss: 0.0208
- Final validation loss: 0.0178

Model performance metrics:
- Precision: 0.9035911560058594
- Recall: 0.9006057381629944

## 🖥️ Interactive Demo
The project includes a Gradio web interface for real-time toxicity analysis:
- Input: Textbox for user comments
- Output: Toxicity labels with binary scores (True/False)
- Deployment: Can be run locally or shared via a public URL

## 🚀 Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Gradio
- Matplotlib

### Setup
1. Clone the repository:
```bash
git clone https://github.com/SHAH-MEER/ToxicityAnalysis.git
cd ToxicityAnalysis
```

2. Install required dependencies:
```bash
pip install -r app/requirements.txt
```

## 🔧 Usage

### Model Training
The training process is documented in the `toxicity.ipynb` notebook with the following steps:
1. Data preprocessing and vectorization
2. Model architecture design
3. Training with validation
4. Performance evaluation

### Running the Application
To use the Gradio interface for toxicity analysis:

```python
# Load the saved model
model = tf.keras.models.load_model('toxicity.h5')

# Prepare text input
input_str = vectorizer('your text here')

# Make prediction
res = model.predict(np.expand_dims(input_str,0))

# Launch the interface
interface.launch(share=True)
```

The interface will be available at:
- Local URL: http://127.0.0.1:7864
- Public URL: A temporary URL will be generated for sharing

## 📈 Model Evaluation
The model's performance is evaluated using standard metrics:
- Confusion Matrix: Shows True labels VS Predicted Labels
- Training History: Plot shows loss and metrics over epochs to diagnose overfitting
- Precision and Recall scores

## 🔮 Future Improvements
- Fine-tune hyperparameters for better performance
- Implement additional preprocessing techniques
- Explore more advanced architectures
- Enhance the web interface with additional features

## 📬 Contact
-GitHub: https://github.com/SHAH-MEER/
-Gmail: shahmeershahzad67@gmail.com
