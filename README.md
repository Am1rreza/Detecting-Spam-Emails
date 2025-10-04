# 📧 Email Spam Detection using LSTM

This project builds an **LSTM-based deep learning model** to classify emails as **Spam** or **Ham (Non-Spam)**.  
It uses **Natural Language Processing (NLP)** techniques such as text cleaning, stopword removal, tokenization, and sequence padding.  
The model is implemented using **TensorFlow** and achieves high accuracy in detecting spam emails.

## 📂 Dataset Information

The project uses a dataset of **5,171 email samples**, each labeled as either:

- **Spam** — Unwanted or promotional emails  
- **Ham** — Legitimate (non-spam) emails  

Each record contains:
- `label`: Category of the email (`spam` or `ham`)  
- `text`: The full email content  
- `label_num`: Numeric form of the label (0 for ham, 1 for spam)

Before training, the dataset was **balanced** by downsampling the majority class (ham) to ensure an equal number of spam and ham samples.

## 🧹 Data Preprocessing & Cleaning

To prepare the email texts for modeling, several **NLP preprocessing** steps were performed:

1. **Removing Unnecessary Headers:**  
   Removed occurrences of the word *“Subject”* from all email texts.

2. **Punctuation Removal:**  
   All punctuation marks were removed using Python’s `string.punctuation`.

3. **Stopword Removal:**  
   Common English stopwords (like *“the”*, *“and”*, *“is”*) were removed using NLTK’s predefined stopword list.

4. **Text Normalization:**  
   Converted all text to lowercase to ensure consistency during tokenization.

After preprocessing, the text became cleaner and more meaningful for model training.

## ⚙️ Model Training

After cleaning and tokenizing the data, an LSTM-based neural network was trained to classify emails as **spam** or **ham**.

### Model Structure
- **Embedding layer:** turns words into 32-dimensional vectors.  
- **LSTM layer:** learns patterns in the email text.  
- **Dense layer:** processes the learned features.  
- **Output layer:** predicts spam (1) or ham (0).

### Training Setup
- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy  
- **Epochs:** 20  
- **Batch size:** 32  
- **Callbacks:** EarlyStopping and ReduceLROnPlateau to avoid overfitting.

The model successfully learned patterns that help detect spam messages from email content.

## 📊 Model Evaluation & Results

After training, the model was tested on unseen emails to check how well it can detect spam.

### Test Performance
- **Test Accuracy:** 95.17%  
- **Test Loss:** 0.26  

### Accuracy Plot
We also plotted the **training and validation accuracy** over epochs to see how the model improved during training.

- The plot shows that the model quickly learned to distinguish spam from ham emails.  
- Training and validation accuracy were very close, indicating the model did not overfit.

## 📝 Conclusion & Insights

- The LSTM model can effectively detect spam emails with high accuracy.  
- Preprocessing steps like removing stopwords and punctuation were important for good performance.  
- The model generalizes well on unseen data, making it useful for practical email spam detection.  
- Further improvements could include using more data, experimenting with larger LSTM layers, or trying transformer-based models for even better accuracy.
