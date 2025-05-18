# Emotion-Detection-System

A machine learning-based system to classify emotions (such as **joy, anger, sadness, fear**) from raw text input using the **Random Forest algorithm**. The project leverages various Python libraries for preprocessing, model building, evaluation, visualization, and deployment.

---

## 🚀 Features

- Clean and preprocess text data using `neattext`
- Convert text into numerical features using **TF-IDF**
- Train a **Random Forest Classifier** for multi-class emotion detection
- Visualize performance with **confusion matrix**, **ROC curves**, and **precision-recall curves**
- Save and reuse the trained model for real-time predictions
- Simple demo function for emotion prediction from new text input

---

## 🧰 Tech Stack

- Python 3
- Pandas, NumPy
- NeatText (text preprocessing)
- Scikit-learn (ML modeling)
- Seaborn & Matplotlib (visualizations)
- Joblib (model deployment)

---

## 📂 Project Structure

Emotion-Detection-System/
├── emotion_dataset_raw.csv
├── emotion_model.joblib
├── tfidf_vectorizer.joblib
├── emotion_detection.py
├── README.md



---

## 📊 Model Pipeline

1. **Data Loading:**  
   Load labeled text data with emotions.

2. **Preprocessing:**  
   - Remove stopwords, punctuation, and user handles using `neattext`.

3. **Feature Engineering:**  
   - Convert cleaned text into numerical format using **TF-IDF vectorizer**.

4. **Model Training:**  
   - Train a **Random Forest Classifier** on the processed data.

5. **Evaluation:**  
   - Use classification report, confusion matrix, ROC, and precision-recall curves.

6. **Deployment:**  
   - Save the trained model and vectorizer using `joblib`.

---

## 📈 Example Output

**Confusion Matrix:**

![Confusion Matrix](path/to/confusion_matrix.png)

**ROC Curve:**

![ROC Curve](path/to/roc_curve.png)

*(Add screenshots/plots if available)*

---

## 🔍 Predict Emotion (Demo)

```python
from joblib import load
import neattext.functions as nfx

# Load model and vectorizer
model = load("emotion_model.joblib")
vectorizer = load("tfidf_vectorizer.joblib")

def predict_emotion(text):
    cleaned = nfx.remove_stopwords(nfx.remove_userhandles(nfx.remove_punctuations(text)))
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)
    return prediction[0]

# Example
print(predict_emotion("I am so excited today!"))  # Output: joy


📝 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/Emotion-Detection-System.git
cd Emotion-Detection-System
Install required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python emotion_detection.py
📌 Future Improvements
Expand dataset to more emotion classes

Integrate with a web interface using Flask or Streamlit

Deploy as an API or chatbot

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
Dataset: [Add source if applicable]

Libraries: Scikit-learn, NeatText, Seaborn, Joblib

pgsql
Copy
Edit

Would you like me to also generate the `requirements.txt` file or provide a sample dataset format (`emotion_dataset_raw.csv`)?
