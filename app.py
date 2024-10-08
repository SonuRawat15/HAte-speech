import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Download NLTK data (only need to do this once)
nltk.download('stopwords')
nltk.download('wordnet')

# Step 1: Data Collection (Load the dataset)
df = pd.read_csv('hate_speech_dataset.csv')  # Replace with your dataset path
print(df.head())

# Step 2: Data Preprocessing
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess text
def clean_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#','', text)
    
    # Remove punctuations and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    # Tokenize the text
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop]
    
    return ' '.join(tokens)

# Apply the cleaning and preprocessing functions
df['clean_tweet'] = df['tweet'].apply(clean_text)
df['preprocessed_tweet'] = df['clean_tweet'].apply(preprocess_text)

# Display cleaned and preprocessed tweets
print(df[['tweet', 'clean_tweet', 'preprocessed_tweet']].head())

# Step 3: Handling Class Imbalance (Optional)
sns.countplot(x='label', data=df)
plt.show()

# You can handle class imbalance here if needed (e.g., resampling)

# Step 4: Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['preprocessed_tweet']).toarray()
y = df['label'].values

# Step 5: Model Training (Train-Test Split and Logistic Regression)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Classification Report
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 7: Save the Model and Vectorizer for Deployment
joblib.dump(model, 'hate_speech_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved.")
