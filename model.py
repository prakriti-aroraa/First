import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib


# Importing data 
df=pd.read_csv("spam.csv",encoding="latin-1")
df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace=True)
df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# Data Preprocessing
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Ensure X is always a list/iterable
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif not isinstance(X, list):
            X = [X]
        
        return [self.process_text(text) for text in X]
    
    def process_text(self, text):
        # Convert to lowercase
        text = str(text).lower()
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and non-alphanumeric tokens, apply stemming
        processed_tokens = [
            self.ps.stem(word) for word in tokens 
            if word.isalnum() and word not in self.stop_words
        ]
        
        return " ".join(processed_tokens)

#Splitting training and test data
x_train,x_test,y_train,y_test=train_test_split(df["text"],df["target"],test_size=0.3)

# Create the pipeline
spam_pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer()),
    ('classifier', SVC(kernel='linear'))
])

#Fitting pipeline
spam_pipeline.fit(x_train,y_train)

#Dumpping model
joblib.dump(spam_pipeline, 'spam_classification_pipeline.pkl')
print("Model saved successfully!")