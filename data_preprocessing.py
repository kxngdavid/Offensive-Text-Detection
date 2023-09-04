# Preprocessing class

import re
import nltk
import string
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataPreprocessing:
    def __init__(self, max_length):
        self.max_length = max_length
        self.stopwords = set(stopwords.words('english'))
        self.tokenizer = Tokenizer()
    
    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'@[^:]*:', '', text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
        text = re.sub(r'[^a-zA-Z]',' ', text)
        text = word_tokenize(text)
        text = [text for text in text if text.lower() not in self.stopwords]
        text = ' '.join(text)
        return text
    
    def preprocess_data(self, training_sentences, testing_sentences, training_labels, testing_labels):
        self.tokenizer.fit_on_texts(training_sentences)
        training_sequences = self.tokenizer.texts_to_sequences(training_sentences)
        training_padded = pad_sequences(
            training_sequences,
            maxlen=self.max_length,
            padding="post", truncating="post")
        
        testing_sequences = self.tokenizer.texts_to_sequences(testing_sentences)
        testing_padded = pad_sequences(testing_sequences, maxlen=self.max_length, padding="post", truncating="post")
        
        return training_padded, testing_padded, np.array(training_labels), np.array(testing_labels)
    
    def cleanup_text(self, new_text):
        cleaned_text = self.clean_text(new_text)
        padded_text = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_text = pad_sequences(padded_text, maxlen=self.max_length, padding="post", truncating="post")
        return padded_text
