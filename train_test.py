import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TrainTest:
    def __init__(self, vocab_size, max_length, embedding_dim, lstm_dim):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        self.stop_words = set(stopwords.words('english'))
        self.lstm_dim = lstm_dim
        self.model = self.build_model()
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        
        

    def build_model(self):
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_dim)),
                tf.keras.layers.Dense(24, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid")
            ])
            return model
            
    
    def train(self, training_padded, training_labels, testing_padded, testing_labels, num_epochs=30):
        history = self.model.fit(
            training_padded, training_labels,
            epochs=num_epochs,
            validation_data=(testing_padded, testing_labels),
            verbose=2
        )
        return history
    

    
    def save(self, path):
        print("=======saving model===============================")
        self.model.save(path)
        print("=======model has been saved successfully==========")

        
    def load(self, path):
        self.loaded_model = tf.keras.models.load_model(path)
        print("=====model has been loaded successfully=======")
        
    def cleanup_text(self, new_text):
        cleaned_text = self.clean_text(new_text)
        padded_text = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_text = pad_sequences(padded_text, maxlen=self.max_length, padding="post", truncating="post")
        return padded_text    