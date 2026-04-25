import wikipediaapi
import re
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Config
MAX_SEQUENCE_LEN = 20
EPOCHS = 5

print("Fetching data from Wikipedia API...")
wiki_wiki = wikipediaapi.Wikipedia(user_agent='MyTextPredictionApp/1.0 (https://example.com/my-text-prediction)', language='en')
topics = ['Artificial intelligence', 'Machine learning', 'Deep learning', 'Neural network']

raw_text = ""
for topic in topics:
    page = wiki_wiki.page(topic)
    if page.exists():
        raw_text += page.text + "\n"

print(f"Total characters collected: {len(raw_text)}")

print("Preprocessing data...")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

cleaned_corpus = clean_text(raw_text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([cleaned_corpus])
total_words = len(tokenizer.word_index) + 1

print(f"Total unique words: {total_words}")

input_sequences = []
# Create sequences windowed 
words = cleaned_corpus.split()
window_size = MAX_SEQUENCE_LEN

for i in range(1, len(words)):
    n_gram_sequence = tokenizer.texts_to_sequences([" ".join(words[max(0, i-window_size):i+1])])[0]
    input_sequences.append(n_gram_sequence)

input_sequences = pad_sequences(input_sequences, maxlen=MAX_SEQUENCE_LEN, padding='pre')

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

print(f"Building LSTM model (X shape: {X.shape})...")
model = Sequential()
model.add(Embedding(total_words, 50, input_length=MAX_SEQUENCE_LEN-1))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Training model... this might take 1-3 minutes locally.")
model.fit(X, y, epochs=EPOCHS, verbose=1, batch_size=128)

print("Saving lstm_text_predictor.h5 and tokenizer.pkl ...")
model.save('lstm_text_predictor.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("Training complete! Files are saved and ready for FastAPI.")
