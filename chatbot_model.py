import json
import string
import random
import os
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("wordnet")

# Load the raw data
with open('data.json') as f:
    data = json.load(f)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

words = []
classes = []
doc_x = []
doc_y = []

# Process raw data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_x.append(pattern)
        doc_y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Lemmatize and remove punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

# Prepare training data
training = []
out_empty = [0] * len(classes)

for idx, doc in enumerate(doc_x):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Model parameters
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])

# Path to the saved model
model_path = "chatbot_model.h5"

# Check if the model file exists
if os.path.exists(model_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded from 'chatbot_model.h5'")
else:
    # Create and compile the model
    model = Sequential([
        Dense(128, input_shape=input_shape, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(output_shape, activation='softmax')
    ])

    adam = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print(model.summary())
    model.fit(x=train_X, y=train_y, epochs=500, verbose=1)

    # Save the trained model
    model.save(model_path)
    print("Model saved to 'chatbot_model.h5'")
# Define functions
def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = [labels[r[0]] for r in y_pred]
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "Sorry, I didn't get that."

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)