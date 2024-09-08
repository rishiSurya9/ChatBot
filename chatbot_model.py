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

# Define the raw data
data = {
    "intents": [
        {"tag": "greeting",
         "patterns": ["Hello", "How are you?", "Hi There", "Hi", "What's up", "Good morning", "Hey"],
         "responses": ["Howdy Partner!", "Hello there!", "How are you doing?", "Greetings!", "Hey! How's it going?"]
        },
        {"tag": "age",
         "patterns": ["how old are you", "when is your birthday", "when were you born", "what's your age", "tell me your age"],
         "responses": ["I am 24 years old", "I was born in 1966", "My birthday is July 3rd and I was born in 1996", "born on 03/07/1996", "I’m ageless – I was created in the 21st century."]
        },
        {"tag": "date",
         "patterns": ["what are you doing this weekend", "do you want to hang out sometime?", "what are your plans for this week", "any plans for today", "how's your week going"],
         "responses": ["I am available this week", "I don't have any plans", "I am not busy", "Just hanging out in the digital realm.", "I’m always here to chat!"]
        },
        {"tag": "name",
         "patterns": ["what's your name", "what are you called", "who are you", "can you tell me your name", "who should I call you"],
         "responses": ["My name is Kippi", "I'm Kippi", "You can call me Kippi", "I go by Kippi!", "I’m Kippi, at your service."]
        },
        {"tag": "goodbye",
         "patterns": ["bye", "g2g", "see ya", "adios", "cya", "farewell", "take care", "see you later"],
         "responses": ["It was nice speaking to you", "See you later", "Speak Soon", "Goodbye! Have a great day!", "Catch you later!"]
        },
        {"tag": "weather",
         "patterns": ["what's the weather like", "tell me the weather", "how's the weather today", "what's the forecast", "is it sunny"],
         "responses": ["I can't check the weather, but I hope it's nice out!", "I don't have weather updates, but stay safe!", "Why don't you check a weather app?"]
        },
        {"tag": "joke",
         "patterns": ["tell me a joke", "make me laugh", "do you know any jokes", "tell me something funny", "give me a joke"],
         "responses": ["Why don't scientists trust atoms? Because they make up everything!", "What do you call fake spaghetti? An impasta!", "Why did the scarecrow win an award? Because he was outstanding in his field!"]
        },
        {"tag": "advice",
         "patterns": ["give me some advice", "what's your advice", "help me with something", "I need some advice", "can you give me advice"],
         "responses": ["Always be yourself; everyone else is already taken.", "The best way to predict the future is to invent it.", "Stay positive, work hard, and make it happen."]
        },
        {"tag": "hobby",
         "patterns": ["what are your hobbies", "do you have any hobbies", "what do you like to do", "tell me about your interests", "what's your favorite activity"],
         "responses": ["I enjoy chatting with you! That’s my main hobby.", "I’m here to learn and assist, that’s my favorite activity.", "I like exploring new topics with you."]
        },
        {"tag": "time",
         "patterns": ["what time is it", "tell me the time", "what's the current time", "can you check the time", "what’s the time now"],
         "responses": ["I can't check the time, but you can look at your device.", "Sorry, I don't have the capability to check the time.", "Please check your device for the current time."]
        },
        {"tag": "games",
         "patterns": ["do you play games", "what games do you like", "tell me about games", "recommend a game", "let's play a game"],
         "responses": ["I’m more of a chatbot, but I can suggest some fun games!", "How about trying some classic board games or video games?", "I don't play games, but I can help you find some good ones."]
        },
        {"tag": "music",
         "patterns": ["what's your favorite music", "do you like music", "tell me about music", "what kind of music do you like", "recommend a song"],
         "responses": ["I enjoy all kinds of music, but I don't have personal preferences.", "How about exploring some new genres? Maybe jazz or classical?", "I don't listen to music, but I can suggest popular songs if you like."]
        },
        {"tag": "travel",
         "patterns": ["where have you traveled", "recommend a travel destination", "what's your favorite place", "tell me about travel", "any travel tips"],
         "responses": ["I haven’t traveled, but I can suggest some popular destinations.", "Traveling to new places is exciting – how about visiting Paris or Tokyo?", "I can give you tips on travel if you’re planning a trip."]
        }
    ]
}

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
model_path = 'chatbot_model.h5'

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