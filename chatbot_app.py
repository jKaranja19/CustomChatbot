# chatbot_app.py

import json
import random
import tensorflow as tf
import numpy as np
from nltk_utils import tokenize, bag_of_words, generate_chatbot_response
from tensorflow.keras.models import load_model
from data_preprocess import create_chatbot_vocabulary, clean_chatbot_vocab

# Load intents JSON
with open('intent.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the saved chatbot brain
chatbot_brain = load_model('chatty.keras')

# Get vocabulary and tags from data_preprocess.py
all_words, tags, tokenized_patterns = create_chatbot_vocabulary(file_path="intent.json")
vocabulary = clean_chatbot_vocab(vocabulary=all_words)

# Determine the input size for the model
input_size = len(vocabulary)

# Create the console chatbot app
bot_name = "Dokotari"
print("What's the issue? I'd very much love to help (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    # Preprocess the user request to make it understandable by the AI model
    sentence = tokenize(sentence)
    X = bag_of_words(tokenized_pattern=sentence, vocabulary=vocabulary)
    X = X.reshape(1, -1)

    # Print debugging information
    print(f"Input size: {input_size}")
    print(f"Shape of X: {X.shape}")

    # Predict the user intent
    pred_proba, intent_tag = generate_chatbot_response(chatbot_brain, X, tags)

    # If the chatbot is sure that it understood the user request
    if pred_proba > 0.75:
        for intent in intents['intents']:
            if intent_tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    # Else
    else:
        print(f"{bot_name}: Sorry, I do not understand. Could you make it simpler, please?")
