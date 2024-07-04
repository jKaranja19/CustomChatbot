# data_preprocess.py

import json
import numpy as np
from sklearn.utils import shuffle
from nltk_utils import tokenize, stem, bag_of_words

def create_chatbot_vocabulary(file_path):
    with open(file_path, 'r') as file:
        intents = json.load(file)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tags.append(intent['tag'])
        for pattern in intent['patterns']:
            words = tokenize(pattern)
            all_words.extend(words)
            xy.append((tags[-1], words))
    return sorted(all_words), sorted(tags), xy

def clean_chatbot_vocab(vocabulary):
    ignore_list = ['?', '!', '.', ',']
    return sorted(set([stem(word) for word in vocabulary if word not in ignore_list]))

def create_train_data(vocabulary, tags, patterns):
    X_train = []
    y_train = []

    for tag, tokenized_pattern in patterns:
        bag = bag_of_words(tokenized_pattern=tokenized_pattern, vocabulary=vocabulary)
        X_train.append(bag)
        y_train.append(tags.index(tag))

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)  # Shuffle here

    return X_train, y_train

# Example usage to preprocess data
if __name__ == "__main__":
    all_words, tags, tokenized_patterns = create_chatbot_vocabulary(file_path="intent.json")
    vocabulary = clean_chatbot_vocab(vocabulary=all_words)
    X_train, y_train = create_train_data(vocabulary=vocabulary, tags=tags, patterns=tokenized_patterns)
