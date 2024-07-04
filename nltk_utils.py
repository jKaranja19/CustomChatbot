import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')

# Import the NLTK stemmer
stemmer = PorterStemmer()

# Define the tokenization function
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Define the stemming function
def stem(word):
    return stemmer.stem(word.lower())

# Define the bag of words function
def bag_of_words(tokenized_pattern, vocabulary):
    tokenized_pattern = [stem(word) for word in tokenized_pattern]
    bag = np.zeros(len(vocabulary), dtype=np.float32)

    # For each word in the pattern, create a binary vector
    for index, word in enumerate(vocabulary):
        if word in tokenized_pattern:
            bag[index] = 1.0

    return bag

# Define the model prediction function
def generate_chatbot_response(chatbot, user_request, tags):
    y_pred = chatbot.predict(user_request, verbose=0)
    intent_class_id = np.argmax(y_pred)
    pred_proba = np.max(y_pred)
    intent_tag = tags[intent_class_id]
    return pred_proba, intent_tag
