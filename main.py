# main.py

import numpy as np
import tensorflow as tf
from keras import layers
from data_preprocess import create_chatbot_vocabulary, create_train_data, clean_chatbot_vocab

# Load vocabulary, tags, and patterns from data_preprocess.py
all_words, tags, tokenized_patterns = create_chatbot_vocabulary(file_path="intent.json")
vocabulary = clean_chatbot_vocab(vocabulary=all_words)
X_train, y_train = create_train_data(vocabulary=vocabulary, tags=tags, patterns=tokenized_patterns)

input_size = X_train.shape[1]
output_size = np.unique(y_train).shape[0]

# Debugging: Print the shapes and unique labels
print(f"Input size: {input_size}")
print(f"Output size: {output_size}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Unique y_train values: {np.unique(y_train)}")

chatbot_brain = tf.keras.Sequential([
    layers.Input(shape=(input_size,)),  # Correct input_size
    layers.Dense(100, activation='relu', name='layer1'),
    layers.Dense(50, activation='relu', name='layer2'),
    layers.Dense(output_size, activation='softmax', name='layer3')
])

chatbot_brain.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.008),
    metrics=['accuracy']
)

history = chatbot_brain.fit(
    X_train,
    y_train,
    epochs=70,
    batch_size=64,
    verbose=1,
)

chatbot_brain.save('chatty.keras')
