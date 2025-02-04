# -Attention-based-Neural-Machine-Translation

import tensorflow as tf
from tensorflow.keras import layers, models

# Define the encoder model
def encoder_model(input_dim, embedding_dim, hidden_dim):
    model = models.Sequential([
        layers.Embedding(input_dim, embedding_dim),
        layers.LSTM(hidden_dim, return_sequences=True),
        layers.LSTM(hidden_dim, return_sequences=True)
    ])
    return model

# Define the decoder model
def decoder_model(output_dim, embedding_dim, hidden_dim):
    model = models.Sequential([
        layers.Embedding(output_dim, embedding_dim),
        layers.LSTM(hidden_dim, return_sequences=True),
        layers.LSTM(hidden_dim, return_sequences=True),
        layers.Dense(output_dim, activation='softmax')
    ])
    return model

# Define the attention mechanism
def attention_model(hidden_dim):
    model = models.Sequential([
        layers.Dense(hidden_dim, activation='tanh'),
        layers.Dense(1, activation='softmax')
    ])
    return model

# Create the encoder, decoder, and attention models
encoder = encoder_model(10000, 128, 256)
decoder = decoder_model(10000, 128, 256)
attention = attention_model(256)

# Compile the models
encoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
decoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
attention.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the models
encoder.fit(x_train, y_train, epochs=10)
decoder.fit(x_train, y_train, epochs=10)
attention.fit(x_train, y_train, epochs=10)
