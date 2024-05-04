import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences

def load_glove_model(glove_file):
    print("Loading GloVe model...")
    with open(glove_file, 'r', encoding='utf-8') as f:
        word_to_index = {}
        embeddings_index = {}
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            word_to_index[word] = i
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print("Done.")
    return word_to_index, embeddings_index

def prepare_data(X, y, word_to_index, max_len):
    X_indices = []
    for doc in X:
        words = doc.lower().split()
        indices = [word_to_index.get(word, 0) for word in words]
        X_indices.append(indices)
    X_pad = pad_sequences(X_indices, maxlen=max_len)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X_pad, y_encoded

def build_cnn_model(embedding_matrix, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_matrix.shape[1],
                        weights=[embedding_matrix],
                        input_length=max_len,
                        trainable=False))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Example usage:
X = [...]  # Your text data
y = [...]  # Your labels
glove_file = "path/to/glove.6B.50d.txt"  # Path to your GloVe file

# Load GloVe embeddings
word_to_index, embeddings_index = load_glove_model(glove_file)

# Prepare data
max_len = 100  # Maximum length of a document
X_pad, y_encoded = prepare_data(X, y, word_to_index, max_len)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_encoded, test_size=0.2, random_state=42)

# Create embedding matrix
vocab_size = len(word_to_index) + 1  # Add one for padding
embedding_dim = len(embeddings_index['a'])
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Build CNN model
num_classes = len(np.unique(y))
model = build_cnn_model(embedding_matrix, max_len)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
