# %%
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

import matplotlib.pyplot as plt
import numpy as np

from got_loader import load_got_data

# %%
def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


# %% Load training data

# Get training data and labels from custom loader
sentences, seasons = load_got_data()



# %% Training / validation split + vectorization

# Create training / validation split
train_sents, test_sents, train_label, test_label = train_test_split(sentences, 
                                                    seasons, 
                                                    test_size=0.2, 
                                                    random_state=1337)



# %% Tokenize

tokenizer = Tokenizer(num_words=5000)
# fit to training data
tokenizer.fit_on_texts(train_sents)

# tokenized training and test data
train_sents_toks = tokenizer.texts_to_sequences(train_sents)
test_sents_toks = tokenizer.texts_to_sequences(test_sents)

# overall vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index


# %% Pad sentences

sent_lengths = map(len, train_sents_toks)
maxlen = max(sent_lengths)

train_sents_toks_pad = pad_sequences(train_sents_toks,
                                     padding='post',
                                     maxlen=maxlen)
test_sents_toks_pad = pad_sequences(test_sents_toks,
                                     padding='post',
                                     maxlen=maxlen)


# %% Use glove
embedding_dim = 50

embedding_matrix = create_embedding_matrix(os.path.join('glove', f'glove.6B.{embedding_dim}d.txt'),
                                           tokenizer.word_index, 
                                           embedding_dim)



# %% Define model

# new model
model = Sequential()
# Embedding layer
model.add(Embedding(vocab_size, 
                    embedding_dim, 
                    weights=[embedding_matrix],  # we've added our pretrained GloVe weights
                    input_length=maxlen, 
                    trainable=False))            # embeddings are static - not trainable

# MaxPool -> FC -> Output
model.add(GlobalMaxPool1D())
model.add(Dense(48, 
                activation='relu'))
model.add(Dense(len(set(train_label)), 
                activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
# print summary
model.summary()



# %% Train model

history = model.fit(train_sents_toks_pad, train_label,
                    epochs=30,
                    verbose=False,
                    validation_data=(test_sents_toks_pad, test_label),
                    batch_size=10)

# evaluate 
loss, accuracy = model.evaluate(train_sents_toks_pad, train_label, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_sents_toks_pad, test_label, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# plot
def plot_history(H, epochs):
    """
    Utility function for plotting model history using matplotlib
    
    H: model history 
    epochs: number of epochs for which the model was trained
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
plot_history(history, epochs = 50)
# %%
