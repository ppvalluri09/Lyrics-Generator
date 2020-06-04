import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dropout, Dense, LSTM, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
import pandas as pd
import tensorflow.keras.utils as ku
import numpy as np

def get_data():
    df = pd.read_csv("./data/songdata.csv")['text'].values.tolist()
    return df

corpus = " ".join(get_data())
corpus = corpus.lower().split("\n")

print("Corpus Prepared", "\n", "="*32, "\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

word_index = tokenizer.word_index
total_words = len(word_index) + 1

print("Vocab Size", total_words)

##BATCH_SIZE = 128
##difference = input_sequences.shape[0] - BATCH_SIZE
##input_sequences = input_sequences[:input_sequences.shape[0] - difference, :]
##input_sequences = input_sequences.reshape(BATCH_SIZE,
##                                          input_sequences.shape[0]//BATCH_SIZE,
##                                          28)
##print("Data Shape", input_sequences.shape)
##x, y = input_sequences[:, :, :-1], input_sequences[:, :, -1]

TRAIN = False
if TRAIN:

    sequences = []

    for line in corpus:
        sequence = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(sequence)):
            each_line = sequence[:i+1]
            sequences.append(each_line)

    max_len = max([len(x) for x in sequences])
    
    input_sequences = pad_sequences(sequences, maxlen=max_len, padding="pre")
    input_sequences = input_sequences[:10000, :]

    print("Data Shape", input_sequences.shape)
    print("Preparing Model\n")
    model = Sequential()
    model.add(Embedding(total_words, 100, input_shape=(max_len-1,)))
    model.add(Bidirectional(LSTM(150, activation=tf.nn.relu, return_sequences=True)))
    model.add(Bidirectional(LSTM(100, activation=tf.nn.relu, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dense(1000, activation=tf.nn.relu))
    model.add(Dense(800, activation=tf.nn.relu))
    model.add(Dense(total_words, activation=tf.nn.softmax))

    model.compile(optimizer="SGD", loss="sparse_categorical_crossentropy", metrics=["acc"])
    print("Model Architecture\n", "="*32, "\n", model.summary())

    history = model.fit(input_sequences[:, :-1],
                        input_sequences[:, -1],
                        batch_size=64,
                        epochs=100, verbose=1)
    model.save("./models/lyric_generatorE100.h5")

    next_words = 200
    seed = "what are you"

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], padding='pre', maxlen=max_len-1)
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in word_index.items():
            if index == predicted:
                output_word = word
                break
        seed += " " + output_word
    print("Generated Text\n", seed)  

else:
    max_len = int(open("maxlen.txt", "r").read())
    model = tf.keras.models.load_model("./models/lyric_generatorE100.h5")
    next_words = int(input("Enter the number of words to generate:- "))
    seed = str(input("Enter the starting words of your desired lyrics:- ")).lower()

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], padding='pre', maxlen=max_len-1)
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        output_word = ""
        for word, index in word_index.items():
            if index == predicted:
                output_word = word
                break
        if len(seed.split()) % 15 == 0:
            seed += "\n"
        seed += " " + output_word
    print("Generated Text\n", seed)    
