import os

# python 3.8 a viac odporucam ale vsetko nad 3.5 by malo byt ok
# https://www.python.org/downloads/ ale na macu ho budes mat uz od zaciatku asi

# toto treba dat do konzoly/terminalu (ak to nepojde tak len pip)
# pip3 install tensorflow pandas matplotlib sklearn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # len aby sa nevypisovali nezmyselne logy
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

hidden_layers = [20, 20, 6] # pocty neuronov v skrytych vrstvach

# stlpce relevantne pre trenovanie
useful_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'] 

# potrebne v mojom pripade, lebo nueonky nerobia s textom
# toto je taky skolkarsky pristup :D je na to aj funkcia
def numerizeGenre(data):
    for index, row in data.iterrows():
        if row['playlist_genre'] == "edm":
            data.at[index, 'playlist_genre'] = '0'
        elif row['playlist_genre'] == "latin":
            data.at[index, 'playlist_genre'] = '1'
        elif row['playlist_genre'] == "pop":
            data.at[index, 'playlist_genre'] = '2'
        elif row['playlist_genre'] == "rap":
            data.at[index, 'playlist_genre'] = '3'
        elif row['playlist_genre'] == "rock":
            data.at[index, 'playlist_genre'] = '4'
        elif row['playlist_genre'] == "r&b":
            data.at[index, 'playlist_genre'] = '5'

    data['playlist_genre'] = data['playlist_genre'].astype('int64')
    return data


def loadData():
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")

    #zmazanie riadkov, ktore obsahuju prazdne bunky (NaN)
    test.dropna(inplace=True)
    train.dropna(inplace=True)

    return test, train

def prepareData(train, test):
    #rozdelenie na trenovacie a testovacie data (testovacie su 20% zo vsetkych)
    train, val = train_test_split(train, test_size=0.2)
    train_pred = train['playlist_genre']
    train = train[useful_cols]
    train_pred = pd.get_dummies(train_pred, columns="playlist_genre")

    X_val = val[useful_cols]
    Y_val = val['playlist_genre']
    Y_val = pd.get_dummies(Y_val, columns="playlist_genre")

    test_data = test[useful_cols]
    test_predictions = test["playlist_genre"]
    test_predictions = pd.get_dummies(test_predictions, columns="playlist_genre")

    # upravi hodnoty tak aby boli v co najmensom mozno intervale < -1, 1 > pri dodrzani pomerov medzi hodnotami 
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    X_val = scaler.fit_transform(X_val)
    test_data = scaler.fit_transform(test_data)

    # *_pred su vysledky k datam ktore vkladas,   X_val a Y_val su validacne, 
    # kde X su 'useful_cols' na ktorych testujes a Y_val su vysledky aby si vedela povedat ci si urcila spravne alebo zle
    return train, train_pred, test_data, test_predictions, X_val, Y_val

# tu sa da vyhrat s parametrami pri ktorych budes mat naj vysledky 
def neuralNetwork(train, test):
    train_data, train_predictions, test_data, test_predictions, val_data, val_predictions = prepareData(train, test)

    # nieco k regularizacii, ale nie je potrebna tento krok vyplyval z nasho zadania
    # https://keras.io/api/layers/regularizers/
    # regularizer = tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00002)

    model = tf.keras.models.Sequential()
    
    # shape=(11, ) pocet parametrov na vstupe v dalo by sa zapisat aj len(useful_cols)
    model.add(tf.keras.layers.Input(shape=(11,)))
    model.add(tf.keras.layers.Dense(hidden_layers[0], dtype='float64', activation="relu")) # , kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Dense(hidden_layers[1], dtype='float64', activation="relu")) # , kernel_regularizer=regularizer))
    model.add(tf.keras.layers.Dense(hidden_layers[2], dtype='float64', activation="softmax"))

    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile nieco k parametrom co sa tu daju ponastavovat
    # a este nieco k optimalizator0m https://www.tensorflow.org/api_docs/python/tf/keras/optimizers#classes_2
    # Adam je ale uplne vhodny na taketo a nezhltne vsetku RAM :D 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # tu nastavujes parametre modelu
    # epochs = 70 -  vsetky data tam prebehnu 70krat
    # batch_size = 1000 - data tam idu po "balickoch" cize 1000 dat naraz
    h = model.fit(train_data, train_predictions, epochs=70, batch_size=1000, use_multiprocessing=True, workers=8,
                  validation_data=(val_data, val_predictions))

    print("----------------------------TEST----------------------------")

    test_result = model.evaluate(x=test_data, y=test_predictions, batch_size=5)

    print(f'Test data accuracy is {round(test_result[1] * 100, 2)}%')

    # vykreslenie grafu (otvori sa v novom okne)
    plt.plot(h.history['accuracy'], color="blue")
    plt.plot(h.history['val_accuracy'], color="red")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs") # epocha je prechod vsetkych dat cez siet
    plt.show()


def svm(train, test):
    train = numerizeGenre(train)
    test = numerizeGenre(test)
    
    # pre ten regresor treba zmenit format toho pola data sa nemenia
    X_train = train[useful_cols].to_numpy()
    y_train = train['playlist_genre'].to_numpy()

    # X su vzdy data na ktorych odhadujes / trenujes
    # Y su vzdy vysledky k datam v X pre urcenie presnosti
    X_test = test[useful_cols].to_numpy()
    y_test = test['playlist_genre'].to_numpy()

    print("-------------------------------SVM----------------------------------")
    
    # aby ta to nemiatlo, SVC patri pod SVM (Support Vector Machine), musi to byt klasifikator preto SVC
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Tu sa volaju, vyssie definovane funkcie
test, train = loadData()

neuralNetwork(train, test)

# s tymto opatrne trva to faaaaaaaaaaaaakt dlho 
# svm(train, test)
