import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque

import numpy as np
import pandas as pd
import random

# Her defasinda ayni farkli sonuclara cikmamak icin random belirlenmis sayilarimiza seed belirleriz.
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1, 
                test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    """
    Data setimizi Yahoo Finance sitesinden load yapar.
    Parametreler:
        ticker (str/pd.DataFrame): tahmin etmek istedigimiz borsanin ticker degeri, ornegin AAPL, TESL, vs...
        n_steps (int): birgun sonrasini tahmin etmek icin kac gunun verisi kullanilacaktir, default olarak 50 set edilmistir.
        scale (bool): fiyat datasini 0-1 arasi scale edecektir, default olarak True set edildi.
        shuffle (bool): Data seti karistirip karistirmayacagi, default olarak True belirlendi
        lookup_step (int): Sonraki kac adimi tahmin edecegi, default olarak 1 belirlendi(gun)
        test_size (float): data setimizin test orani, default olarak 0.2 belirlendi (20% test)
        feature_columns (list): modelimizin ihtiyac duyacagi sutunlar listesi, default olarak butun sutunlari alacaktir
    """
    # istedigimiz ticker degeri daha once indirildiyse yeniden indirmeye gerek kalmadan data klasorunden yukleyecek.
    if isinstance(ticker, str):
        #aksi taktirde yahoo finance sitesinden indirecek
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    # Veri setimizin butun elemanlarini result'ta kaydedecek
    result = {}
    result['df'] = df.copy()

    # load_data metodunda belirltilen sutunlarin veri setinde olup olmadigini kontrol etmek icin:
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    if scale:
        column_scaler = {}
        # fiyat verisini 0-1 arasi scale etmek icin:
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # MinMaxScaler'dan donen instanceleri sutunlara yerlestirmek icin:
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)

    # NaN olarak nitelendirilen bos veya sayi bazli veri olmayan veri hanelerini cikartiyoruz:
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # son pointerin nerede oldugunu elde etmek icin n_steps + look_up steps -1 yapariz.Ve bunu rsult kumesine ekleriz
    last_sequence = list(sequences) + list(last_sequence)
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    result['last_sequence'] = last_sequence
    
    # X ve Y veri setlerini hazirlamak icin:
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # N=Bu veri setlerini Numpy array listelerine donusturmek icin:
    X = np.array(X)
    y = np.array(y)

    # X'i neural networka sigdirabilmek icin reshape yapmaliyiz/
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    
    # veri setini, ogrenme ve test fazlarina bolmek icin:
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)
    return result


def create_model(sequence_length, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # birinci layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
        elif i == n_layers - 1:
            # son layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # gizli layerlar
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # her layer sonrasinda bir dropout eklemek:
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model