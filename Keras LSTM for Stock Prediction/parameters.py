import os
import time
from tensorflow.keras.layers import LSTM

# tahmin asamasi icin kac gunun verilerinin kullanilacagi
N_STEPS = 250
# kac gun sonrasinin verileri tahmin edilecektir
LOOKUP_STEP = 30

# test yuzdesi
TEST_SIZE = 0.2
# hangi sutunlar kullanilacaktir
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# tarih ayarlari
date_now = time.strftime("%Y-%m-%d")

### modelimiz icin kullanilacak parametreler

N_LAYERS = 3
# LSTM hucreleri(cells)
CELL = LSTM
# 256 LSTM noronlari
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# Cift yonluluk
BIDIRECTIONAL = False

### Ogrenim parametreleri

# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 100

# tickerin aldigi deger ogrenme icin
ticker = "AAPL"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# modeli kaydetmek icin:
model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"