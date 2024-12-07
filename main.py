from utils.LSTM import run_LSTM
from utils.data_update import data_update
import os

if __name__ == '__main__':
    data_update("data")
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file.endswith(".csv"):
                run_LSTM(root + "/" + file, file.split("_")[0])
