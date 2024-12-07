from utils.LSTM import run_LSTM
from utils.data_update import data_update

if __name__ == '__main__':
    data_update("data")
    run_LSTM("data/stock_data.csv")