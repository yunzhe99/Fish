import numpy as np

from data_io import load_nc
from train import xgb_model

if __name__ == "__main__":
    train_X, train_y, test_X, test_y = load_nc(filename='ERsst.mnmean.nc')

    mae = xgb_model(train_X, train_y, test_X, test_y)

    print(mae)
