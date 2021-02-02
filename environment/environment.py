import joblib
from utils.data_io import load_nc
from utils.train import xgb_model


def predict():
    x = [55.86, -4.26, 30, 9, 1]
    model = joblib.load('../xgboost_model.m')
    temp = model.predict(x)
    print(temp)


if __name__ == "__main__":
    # train_X, train_y, test_X, test_y = load_nc(filename='ERsst.mnmean.nc')
    #
    # mae = xgb_model(train_X, train_y, test_X, test_y)
    #
    # print(mae)

    predict()
