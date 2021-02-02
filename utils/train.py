import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error


def xgb_model(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(gpu_id=0)
    model.fit(X_train, y_train)

    joblib.dump(model, '../xgboost_model.m')
    model = joblib.load('../xgboost_model.m')

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_pred, y_test)

    return mae
