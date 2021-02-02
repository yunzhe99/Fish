import netCDF4 as nc
import pandas as pd
import numpy as np
import joblib


def load_nc(filename='./ERsst.mnmean.nc', train_test_ratio=0.7):
    # file_obj = nc.Dataset(filename)
    #
    # lat = file_obj.variables['lat'][:]
    # lon = file_obj.variables['lon'][:]
    # sst = file_obj.variables['sst'][:]
    # time = file_obj.variables['time']
    # times = nc.num2date(time[:], time.units, only_use_python_datetimes=True, only_use_cftime_datetimes=False)
    #
    # data_list = []
    #
    # for lat_index in range(len(lat)):
    #     for lon_index in range(len(lon)):
    #         for time_index in range(len(time)):
    #             sst_value = sst[time_index][lat_index][lon_index]
    #             if sst_value is not "--":
    #                 data_list.append([lat[lat_index],
    #                                   lon[lon_index],
    #                                   times[time_index].date().year,
    #                                   times[time_index].date().month,
    #                                   times[time_index].date().day,
    #                                   sst_value])
    #
    # data_list = np.array(data_list)
    # data_list = pd.DataFrame(data_list, columns=['lat', 'lon', 'year', 'month', 'day', 'sst'])
    # # data_list.to_csv('data.csv')
    #
    # joblib.dump(data_list, 'data.j')
    data_list = joblib.load('../data/data.j')
    data = data_list.dropna(axis=0, how='any')
    data = np.array(data).astype(float)

    data_len = len(data)

    train_test_split = int(data_len * train_test_ratio)

    train_X = data[:train_test_split][:, 0:-1]
    train_y = data[:train_test_split][:, -1]
    test_X = data[train_test_split + 1:][:, 0:-1]
    test_y = data[train_test_split + 1:][:, -1]

    return train_X, train_y, test_X, test_y
