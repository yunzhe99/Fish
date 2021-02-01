import numpy as np
import netCDF4 as nc

file_path = './ERsst.mnmean.nc'

file_obj = nc.Dataset(file_path)

time_bnds = file_obj.variables['time_bnds'][:]

sst = file_obj.variables['sst'][:]

time = file_obj.variables['time']
times = nc.num2date(time[:], time.units)

print(sst.shape)
