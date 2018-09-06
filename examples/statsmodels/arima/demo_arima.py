import os
import sys
from pprint import pprint
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from importlib import reload
from nyoka import ArimaToPMML



def main():

    # Load data from a csv file
    data_f_name = 'sales_data.csv'
    ts = pd.read_csv(data_f_name, header=0, parse_dates=[0], index_col=0, squeeze=True)
    ts = ts.astype('float64')

    # Non seasonal ARIMA model
    n_ar = 1
    n_diff = 1
    n_ma = 1
    model = sm.tsa.ARIMA(endog=ts, order=(n_ar, n_diff, n_ma))
    results = model.fit()

    # Use exporter to create pmml file
    pmml_f_name = 'non_seasonal_arima.pmml'
    ArimaToPMML(
        time_series_data=ts,
        model_obj=model,
        results_obj=results,
        pmml_file_name=pmml_f_name
    )








if __name__ == '__main__':
    main()
