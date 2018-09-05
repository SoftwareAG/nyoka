import os
import sys
from pprint import pprint
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from importlib import reload
sys.path.insert(0, "/home/osboxes/zementis/pmml_stsmdl")  # <------------------ change this
from nyoka.pmml.statsmodels.exporters.arima import ArimaToPMML
from nyoka.pmml.statsmodels.exporters.arima import reconstruct


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

    # Reconstruction
    recon_results, recon_arima_mdl = reconstruct(pmml_f_name)

    # Forecast
    n_forecast = 5
    if n_diff:
        stsmdl_forecast = model.predict(params=results.params, end=n_forecast)
        recon_forecast = recon_arima_mdl.predict(params=recon_results.params, end=n_forecast + 1)
    else:
        stsmdl_forecast = model.predict(params=results.params, end=n_forecast)
        recon_forecast = recon_arima_mdl.predict(params=recon_results.params, end=n_forecast)


    # Compare forecasts
    if np.array_equal(stsmdl_forecast, recon_forecast):
        print('reconstruction successful ---------------------------------------------------------------------')
    else:
        print('reconstruction failed -------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
