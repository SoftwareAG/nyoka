from __future__ import absolute_import

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


from pprint import pprint
import warnings
from .statsmodels_to_pmml import StatsmodelsToPmml

class ArimaToPMML:
    """
    Exports time-series models from statsmodels library into PMML

    Parameters:
    -----------
    results_obj: 
        Instance of AR(I)MAResultsWrapper / (SARI/VAR)MAXResultsWrapper from statsmodels
    pmml_file_name: string
        Name of the PMML
    conf_int : list (optional)
        Confidence intervel. A list of values mentioning the percentage of confidence.
        e.g., conf_int = [80,95] will create OutputField for lower bound and upper bound of confidence interval with 80% and 95%.
    model_name : string (optional)
        Name of the model
    description : string (optional)
        Description of the model
    Returns
    -------
    Generates PMML object and exports it to `pmml_file_name`
    """
    def __init__(self, results_obj=None, pmml_file_name="from_arima.pmml", conf_int=None, model_name=None, description=None):
        warnings.warn("`ArimaToPMML` is deprecated and it will be removed in 4.3 release. Use `StatsmodelsToPmml` instead."\
            ,DeprecationWarning,stacklevel=2)
        StatsmodelsToPmml(results_obj,pmml_file_name,conf_int,model_name,description)