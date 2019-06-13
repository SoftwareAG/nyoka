import queue
import numpy as np
from sklearn.utils import check_array
from sklearn.base import TransformerMixin
FLOAT_DTYPES = (np.float64, np.float32, np.float16)


class Lag(TransformerMixin):
    """
    The Lag class takes `value` number of previous record of the fields where it is applied and applies `aggregation` to those values.

    Parameters:
    -----------
    aggregation : String
        aggregation type. The valid types are ["min", "max", "sum", "avg", "median", "product", "stddev"]
    value : Integer (default = 1)
        The number of previous record to aggregate

    
    """
    
    _VALID_AGGS = ["min", "max", "sum", "avg", "median", "product", "stddev"]
    _AGG_FUNC_MAP = {
        "min" : np.min,
        "max" : np.max,
        "sum" : np.sum,
        "avg" : np.mean,
        "median" : np.median,
        "product" : np.product,
        "stddev" : np.std
    }
    
    def __init__(self, aggregation, value=1, copy=True):
        assert aggregation in self._VALID_AGGS, f"Invalid `aggregation` type. Valid types are {self._VALID_AGGS}"
        self.aggregation = aggregation
        self.value = value
        self.copy = copy
        
    def fit(self, X, y=None):
        self._transformed_X = list()
        X = check_array(X, copy=self.copy, warn_on_dtype=True, estimator=self,\
                        dtype=FLOAT_DTYPES,force_all_finite="allow-nan")       
        q_list = [queue.Queue() for i in range(len(X[0]))]
            
        for _ in range(self.value):
            for q_ in q_list:
                q_.put(0.0)
        
        for row in X:
            aggregated_vals = [self._AGG_FUNC_MAP[self.aggregation](q_.queue) for q_ in q_list]
            self._transformed_X.append(aggregated_vals)
            for idx, col in enumerate(row):
                q_list[idx].put(col)
                q_list[idx].get()
        
        return self
            
        
    def transform(self, X, y=None):
        return np.array(self._transformed_X)
        
    
    def __repr__(self):
        return f"Lag(aggregation='{self.aggregation}', value={self.value})"