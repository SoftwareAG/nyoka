import queue
import numpy as np
from sklearn.utils import check_array
FLOAT_DTYPES = (np.float64, np.float32, np.float16)

class Lag:
    """
    The Lag class takes `value` number of previous record of the field where it is applied and applies `aggregation` to those values.

    Parameters:
    -----------
    aggregation : String
        aggregation type. The valid types are ["min", "max", "sum", "avg", "median", "product", "stddev"]
    value : Integer (default = 1)
        The number of previous record to aggregate

    
    """
    
    _valid_aggs = ["min", "max", "sum", "avg", "median", "product", "stddev"]
    
    def __init__(self, aggregation, value=1, copy=True):
        assert aggregation in self._valid_aggs, f"Invalid `aggregation` type. Valid types are {self._valid_aggs}"
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
        if self.aggregation == "stddev":
            for row in X:
                std_devs = [np.std(list(q_.queue)) for q_ in q_list]
                self._transformed_X.append(std_devs)
                for idx, col in enumerate(row):
                    q_list[idx].put(col)
                    q_list[idx].get()
        else:
            NotImplementedError(f"The aggregation type '{self.aggregation}' is not implemented!")
            
        
    def transform(self, X, y=None):
        return np.array(self._transformed_X)
        
    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X,y)
    
    def __repr__(self):
        return f"Lag(aggregation='{self.aggregation}', value={self.value})"