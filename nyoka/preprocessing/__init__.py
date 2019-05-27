import queue
import numpy as np
import pandas as pd
import inspect, marshal
from types import FunctionType
from sklearn.utils import check_array
from sklearn.base import TransformerMixin
FLOAT_DTYPES = (np.float64, np.float32, np.float16)


class NyokaFunctionTransformer(TransformerMixin):
    """
    The NyokaFunctionTransformer class takes a custom function and transforms the data according to that function.

    Note - If the custom function is treating the dataset as pandas DataFrame then only use the transformer inside DataframeMapper, otherwise use it as a step in the pipeline.

    Parameters:
    ----------
    function : A python function
        The custom function
    input_cols : list
        The column names of the dataset before executing the custom function
    output_cols : list
        The column names of the dataset after executing the custom function


    """
    def __init__(self, function, input_cols, output_cols):
        self.func = function
        self.func_code = marshal.dumps(self.func.__code__)
        self.source_code = inspect.getsource(function)
        self.globals = function.__globals__
        self.input_cols = input_cols
        self.output_cols = output_cols
        del self.func
    
    def fit(self, X, y=None):
        import inspect
        calframe = inspect.getouterframes(inspect.currentframe(), 2)
        caller_name = calframe[3][1]
        if caller_name.endswith("memory.py"):
            self.data_format = "array"
        elif caller_name.endswith("mapper.py"):
            self.data_format = "dataframe"
        self.func = FunctionType(marshal.loads(self.func_code), self.globals)
        return self
    
    def transform(self, X, y=None):
        X = check_array(X)
        if self.data_format == "dataframe":
            X = pd.DataFrame(X,columns=self.input_cols)
        return self.func(X)
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def __repr__(self):
        return f"NyokaFunctionTransformer(function={self.func.__name__}, input_cols={self.input_cols}, output_cols={self.output_cols})"


class Lag(TransformerMixin):
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
            raise NotImplementedError(f"The aggregation type '{self.aggregation}' is not implemented!")
        return self
            
        
    def transform(self, X, y=None):
        return np.array(self._transformed_X)
        
    
    def __repr__(self):
        return f"Lag(aggregation='{self.aggregation}', value={self.value})"