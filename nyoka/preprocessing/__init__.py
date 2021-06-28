"""
 Copyright (c) 2004-2016 Zementis, Inc.
 Copyright (c) 2016-2021 Software AG, Darmstadt, Germany and/or Software AG USA Inc., Reston, VA, USA, and/or its

 SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 """
import queue
import numpy as np
from sklearn.utils import check_array
from sklearn.base import TransformerMixin
FLOAT_DTYPES = (np.float64, np.float32, np.float16)


class Lag(TransformerMixin):
    """
    The Lag class takes `value` number of previous record of the fields where it is applied and applies `aggregation` to those values.

    Parameters
    ----------
    aggregation : String
        aggregation type. The valid types are ["min", "max", "sum", "avg", "median", "product", "stddev"]
    value : Integer (default = 2)
        The number of previous record to aggregate. Should be greater than 1.

    
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
    
    def __init__(self, aggregation, value=2, copy=True):
        assert aggregation in self._VALID_AGGS, f"Invalid `aggregation` type. Valid types are {self._VALID_AGGS}"
        assert value > 1, "`value` should be greater than 1"
        self.aggregation = aggregation
        self.value = value
        self.copy = copy
        
    def fit(self, X, y=None):
        """
        Does nothing.

        Returns
        -------
        The same object
        """   
        return self
            
        
    def transform(self, X, y=None):
        """
        Transforms the given X by taking `value` number of previous records and applying `aggregation` method

        Parameters
        ----------
        X : Pandas DataFrame or numpy array
            The input data
        y : 
            It is ignored.

        Returns
        -------
        Transformed X as numpy array  
        """
        self._transformed_X = list()
        X = check_array(X, copy=self.copy, estimator=self)
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
        return np.array(self._transformed_X)
        
    
    def __repr__(self):
        return f"Lag(aggregation='{self.aggregation}', value={self.value})"