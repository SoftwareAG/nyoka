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

class HEADER_INFO:
    COPYRIGHT = "Copyright (c) 2021 Software AG"
    APPLICATION_NAME = "Nyoka"
    APPLICATION_VERSION = "5.3.0"
    DEFAULT_DESCRIPTION = "Default description"

class MISSING_VALUE_TREATMENT_METHOD:
    AS_IS = "asIs"
    AS_MEAN = "asMean"
    AS_MODE = "asMode"
    AS_MEDIAN = "asMedian"
    AS_VALUE = "asValue"
    RETURN_INVALID = "returnInvalid"

class PMML_SCHEMA:
    VERSION = "4.4.1"

class TREE_SPLIT_CHARACTERISTIC:
    BINARY = "binarySplit"
    MULTI = "multiSplit"

class FUNCTION:
    ADDITION = "+"
    SUBSTRACTTION = "-"
    MULTIPLICATION = "*"
    DIVISION = "/"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    AVERAGE = "avg"
    MEDIAN = "median"
    PRODUCT = "product"
    LOG10 = "log10"
    LOGN = "ln"
    SQUARE_ROOT = "sqrt"
    ABSOLUTE = "abs"
    EXPONENT = "exp"
    POWER = "pow"
    THRESHOLD = "threshold"
    FLOOR = "floor"
    CEILING = "ceil"
    ROUND = "round"
    MODULO = "modulo"
    IS_MISSING = "isMissing"
    IS_NOT_MISSING = "isNotMissing"
    IS_VALID = "isValid"
    IS_NOT_VALID = "isNotValid"
    EQUAL = "equal"
    NOT_EQUAL = "notEqual"
    LESS_THAN = "lessThan"
    LESS_OR_EQUAL = "lessOrEqual"
    GREATER_THAN = "greaterThan"
    GREATER_OR_EQUAL = "greaterOrEqual"
    AND = "and"
    OR = "or"
    NOT = "not"
    IS_IN = "isIn"
    IS_NOT_IN = "isNotIn"
    IF = "if"
    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"
    SUBSTRING = "substring"
    TRIM_BLANKS = "trimBlanks"
    CONCAT = "concat"
    REPLACE = "replace"
    MATCHES = "matches"
    FORMAT_NUMBER = "formatNumber"
    FORMAT_DATETIME = "formatDatetime"
    DATEDAYS_SINCE_YEAR = "dateDaysSinceYear"
    DATESECONDS_SINCE_YEAR = "dateSecondsSinceYear"
    DATESECONDS_SINCE_MIDNIGHT = "dateSecondsSinceMidnight"
    NORMAL_CDF = "normalCDF"
    NORMAL_PDF = "normalPDF"
    STANDARD_NORMAL_CDF = "stdNormalCDF"
    STANDARD_NORMAL_PDF = "stdNormalPDF"
    ERF = "erf"
    NORMAL_IDF = "normalIDF"
    STANDARD_NORMAL_IDF = "stdNormalIDF"
    EXPM1 = "expm1"
    HYPOT = "hypot"
    LOGN1P = "ln1p"
    RINT = "rint"
    SIN = "sin"
    ASIN = "asin"
    SINH = "sinh"
    COS = "cos"
    ACOS = "acos"
    COSH = "cosh"
    TAN = "tan"
    ATAN = "atan"
    TANH = "tanh"


class CATEGORICAL_SCORING_METHOD:
    MAJORITY_VOTE = "majorityVote"
    WEIGHTED_MAJORITY_VOTE = "weightedMajorityVote"

class CONTINUOUS_SCORING_METHOD:
    MEDIAN = "median"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weightedAverage"

class REGRESSION_NORMALIZATION_METHOD:
    SIMPLEMAX = "simplemax"
    SOFTMAX = "softmax"
    LOGISTIC = "logit"
    PROBIT = "probit"
    CLOGLOG = "cloglog"
    EXPONENTIAL = "exp"
    LOGLOG = "loglog"
    CAUCHIT = "cauchit"

class ARRAY_TYPE:
    INTEGER = "int"
    REAL = "real"
    STRING = "string"

class MINING_FUNCTION:
    ASSOCIATION_RULES = "associationRules"
    SEQUENCES = "sequences"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIMESERIES = "timeSeries"
    MIXED = "mixed"

class SVM_REPRESENTATION:
    SUPPORT_VECTORS = "SupportVectors"
    COEFFICIENTS = "Coefficients"

class SVM_CLASSIFICATION_METHOD:
    OVR = "OneAgainstAll"
    OVO = "OneAgainstOne"

class MULTIPLE_MODEL_METHOD:
    MAJORITY_VOTE = "majorityVote"
    WEIGHTED_MAJORITY_VOTE = "weightedMajorityVote"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weightedAverage"
    MEDIAN = "median"
    WEIGHTED_MEDIAN = "weightedMedian"
    MAX = "max"
    SUM = "sum"
    WEIGHTED_SUM = "weightedSum"
    SELECT_FIRST = "selectFirst"
    SELECT_ALL = "selectAll"
    MODEL_CHAIN = "modelChain"

class COMPARISON_MEASURE_KIND:
    DISTANCE = "distance"
    SIMILARITY = "similarity"

class CLUSTERING_FILED_COMPARE_FUNCTION:
    ABSOLUTE_DIFF = "absDiff"
    GAUSSIAN_SIMILARITY = "gaussSim"
    DELTA = "delta"
    EQUAL = "equal"
    TABLE = "table"

class CLUSTERING_MODEL_CLASS:
    CENTER_BASED = "centerBased"
    DISTRIBUTION_BASED = "distributionBased"

class NN_NORMALIZATION_METHOD:
    SIMPLEMAX = "simplemax"
    SOFTMAX = "softmax"

class NN_ACTIVATION_FUNCTION:
    THRESHOLD = "threshold"
    LOGISTIC = "logistic"
    TANH = "tanh"
    IDENTITY = "identity"
    EXPONENTIAL = "exponential"
    RECIPROCAL = "reciprocal"
    SQUARE = "square"
    GAUSS = "Gauss"
    SINE = "sine"
    COSINE = "cosine"
    ELLIOTT = "Elliott"
    ARCTAN = "arctan"
    RECTIFIER = "rectifier"
    RADIALBASIS = "radialBasis"

class MAXIMUM_LIKELIHOOD_STAT_METHOD:
    KALMAN = "kalman"
    THETA_RECURSION = "thetaRecursion"

class ARIMA_PREDICTION_METHOD:
    CSS = "conditionalLeastSquares"
    MLE = "exactLeastSquares"

class EXPONENTIAL_SMOOTHING_SEASONALITY:
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"

class EXPONENTIAL_SMOOTHING_TREND:
    ADDITIVE = "additive"
    DAMPED_ADDITIVE = "damped_additive"
    MULTIPLICATIVE = "multiplicative"
    DAMPED_MULTIPLICATIVE = "damped_multiplicative"
    POLYNOMIAL_EXPONENTIAL = "polynomial_exponential"

class EXPONENTIAL_SMOOTHING_AND_ARIMA_TRANSFORMATION:
    LOGARITHMIC = "logarithmic"
    SQUARE_ROOT = "squareroot"

class TIME_ANCHOR:
    DATETIMEMILLISECONDS_SINCE_0 = "dateTimeMillisecondsSince[0]"
    DATETIMEMILLISECONDS_SINCE_1960 = "dateTimeMillisecondsSince[1960]"
    DATETIMEMILLISECONDS_SINCE_1970 = "dateTimeMillisecondsSince[1970]"
    DATETIMEMILLISECONDS_SINCE_1980 = "dateTimeMillisecondsSince[1980]"
    DATETIMESECONDS_SINCE_0 = "dateTimeSecondsSince[0]"
    DATETIMESECONDS_SINCE_1960 = "dateTimeSecondsSince[1960]"
    DATETIMESECONDS_SINCE_1970 = "dateTimeSecondsSince[1970]"
    DATETIMESECONDS_SINCE_1980 = "dateTimeSecondsSince[1980]"
    DATEDAYS_SINCE_0 = "dateDaysSince[0]"
    DATEDAYS_SINCE_1960 = "dateDaysSince[1960]"
    DATEDAYS_SINCE_1970 = "dateDaysSince[1970]"
    DATEDAYS_SINCE_1980 = "dateDaysSince[1980]"
    DATEMONTHS_SINCE_0 = "dateMonthsSince[0]"
    DATEMONTHS_SINCE_1960 = "dateMonthsSince[1960]"
    DATEMONTHS_SINCE_1970 = "dateMonthsSince[1970]"
    DATEMONTHS_SINCE_1980 = "dateMonthsSince[1980]"
    DATEYEARS_SINCE_0 = "dateYearsSince[0]"

class TIMESERIES_USAGE:
    ORIGINAL = "original"
    LOGICAL = "logical"
    PREDICTION = "prediction"

class TIMESERIES_ALGORITHM:
    ARIMA = "ARIMA"
    EXPONENTIAL_SMOOTHING = "ExponentialSmoothing"
    SEASONAL_TREND_DECOMPOSE = "SeasonalTrendDecomposition"
    SPECTRAL_ANALYSIS = "SpectralAnalysis"
    STATE_SPACE_MODEL = "StateSpaceModel"
    GARCH = "GARCH"

class LAG_AGGREGATION:
    AVERAGE = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    PRODUCT = "product"
    SUM = "sum"
    STANDARD_DEVIATION = "stddev"

class ANOMALY_DETECTION_ALGORITHM:
    ISOLATION_FOREST = "iforest"
    ONE_CLASS_SVM = "ocsvm"
    CLUSTER_MEAN_DISTANCE = "clusterMeanDist"
    OTHER = "other"

class SIMPLE_PREDICATE_OPERATOR:
    EQUAL = "equal"
    NOT_EQUAL = "notEqual"
    LESS_THAN = "lessThan"
    LESS_OR_EQUAL = "lessOrEqual"
    GREATER_THAN = "greaterThan"
    GREATER_OR_EQUAL = "greaterOrEqual"
    IS_MISSING = "isMissing"
    IS_NOT_MISSING = "isNotMissing"

class FIELD_USAGE_TYPE:
    ACTIVE = "active"
    PREDICTED = "predicted"
    TARGET = "target"
    SUPPLEMENTARY = "supplementary"
    GROUP = "group"
    ORDER = "order"
    FREQUENCY_WEIGHT = "frequencyWeight"
    ANALYSIS_WEIGHT = "analysisWeight"

class RESULT_FEATURE:
    PREDICTED_VALUE = "predictedValue"
    PREDCITED_DISPLAY_VALUE = "predictedDisplayValue"
    TRANSFORMED_VALUE = "transformedValue"
    DECISION = "decision"
    PROBABILITY = "probability"
    TOP_CATEGORIES = "topCategories"
    AFFINITY = "affinity"
    RESIDUAL = "residual"
    STANDARD_ERROR = "standardError"
    STANDARD_DEVIATION = "standardDeviation"
    CLUSTER_ID = "clusterId"
    CLUSTER_AFFINITY = "clusterAffinity"
    ENTITY_ID = "entityId"
    ENTITY_AFFINITY = "entityAffinity"
    WARNING = "warning"
    RULE_VALUE = "ruleValue"
    REASON_CODE = "reasonCode"
    ANTECEDENT = "antecedent"
    CONSEQUENT = "consequent"
    RULE = "rule"
    RULE_ID = "ruleId"
    CONFIDENCE = "confidence"
    SUPPORT = "support"
    LIFT = "lift"
    LEVERAGE = "leverage"
    CONFIDENCE_INTERVAL_UPPER = "confidenceIntervalUpper"
    CONFIDENCE_INTERVAL_LOWER = "confidenceIntervalLower"

class OPTYPE:
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    CONTINUOUS = "continuous"

class DATATYPE:
    STRING = "string"
    INTEGER = "integer"
    BINARY = "binary"
    FLOAT = "float"
    DOUBLE = "double"
    BOOLEAN = "boolean"
    DATE = "date"
    TIME = "time"
    DATETIME = "dateTime"
    DATEDAYS_SINCE_0 = "dateDaysSince[0]"
    DATEDAYS_SINCE_1960 = "dateDaysSince[1960]"
    DATEDAYS_SINCE_1970 = "dateDaysSince[1970]"
    DATEDAYS_SINCE_1980 = "dateDaysSince[1980]"
    TIMESECONDS = "timeSeconds"
    DATETIMESECONDS_SINCE_0 = "dateTimeSecondsSince[0]"
    DATETIMESECONDS_SINCE_1960 = "dateTimeSecondsSince[1960]"
    DATETIMESECONDS_SINCE_1970 = "dateTimeSecondsSince[1970]"
    DATETIMESECONDS_SINCE_1980 = "dateTimeSecondsSince[1980]"
