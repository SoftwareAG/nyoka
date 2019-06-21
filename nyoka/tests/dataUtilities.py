from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

class DataUtility:

    def __init__(self):
        iris = datasets.load_iris()

        df_reg = pd.DataFrame(iris.data, columns=iris.feature_names)
        
        df_multi_class = pd.DataFrame(iris.data, columns=iris.feature_names) 
        df_multi_class['species'] = iris.target

        df_bin_class = pd.DataFrame(iris.data, columns=iris.feature_names)
        df_bin_class['binary'] = [i%2 for i in range(df_bin_class.shape[0])]
        df_bin_class['test'] = [i%3 for i in range(df_bin_class.shape[0])]

        X_multi = df_multi_class[iris.feature_names]
        X_binary = df_bin_class[iris.feature_names]
        X_binary["test"] = df_bin_class.test
        X_reg = df_reg.drop(['sepal length (cm)'], axis=1)

        X_train, X_test, y_train , _ = train_test_split(X_multi, df_multi_class.species, test_size=0.3, random_state=11)
        X_bin_train, X_bin_test, y_bin_train , _ = train_test_split(X_binary, df_bin_class.binary, test_size=0.3, random_state=11)
        X_reg_train, X_reg_test, y_reg_train, _ = train_test_split(X_reg, df_reg['sepal length (cm)'], test_size=0.3, random_state=11)

        self.multi_class = (X_train, X_test, y_train, iris.feature_names, 'species', 'nyoka/tests/test.csv')
        self.binary_class = (X_bin_train, X_bin_test, y_bin_train, iris.feature_names+["test"], 'binary', 'nyoka/tests/test_bin.csv')
        self.regression = (X_reg_train, X_reg_test, y_reg_train, X_reg_test.columns, 'sepal length (cm)', 'nyoka/tests/test_reg.csv')

        pd.DataFrame(X_test, columns=iris.feature_names).to_csv("nyoka/tests/test.csv", index=False)
        pd.DataFrame(X_bin_test, columns=iris.feature_names+["test"]).to_csv("nyoka/tests/test_bin.csv", index=False)
        pd.DataFrame(X_reg_test, columns=X_reg_test.columns).to_csv("nyoka/tests/test_reg.csv", index=False)

    def get_data_for_binary_classification(self):
        return self.binary_class

    def get_data_for_multi_class_classification(self):
        return self.multi_class

    def get_data_for_regression(self):
        return self.regression