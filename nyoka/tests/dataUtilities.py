from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

class DataUtility:

    def __init__(self):
        iris = datasets.load_iris()
        df_multi_class = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df_multi_class['Species'] = iris.target
        features_multi = iris.feature_names

        df_bin_class = pd.read_csv("nyoka/tests/titanic_train.csv")
        features_bin = [name for name in df_bin_class.columns if name != 'Survived']

        df_reg = pd.read_csv('nyoka/tests/auto-mpg.csv')
        df_reg = df_reg.drop(['car name'], axis=1)
        features_reg = list(df_reg.columns.drop(['mpg',]))

        X_multi = df_multi_class[iris.feature_names]
        X_bin = df_bin_class[features_bin]
        X_reg = df_reg[features_reg]

        X_train, X_test, y_train , _ = train_test_split(X_multi, df_multi_class.Species, test_size=0.3, random_state=11)
        X_bin_train, X_bin_test, y_bin_train , _ = train_test_split(X_bin, df_bin_class.Survived, test_size=0.3, random_state=11)
        X_reg_train, X_reg_test, y_reg_train, _ = train_test_split(X_reg, df_reg.mpg, test_size=0.3, random_state=11)

        self.multi_class = (X_train, X_test, y_train, features_multi, 'Species', 'nyoka/tests/test_multi.csv')
        self.binary_class = (X_bin_train, X_bin_test, y_bin_train, features_bin, 'Survived', 'nyoka/tests/test_bin.csv')
        self.regression = (X_reg_train, X_reg_test, y_reg_train, features_reg, 'mpg', 'nyoka/tests/test_reg.csv')

        pd.DataFrame(X_test, columns=features_multi).to_csv("nyoka/tests/test_multi.csv", index=False)
        pd.DataFrame(X_bin_test, columns=features_bin).to_csv("nyoka/tests/test_bin.csv", index=False)
        pd.DataFrame(X_reg_test, columns=features_reg).to_csv("nyoka/tests/test_reg.csv", index=False)

    def get_data_for_binary_classification(self):
        return self.binary_class

    def get_data_for_multi_class_classification(self):
        return self.multi_class

    def get_data_for_regression(self):
        return self.regression