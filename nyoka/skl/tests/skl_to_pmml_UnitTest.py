import nyoka.skl.skl_to_pmml as sklToPmml
from nyoka import PMML43Ext as pml
import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
class TestMethods(unittest.TestCase):

    def test_Neural_Network(self):

        model = MLPClassifier()
        sk_model,features_name,target_name = iris_dataset_for_classification(model)
        input = sklToPmml.get_neuron_input(features_name)
        self.assertEqual(input.get_numberOfInputs(), len(features_name))

        input1 = sklToPmml.get_neural_layer(sk_model, features_name, target_name)
        input1 = input1[0]
        neuron = input1[0].get_Neuron()
        neuron = neuron[0].get_bias()
        self.assertEqual(len(input1[0].get_Neuron()), 100)
        input2 = sklToPmml.get_neural_layer(sk_model, features_name, target_name)[1]
        self.assertEqual(len(input2.get_NeuralOutput()), 3)

    def test_RegressionModels(self):
        model = LogisticRegression()
        sk_model,features_name,target_name = iris_dataset_for_classification(model)
        categoric_values=()

        funct = sklToPmml.get_mining_func(model)
        self.assertEqual(funct, 'classification')
        reg_table = sklToPmml.get_regrs_tabl(model, features_name, target_name,categoric_values)
        self.assertEqual(reg_table[2].get_intercept(), -0.88331604183313872)

    def test_Naive_Bayes(self):
        model = GaussianNB()
        sk_model,features_names,target_name = iris_dataset_for_classification(model)

        input3 = sklToPmml.get_bayes_inputs(sk_model, features_names)
        d = input3.get_BayesInput()
        self.assertEqual(d[0].get_fieldName(), 'sepal_length')

        output = sklToPmml.get_bayes_output(sk_model, target_name)
        self.assertEqual(output.get_fieldName(), 'species')

        threshold = sklToPmml.get_threshold()
        self.assertEqual(threshold, '0.001')

    def test_KNN(self):
        model = KNeighborsClassifier()
        sk_model,features_name,target_name = iris_dataset_for_classification(model)
        noOfNeighbors = sklToPmml.get_knn_inputs(features_name)
        self.assertEqual(noOfNeighbors.get_KNNInput()[0].get_field(), 'sepal_length')

        comparison = sklToPmml.get_comparison_measure(sk_model)
        self.assertEqual(comparison.get_compareFunction(), 'absDiff')
        trainingInstance = sklToPmml.get_training_instances(sk_model, features_name, target_name)
        trainingInstance = trainingInstance.get_InstanceFields()
        self.assertEqual(trainingInstance.get_InstanceField()[0].get_field(), 'species')


    def test_SVM(self):
        model=SVR(kernel='sigmoid')
        sk_model,feature_names,target_name=auto_dataset_for_regression(model)
        categoric_values=()

        self.assertEqual(
            sklToPmml.get_kernel_type(sk_model)['SigmoidKernelType'].__class__.__name__,
            pml.SigmoidKernelType().__class__.__name__)

        self.assertEqual(
            sklToPmml.get_kernel_type(sk_model)['SigmoidKernelType'].get_gamma(),
            0.14285714285714285)

        self.assertEqual(
            sklToPmml.get_classificationMethod(sk_model),
            'OneAgainstAll'
        )

        self.assertEqual(
            sklToPmml.get_vectorDictionary(sk_model,feature_names,categoric_values).__class__.__name__,
            pml.VectorDictionary().__class__.__name__
        )

        self.assertEqual(
            sklToPmml.get_vectorDictionary(sk_model, feature_names,categoric_values).get_VectorFields().get_FieldRef()[0].get_field(),
            'cylinders'
        )

        self.assertEqual(
            len(sklToPmml.get_vectorDictionary(sk_model, feature_names,categoric_values).get_VectorFields().get_FieldRef()),
            len(feature_names)
        )


        self.assertEqual(
            sklToPmml.get_supportVectorMachine(sk_model)[0].get_Coefficients().get_absoluteValue(),
            23.0
        )


    def test_allTreeModels(self):
        model1=GradientBoostingClassifier()
        model2=GradientBoostingRegressor()
        model3 = DecisionTreeClassifier()



        sk_model1,feature_names1,target_name1=iris_dataset_for_classification(model1)
        sk_model2,feature_names2,target_name2=auto_dataset_for_regression(model2)
        sk_model3, feature_names3, target_name3 = iris_dataset_for_classification(model3)
        mining_imp_val=()
        categoric_values=()

        self.assertEqual(
            sklToPmml.get_node(sk_model3, feature_names3).__class__.__name__,
            pml.Node().__class__.__name__
        )

        self.assertEqual(
            sklToPmml.get_ensemble_models(
                sk_model1,feature_names1,feature_names1,target_name1,mining_imp_val,categoric_values)[0].get_functionName(),
            'classification'

        )

        self.assertEqual(
            len(sklToPmml.get_ensemble_models(
                sk_model1, feature_names1, feature_names1, target_name1, mining_imp_val,categoric_values)[
                    0].get_MiningSchema().get_MiningField()),
            len(feature_names1)+1

        )

        self.assertEqual(
            len(sklToPmml.get_ensemble_models(
                sk_model1, feature_names1, feature_names1, target_name1, mining_imp_val,categoric_values)[
                    0].get_MiningSchema().get_MiningField()),
            len(feature_names1) + 1

        )

        self.assertEqual(
            sklToPmml.get_ensemble_models(
                sk_model1, feature_names1, feature_names1, target_name1, mining_imp_val,categoric_values)[
                    0].get_Segmentation().get_multipleModelMethod(),
            'modelChain'

        )

        self.assertEqual(
            len(sklToPmml.get_ensemble_models(
                sk_model1, feature_names1, feature_names1, target_name1, mining_imp_val,categoric_values)[
                0].get_Segmentation().get_Segment()),
            len(model1.classes_)+1

        )

        self.assertEqual(
            sklToPmml.get_outer_segmentation(sk_model2,feature_names2,feature_names2,target_name2,mining_imp_val,categoric_values).get_multipleModelMethod(),
           'sum'

        )

        self.assertEqual(
            len(sklToPmml.get_inner_segments(sk_model2, feature_names2,feature_names2,0)),
            model2.n_estimators
        )

        self.assertEqual(
            len(sklToPmml.get_segments_for_gbc(sk_model1, feature_names1, feature_names1, target_name1,
                                             mining_imp_val,categoric_values)),
            len(sk_model1.classes_)+1
        )



def iris_dataset_for_classification(sk_model):
    df = pd.read_csv('iris.csv')
    feature_names = df.columns.drop('species')
    feature_names = feature_names._data
    target_name = 'species'
    x_train, x_test, y_train, y_test = train_test_split(df[feature_names], df[target_name], test_size=0.33,
                                                        random_state=101)
    sk_model=sk_model.fit(x_train,y_train)
    return sk_model,feature_names,target_name

def auto_dataset_for_regression(sk_model):
    df = pd.read_csv('auto-mpg.csv')
    X = df.drop(['mpg','car name'], axis=1)
    y = df['mpg']
    feature_names = [name for name in df.columns if name not in ('mpg','car name')]
    target_name='mpg'
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=101)
    sk_model=sk_model.fit(x_train,y_train)
    return sk_model,feature_names,target_name


if __name__=='__main__':
    unittest.main(warnings='ignore')







