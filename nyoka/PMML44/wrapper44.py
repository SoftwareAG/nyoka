def parse(inFileName, silence=False):
    orig_init()
    result = parseSub(inFileName, silence)
    new_init()
    
    return result

def new_init():

    def ArrayType_init(self, content=None, n=None, type_=None, mixedclass_=None):
        self.original_tagname_ = None
        self.n = supermod._cast(None, n)
        self.type_ = supermod._cast(None, type_)
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        self.content_ = [supermod.MixedContainer(1, 2, "", str(content))]
        self.valueOf_ = str(content)

    def Annotation_init(self, content=None, Extension=None, mixedclass_=None):
        self.original_tagname_ = None
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        self.content_ = [supermod.MixedContainer(1, 2, "", str(content))]
        self.valueOf_ = str(content)
    
    def Timestamp_init(self, content=None, Extension=None, mixedclass_=None):
        self.original_tagname_ = None
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        self.content_ = [supermod.MixedContainer(1, 2, "", str(content))]
        self.valueOf_ = str(content)

    def PMML_init(self, version='4.4', Header=None, MiningBuildTask=None, DataDictionary=None, TransformationDictionary=None, AssociationModel=None, AnomalyDetectionModel=None, BayesianNetworkModel=None, BaselineModel=None, ClusteringModel=None, GaussianProcessModel=None, GeneralRegressionModel=None, MiningModel=None, NaiveBayesModel=None, NearestNeighborModel=None, NeuralNetwork=None, RegressionModel=None, RuleSetModel=None, SequenceModel=None, Scorecard=None, SupportVectorMachineModel=None, TextModel=None, TimeSeriesModel=None, TreeModel=None, Extension=None):
        self.original_tagname_ = None
        self.version = supermod._cast(None, version)
        self.Header = Header
        self.MiningBuildTask = MiningBuildTask
        self.DataDictionary = DataDictionary
        self.TransformationDictionary = TransformationDictionary
        if AssociationModel is None:
            self.AssociationModel = []
        else:
            self.AssociationModel = AssociationModel
        if AnomalyDetectionModel is None:
                self.AnomalyDetectionModel = []
        else:
            self.AnomalyDetectionModel = AnomalyDetectionModel
        if BayesianNetworkModel is None:
            self.BayesianNetworkModel = []
        else:
            self.BayesianNetworkModel = BayesianNetworkModel
        if BaselineModel is None:
            self.BaselineModel = []
        else:
            self.BaselineModel = BaselineModel
        if ClusteringModel is None:
            self.ClusteringModel = []
        else:
            self.ClusteringModel = ClusteringModel
        if GaussianProcessModel is None:
            self.GaussianProcessModel = []
        else:
            self.GaussianProcessModel = GaussianProcessModel
        if GeneralRegressionModel is None:
            self.GeneralRegressionModel = []
        else:
            self.GeneralRegressionModel = GeneralRegressionModel
        if MiningModel is None:
            self.MiningModel = []
        else:
            self.MiningModel = MiningModel
        if NaiveBayesModel is None:
            self.NaiveBayesModel = []
        else:
            self.NaiveBayesModel = NaiveBayesModel
        if NearestNeighborModel is None:
            self.NearestNeighborModel = []
        else:
            self.NearestNeighborModel = NearestNeighborModel
        if NeuralNetwork is None:
            self.NeuralNetwork = []
        else:
            self.NeuralNetwork = NeuralNetwork
        if RegressionModel is None:
            self.RegressionModel = []
        else:
            self.RegressionModel = RegressionModel
        if RuleSetModel is None:
            self.RuleSetModel = []
        else:
            self.RuleSetModel = RuleSetModel
        if SequenceModel is None:
            self.SequenceModel = []
        else:
            self.SequenceModel = SequenceModel
        if Scorecard is None:
            self.Scorecard = []
        else:
            self.Scorecard = Scorecard
        if SupportVectorMachineModel is None:
            self.SupportVectorMachineModel = []
        else:
            self.SupportVectorMachineModel = SupportVectorMachineModel
        if TextModel is None:
            self.TextModel = []
        else:
            self.TextModel = TextModel
        if TimeSeriesModel is None:
            self.TimeSeriesModel = []
        else:
            self.TimeSeriesModel = TimeSeriesModel
        if TreeModel is None:
            self.TreeModel = []
        else:
            self.TreeModel = TreeModel
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension

    ArrayType.__init__ = ArrayType_init
    Annotation.__init__ = Annotation_init
    Timestamp.__init__ = Timestamp_init
    PMML.__init__ = PMML_init

def orig_init():

    def ArrayType_init(self, n=None, type_=None, valueOf_=None, mixedclass_=None, content_=None):
        self.original_tagname_ = None
        self.n = supermod._cast(None, n)
        self.type_ = supermod._cast(None, type_)
        self.valueOf_ = valueOf_
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_
        self.valueOf_ = valueOf_

    def Annotation_init(self, Extension=None, valueOf_=None, mixedclass_=None, content_=None):
        self.original_tagname_ = None
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        self.valueOf_ = valueOf_
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_
        self.valueOf_ = valueOf_

    def Timestamp_init(self, Extension=None, valueOf_=None, mixedclass_=None, content_=None):
        self.original_tagname_ = None
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        self.valueOf_ = valueOf_
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_
        self.valueOf_ = valueOf_

    def PMML_init(self, version=None, Header=None, MiningBuildTask=None, DataDictionary=None, TransformationDictionary=None, AssociationModel=None, AnomalyDetectionModel=None, BayesianNetworkModel=None, BaselineModel=None, ClusteringModel=None, GaussianProcessModel=None, GeneralRegressionModel=None, MiningModel=None, NaiveBayesModel=None, NearestNeighborModel=None, NeuralNetwork=None, RegressionModel=None, RuleSetModel=None, SequenceModel=None, Scorecard=None, SupportVectorMachineModel=None, TextModel=None, TimeSeriesModel=None, TreeModel=None, Extension=None):
        self.original_tagname_ = None
        self.version = supermod._cast(None, version)
        self.Header = Header
        self.MiningBuildTask = MiningBuildTask
        self.DataDictionary = DataDictionary
        self.TransformationDictionary = TransformationDictionary
        if AssociationModel is None:
            self.AssociationModel = []
        else:
            self.AssociationModel = AssociationModel
        if AnomalyDetectionModel is None:
                self.AnomalyDetectionModel = []
        else:
            self.AnomalyDetectionModel = AnomalyDetectionModel
        if BayesianNetworkModel is None:
            self.BayesianNetworkModel = []
        else:
            self.BayesianNetworkModel = BayesianNetworkModel
        if BaselineModel is None:
            self.BaselineModel = []
        else:
            self.BaselineModel = BaselineModel
        if ClusteringModel is None:
            self.ClusteringModel = []
        else:
            self.ClusteringModel = ClusteringModel
        if GaussianProcessModel is None:
            self.GaussianProcessModel = []
        else:
            self.GaussianProcessModel = GaussianProcessModel
        if GeneralRegressionModel is None:
            self.GeneralRegressionModel = []
        else:
            self.GeneralRegressionModel = GeneralRegressionModel
        if MiningModel is None:
            self.MiningModel = []
        else:
            self.MiningModel = MiningModel
        if NaiveBayesModel is None:
            self.NaiveBayesModel = []
        else:
            self.NaiveBayesModel = NaiveBayesModel
        if NearestNeighborModel is None:
            self.NearestNeighborModel = []
        else:
            self.NearestNeighborModel = NearestNeighborModel
        if NeuralNetwork is None:
            self.NeuralNetwork = []
        else:
            self.NeuralNetwork = NeuralNetwork
        if RegressionModel is None:
            self.RegressionModel = []
        else:
            self.RegressionModel = RegressionModel
        if RuleSetModel is None:
            self.RuleSetModel = []
        else:
            self.RuleSetModel = RuleSetModel
        if SequenceModel is None:
            self.SequenceModel = []
        else:
            self.SequenceModel = SequenceModel
        if Scorecard is None:
            self.Scorecard = []
        else:
            self.Scorecard = Scorecard
        if SupportVectorMachineModel is None:
            self.SupportVectorMachineModel = []
        else:
            self.SupportVectorMachineModel = SupportVectorMachineModel
        if TextModel is None:
            self.TextModel = []
        else:
            self.TextModel = TextModel
        if TimeSeriesModel is None:
            self.TimeSeriesModel = []
        else:
            self.TimeSeriesModel = TimeSeriesModel
        if TreeModel is None:
            self.TreeModel = []
        else:
            self.TreeModel = TreeModel
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension

    ArrayType.__init__ = ArrayType_init
    Annotation.__init__ = Annotation_init
    Timestamp.__init__ = Timestamp_init
    PMML.__init__ = PMML_init

new_init()

def showIndent(outfile, level, pretty_print=True):
    if pretty_print:
        for idx in range(level):
            outfile.write('\t')
