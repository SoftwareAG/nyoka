def parse(inFileName, silence=False):
    orig_init()
    result = parseSub(inFileName, silence)
    new_init()
    
    return result

def new_init():
    def LayerWeights_init(self, weightsShape=None, weightsFlattenAxis=None, content=None, floatType="float32", floatsPerLine=12, src=None, Extension=None, mixedclass_=None):
        self.original_tagname_ = None
        self.weightsShape = supermod._cast(None, weightsShape)
        self.weightsFlattenAxis = supermod._cast(None, weightsFlattenAxis)
        self.src = supermod._cast(None, src)
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        validFloatTypes = ["float6", "float7", "float8", "float16", "float32", "float64"]
        if floatType not in validFloatTypes:
            floatType = "float32"
        from nyoka.Base64 import FloatBase64
        base64string = "\t\t\t\t" + "data:" + floatType + ";base64," + FloatBase64.from_floatArray(content, floatsPerLine)
        base64string = base64string.replace("\n", "\n\t\t\t\t")
        self.content_ = [supermod.MixedContainer(1, 2, "", base64string)]
        self.valueOf_ = base64string

    def LayerRecurrentWeights_init(self, recurrentWeightsShape=None, recurrentWeightsFlattenAxis=None, content=None, floatType="float32", floatsPerLine=12, src=None, Extension=None, mixedclass_=None):
        self.original_tagname_ = None
        self.recurrentWeightsShape = supermod._cast(None, recurrentWeightsShape)
        self.recurrentWeightsFlattenAxis = supermod._cast(None, recurrentWeightsFlattenAxis)
        self.src = supermod._cast(None, src)
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        validFloatTypes = ["float6", "float7", "float8", "float16", "float32", "float64"]
        if floatType not in validFloatTypes:
            floatType = "float32"
        from nyoka.Base64 import FloatBase64
        base64string = "\t\t\t\t" + "data:" + floatType + ";base64," + FloatBase64.from_floatArray(content, floatsPerLine)
        base64string = base64string.replace("\n", "\n\t\t\t\t")
        self.content_ = [supermod.MixedContainer(1, 2, "", base64string)]
        self.valueOf_ = base64string

    def LayerBias_init(self, biasShape=None, biasFlattenAxis=None, content=None, floatType="float32", floatsPerLine=12, src=None, Extension=None, mixedclass_=None):
        self.original_tagname_ = None
        self.biasShape = supermod._cast(None, biasShape)
        self.biasFlattenAxis = supermod._cast(None, biasFlattenAxis)
        self.src = supermod._cast(None, src)
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        validFloatTypes = ["float6", "float7", "float8", "float16", "float32", "float64"]
        if floatType not in validFloatTypes:
            floatType = "float32"
        from nyoka.Base64 import FloatBase64
        base64string = "\t\t\t\t" + "data:" + floatType + ";base64," + FloatBase64.from_floatArray(content, floatsPerLine)
        base64string = base64string.replace("\n", "\n\t\t\t\t")
        self.content_ = [supermod.MixedContainer(1, 2, "", base64string)]
        self.valueOf_ = base64string

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

    def PMML_init(self, version='4.3', Header=None,Data=None, script=None, MiningBuildTask=None, DataDictionary=None, TransformationDictionary=None, AssociationModel=None, AnomalyDetectionModel=None, BayesianNetworkModel=None, BaselineModel=None, ClusteringModel=None, DeepNetwork=None, GaussianProcessModel=None, GeneralRegressionModel=None, MiningModel=None, NaiveBayesModel=None, NearestNeighborModel=None, NeuralNetwork=None, RegressionModel=None, RuleSetModel=None, SequenceModel=None, Scorecard=None, SupportVectorMachineModel=None, TextModel=None, TimeSeriesModel=None, TreeModel=None, Extension=None):
        self.original_tagname_ = None
        self.version = supermod._cast(None, version)
        self.Header = Header
        if script is None:
            self.script = []
        else:
            self.script = script

        if Data is None:
            self.Data=[]
        else:
            print ('print',Data)
            self.Data=Data
        self.MiningBuildTask = MiningBuildTask
        self.DataDictionary = DataDictionary
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
        if DeepNetwork is None:
            self.DeepNetwork = []
        else:
            self.DeepNetwork = DeepNetwork
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
        if TransformationDictionary is None:
            self.TransformationDictionary = []
        else:
            self.TransformationDictionary = TransformationDictionary
        if TreeModel is None:
            self.TreeModel = []
        else:
            self.TreeModel = TreeModel
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension

    def script_init(self, content=None, for_=None, class_=None,scriptPurpose=None, Extension=None):
        self.original_tagname_ = None
        self.for_ = supermod._cast(None, for_)
        self.class_ = supermod._cast(None, class_)
        self.scriptPurpose = supermod._cast(None, scriptPurpose)
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        self.anyAttributes_ = {}
        self.mixedclass_ = supermod.MixedContainer
        self.content_ = [supermod.MixedContainer(1, 2, "", str(content))]
        self.valueOf_ = str(content)

    LayerWeights.__init__ = LayerWeights_init
    LayerRecurrentWeights.__init__=LayerRecurrentWeights_init
    LayerBias.__init__ = LayerBias_init
    ArrayType.__init__ = ArrayType_init
    Annotation.__init__ = Annotation_init
    Timestamp.__init__ = Timestamp_init
    PMML.__init__ = PMML_init
    script.__init__ = script_init

def orig_init():
    def LayerWeights_init(self, weightsShape=None, weightsFlattenAxis=None, src=None, Extension=None, valueOf_=None, mixedclass_=None, content_=None):
        self.original_tagname_ = None
        self.weightsShape = supermod._cast(None, weightsShape)
        self.weightsFlattenAxis = supermod._cast(None, weightsFlattenAxis)
        self.src = supermod._cast(None, src)
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
        
    def LayerRecurrentWeights_init(self, recurrentWeightsShape=None, recurrentWeightsFlattenAxis=None, src=None, Extension=None, valueOf_=None, mixedclass_=None, content_=None):
        self.original_tagname_ = None
        self.recurrentWeightsShape = supermod._cast(None, recurrentWeightsShape)
        self.recurrentWeightsFlattenAxis = supermod._cast(None, recurrentWeightsFlattenAxis)
        self.src = supermod._cast(None, src)
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_
        self.valueOf_ = valueOf_

    def LayerBias_init(self, biasShape=None, biasFlattenAxis=None, src=None, Extension=None, valueOf_=None, mixedclass_=None, content_=None):
        self.original_tagname_ = None
        self.biasShape = supermod._cast(None, biasShape)
        self.biasFlattenAxis = supermod._cast(None, biasFlattenAxis)
        self.src = supermod._cast(None, src)
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

    def PMML_init(self, version=None, Header=None,Data=None, script=None, MiningBuildTask=None, DataDictionary=None, TransformationDictionary=None, AssociationModel=None, AnomalyDetectionModel=None, BayesianNetworkModel=None, BaselineModel=None, ClusteringModel=None, DeepNetwork=None, GaussianProcessModel=None, GeneralRegressionModel=None, MiningModel=None, NaiveBayesModel=None, NearestNeighborModel=None, NeuralNetwork=None, RegressionModel=None, RuleSetModel=None, SequenceModel=None, Scorecard=None, SupportVectorMachineModel=None, TextModel=None, TimeSeriesModel=None, TreeModel=None, Extension=None):
        self.original_tagname_ = None
        self.version = supermod._cast(None, version)
        self.Header = Header
        if script is None:
            self.script = []
        else:
            self.script = script
        if Data is None:
            self.Data = []
        else:
            self.Data = Data
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
        if DeepNetwork is None:
            self.DeepNetwork = []
        else:
            self.DeepNetwork = DeepNetwork
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
        if TransformationDictionary is None:
                self.TransformationDictionary = []
        else:
            self.TransformationDictionary = TransformationDictionary
        if TreeModel is None:
            self.TreeModel = []
        else:
            self.TreeModel = TreeModel
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension

    def script_init(self, for_=None, class_=None,scriptPurpose=None, Extension=None, valueOf_=None, mixedclass_=None, content_=None):
        self.original_tagname_ = None
        self.for_ = supermod._cast(None, for_)
        self.class_ = supermod._cast(None, class_)
        self.scriptPurpose = supermod._cast(None, scriptPurpose)
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = Extension
        self.valueOf_ = valueOf_
        self.anyAttributes_ = {}
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_
        self.valueOf_ = valueOf_

    LayerWeights.__init__ = LayerWeights_init
    LayerRecurrentWeights.__init__ = LayerRecurrentWeights_init
    LayerBias.__init__ = LayerBias_init
    ArrayType.__init__ = ArrayType_init
    Annotation.__init__ = Annotation_init
    Timestamp.__init__ = Timestamp_init
    PMML.__init__ = PMML_init
    script.__init__ = script_init

new_init()

def showIndent(outfile, level, pretty_print=True):
    if pretty_print:
        for idx in range(level):
            outfile.write('\t')
