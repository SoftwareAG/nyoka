import nyoka.PMML44 as pml
from nyoka.base.constants import *
import itertools
import numpy as np
import pandas as pd

class BkdeToPmml:

    def __init__(self, model, model_name, description=None, pmml_file_name="from_bkde.pmml"):
        self.model = model
        self.model_name = model_name
        self.description = description
        self.pmml_file_name = pmml_file_name
        pmml_obj = self._generate_pmml()
        pmml_obj.export(open(pmml_file_name,"w"),0)

    def _generate_pmml(self):

        def get_header():
            return pml.Header(
                Application=pml.Application(
                    name=HEADER_INFO.APPLICATION_NAME,
                    version=HEADER_INFO.APPLICATION_VERSION
                ),
                description=self.description or "BKDE in PMML"
            )

        def get_data_dictionary():
            data_fields = []
            for name in self.model.names:
                data_fields.append(
                    pml.DataField(
                        name=name,
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE
                    )
                )
            return pml.DataDictionary(
                numberOfFields=len(data_fields),
                DataField=data_fields
            )

        def get_transformation_dictionary():

            def mk_breaks(u):
                return u - (np.max(u) - np.min(u)) / (u.size - 1) / 2

            axes_map = {}
            combinations = list(itertools.combinations(list(self.model.names), 2))
            for combination in combinations:
                container = self.model.fingerprint_container[tuple(combination)]
                axes = container["axes"]
                if combination[0] not in axes_map:
                    axes_map[combination[0]] = pd.cut(np.ones(axes[0].shape), mk_breaks(axes[0]))
                if combination[1] not in axes_map:
                    axes_map[combination[1]] = pd.cut(np.ones(axes[0].shape), mk_breaks(axes[1]))
            der_flds = []
            for name in self.model.names:
                bins = []
                categories = list(axes_map[name].categories)
                for idx, interval in enumerate(categories):
                    dbin = pml.DiscretizeBin(
                        binValue=idx,
                        Interval=pml.Interval(closure="openClosed", leftMargin=interval.left,
                                              rightMargin=interval.right)
                    )
                    bins.append(dbin)
                bins.extend([
                    pml.DiscretizeBin(
                        binValue=len(categories),
                        Interval=pml.Interval(closure="openClosed", rightMargin=categories[0].left)
                    ),
                    pml.DiscretizeBin(
                        binValue=len(categories),
                        Interval=pml.Interval(closure="openOpen", leftMargin=categories[-1].right)
                    )
                ])
                der_flds.append(
                    pml.DerivedField(
                    name=f"{name}_bin",
                    optype=OPTYPE.CONTINUOUS,
                    dataType=DATATYPE.DOUBLE,
                    Discretize=pml.Discretize(
                        field=name,
                        dataType=DATATYPE.INTEGER,
                        DiscretizeBin=bins
                        )
                    )
                )
            return pml.TransformationDictionary(
                DerivedField=der_flds
            )

        def get_mining_schema(names):
            mining_flds = []
            for name in names:
                mining_flds.append(
                    pml.MiningField(
                        name=name,
                        usageType=FIELD_USAGE_TYPE.ACTIVE
                    )
                )
            return pml.MiningSchema(
                MiningField=mining_flds
            )

        def get_mining_model():

            def get_regression_model(combination):

                def get_rows(density_matrix):
                    rows = []
                    for i in range(density_matrix.shape[0]):
                        for j in range(density_matrix.shape[1]):
                            row_main = pml.row()
                            row_main.elementobjs_ = ['density', 'row', 'column']
                            row_main.density = str(density_matrix[i][j])
                            row_main.row = str(i)
                            row_main.column = str(j)
                            rows.append(row_main)
                    return rows

                def get_derived_fields(density_matrix):
                    density = pml.DerivedField(
                        name="density",
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE,
                        MapValues=pml.MapValues(
                            outputColumn="density",
                            dataType=DATATYPE.DOUBLE,
                            FieldColumnPair=[
                                pml.FieldColumnPair(
                                    field=f"{combination[0]}_bin",
                                    column="row"
                                ),
                                pml.FieldColumnPair(
                                    field=f"{combination[1]}_bin",
                                    column="column"
                                )
                            ],
                            InlineTable=pml.InlineTable(
                                row=get_rows(density_matrix)
                            )
                        )
                    )
                    log_n = pml.Apply(
                        function=FUNCTION.LOGN,
                        Apply_member=[
                            pml.Apply(
                                function=FUNCTION.DIVISION,
                                FieldRef=[
                                    pml.FieldRef(field="density")
                                ],
                                Constant=[
                                    pml.Constant(valueOf_=str(container["maximum_density"]), dataType=DATATYPE.DOUBLE)
                                ]
                            )
                        ]
                    )
                    log_n.original_tagname_ = "Apply"
                    anomaly_score = pml.DerivedField(
                        name="anomaly-score",
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE,
                        Apply=pml.Apply(
                            function=FUNCTION.MULTIPLICATION,
                            Constant=[
                                pml.Constant(valueOf_=str(-1), dataType=DATATYPE.INTEGER),
                                log_n
                            ]
                        )
                    )
                    thresholds = []
                    for val in container["anomaly_time_series"]:
                        cons = pml.Constant(valueOf_=str(val), dataType="double")
                        cons.original_tagname_ = "Constant"
                        thresholds.append(pml.Apply(
                            function="threshold",
                            FieldRef=[cons, pml.FieldRef(field="anomaly-score")]
                        ))
                    p_value = pml.DerivedField(
                        name=f"{combination[0]}{combination[1]}_p-value",
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE,
                        Apply=pml.Apply(
                            function=FUNCTION.DIVISION,
                            Apply_member=[
                                pml.Apply(
                                    function=FUNCTION.SUM,
                                    Apply_member=thresholds
                                )
                            ],
                            Constant=[
                                pml.Constant(dataType=DATATYPE.DOUBLE, valueOf_=str(len(thresholds)))
                            ]
                        )
                    )
                    return [density, anomaly_score, p_value]

                container = self.model.fingerprint_container[combination]
                density_matrix = container["fhat"]
                local_trans = pml.LocalTransformations(
                    DerivedField=get_derived_fields(density_matrix)
                )
                regression_tab = pml.RegressionTable(
                    intercept="0",
                    NumericPredictor=[
                        pml.NumericPredictor(
                            name=f"{combination[0]}{combination[1]}_p-value",
                            exponent="1",
                            coefficient="-1.0" if len(self.model.names) > 2 else "1.0"
                        )
                    ]
                )
                return pml.RegressionModel(
                    functionName=MINING_FUNCTION.REGRESSION,
                    MiningSchema=get_mining_schema(combination),
                    Output=pml.Output(
                        OutputField=[
                            pml.OutputField(
                                name="p-value",
                                optype=OPTYPE.CONTINUOUS,
                                dataType=DATATYPE.DOUBLE
                            )
                        ]
                    ),
                    LocalTransformations=local_trans,
                    RegressionTable=[regression_tab]
                )

            def get_segmentation():
                combinations = list(itertools.combinations(list(self.model.names), 2))
                segments = []
                for index, combination in enumerate(combinations):
                    reg_model = get_regression_model(combination)
                    segments.append(
                        pml.Segment(
                            id=str(index),
                            True_=pml.True_(),
                            RegressionModel=reg_model
                        )
                    )
                return pml.Segmentation(
                    multipleModelMethod=MULTIPLE_MODEL_METHOD.MAX,
                    Segment=segments
                )

            def get_output():
                def transform_output():
                    apply = pml.Apply(
                        function=FUNCTION.MULTIPLICATION,
                        FieldRef=[
                            pml.FieldRef(
                                field="predicted_p-value",
                            )
                        ],
                        Constant=[
                            pml.Constant(
                                dataType=DATATYPE.INTEGER,
                                valueOf_=str(-1)
                            )
                        ]
                    )
                    return apply

                output_flds = []
                is_multi = len(self.model.names) > 2
                if is_multi:
                    output_flds.append(
                        pml.OutputField(
                            name="predicted_p-value",
                            optype=OPTYPE.CONTINUOUS,
                            dataType=DATATYPE.DOUBLE,
                            feature=RESULT_FEATURE.PREDICTED_VALUE,
                            isFinalResult=False
                        )
                    )
                p_value = pml.OutputField(
                    name="p-value",
                    optype=OPTYPE.CONTINUOUS,
                    dataType=DATATYPE.DOUBLE,
                    feature=RESULT_FEATURE.TRANSFORMED_VALUE if is_multi else RESULT_FEATURE.PREDICTED_VALUE,
                    Apply=transform_output() if is_multi else None
                )
                is_anomaly = pml.OutputField(
                    name="is-anomaly",
                    optype=OPTYPE.CATEGORICAL,
                    dataType=DATATYPE.BOOLEAN,
                    feature=RESULT_FEATURE.TRANSFORMED_VALUE,
                    Apply=pml.Apply(
                        function=FUNCTION.LESS_THAN,
                        FieldRef=[
                            pml.FieldRef(field="p-value")
                        ],
                        Constant=[
                            pml.Constant(valueOf_=str(self.model.threshold))
                        ]
                    )
                )
                output_flds.extend([p_value, is_anomaly])
                return pml.Output(
                    OutputField=output_flds
                )

            mining_schema = get_mining_schema(self.model.names)
            output = get_output()
            segmentation = get_segmentation()
            return pml.MiningModel(
                functionName=MINING_FUNCTION.REGRESSION,
                modelName=self.model_name,
                MiningSchema=mining_schema,
                Output=output,
                Segmentation=segmentation
            )

        header = get_header()
        data_dict = get_data_dictionary()
        trans_dict = get_transformation_dictionary()
        mining_model = get_mining_model()
        return pml.PMML(
            version=PMML_SCHEMA.VERSION,
            Header=header,
            DataDictionary=data_dict,
            TransformationDictionary=trans_dict,
            MiningModel=[mining_model]
        )