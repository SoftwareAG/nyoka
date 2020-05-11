import PMML44 as pml
from base.enums import *
import json

_IS_INSIDE_BOUNDARY = "is_inside_boundary"
_CALCULATE_DISTANCE = "calculate_distance"
_TAG = "Tag"
_TAG_LOWER_BOUNDARY = "Tag_lower_boundary"
_TAG_UPPER_BOUNDARY = "Tag_upper_boundary"
_SUM_OF_DISTANCE = "sum_of_distance"
_PREDICTED_OUTPUT = "predicted_Result"
_TRANSFORMED_OUTPUT = "Percentage"
_UNDERSCORE = "_"
_DERIVED = "distance"
_CONSTANT = "Constant"
_JSON = ".json"
_MAX_VALUE = "maxValue"
_MIN_VALUE = "minValue"
_APPLICATION_NAME = "Nyoka"
_APPLICATION_VERSION = "4.2.1"


class FingerprintToPmml:

    def __init__(self, fingerprint, use_lag=False, pmml_file_name="from_fingerprint.pmml"):
        """
        Converts a fingerprint into PMML

        params
        ------

        fingerprint : str or json

            - the fingerprint content or the path of the file

        use_lag : boolean

            - If True, there will be only one tag and the remaining are generated using Lag.
              If Falses, all the Tags will be present.

        pmml_file_name : str

            - name of the PMML

        -----
        Writes the generated PMML object into given `pmml_file_name`
        """
        if fingerprint.__class__.__name__ == 'str' and fingerprint.endswith(_JSON):
            self.content = json.load(open(fingerprint, "r"))
        elif fingerprint.__class__.__name__ == 'dict':
            self.content = fingerprint
        else:
            raise ValueError("Invalid value for `fingerprint`. Should be a json object or path to a json file.")
        self._use_lag = use_lag
        self.pmml_file_name = pmml_file_name
        self._extract_info()
        self._pmml_obj = self._generate_pmml()
        self._pmml_obj.export(open(pmml_file_name, "w"), 0)

    def _extract_info(self):
        self._fingerprint_name = self.content['name']
        self._fingerprint_description = self.content['description'] if self.content[
                                                                           'description'] != "" else "Fingerprint in PMML"
        self._hulls = self.content['data']['hulls']
        self._length_of_fingerprint = len(self._hulls[0]['values'])
        self._detection_threshold = self.content["data"]["detectionThreshold"]
        self._max_distances = []
        for hull in self._hulls:
            max_value = hull["values"][0]["maxValue"]
            min_value = hull["values"][0]["minValue"]
            for tag in hull["values"][1:]:
                if tag["maxValue"] > max_value:
                    max_value = tag["maxValue"]
                if tag["minValue"] < min_value:
                    min_value = tag["minValue"]
            range_of_fp = abs(max_value - min_value)
            self._max_distances.append(range_of_fp * len(hull["values"]))

    def _generate_pmml(self):

        def get_header():
            header = pml.Header(
                Application=pml.Application(
                    name=_APPLICATION_NAME,
                    version=_APPLICATION_VERSION
                ),
                description=self._fingerprint_description
            )
            return header

        def get_data_dictionary():
            data_fields = []
            if not self._use_lag:
                for i in range(self._length_of_fingerprint):
                    data_fields.append(
                        pml.DataField(
                            name=_TAG + _UNDERSCORE + str(i),
                            optype=OPTYPE.CONTINUOUS.value,
                            dataType=DATATYPE.DOUBLE.value
                        )
                    )
            else:
                data_fields.append(
                    pml.DataField(
                        name=_TAG,
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value
                    )
                )
            data_dict = pml.DataDictionary(
                numberOfFields=len(data_fields),
                DataField=data_fields
            )
            return data_dict

        def get_is_inside_boundary_function():
            is_inside_boundary = pml.DefineFunction(
                name=_IS_INSIDE_BOUNDARY,
                optype=OPTYPE.CATEGORICAL.value,
                dataType=DATATYPE.BOOLEAN.value,
                ParameterField=[
                    pml.ParameterField(
                        name=_TAG,
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value
                    ),
                    pml.ParameterField(
                        name=_TAG_UPPER_BOUNDARY,
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value
                    ),
                    pml.ParameterField(
                        name=_TAG_LOWER_BOUNDARY,
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value
                    )
                ],
                Apply=pml.Apply(
                    function=FUNCTION.AND.value,
                    Apply_member=[
                        pml.Apply(
                            function=FUNCTION.GREATER_THAN.value,
                            FieldRef=[
                                pml.FieldRef(field=_TAG),
                                pml.FieldRef(field=_TAG_LOWER_BOUNDARY)
                            ]
                        ),
                        pml.Apply(
                            function=FUNCTION.LESS_OR_EQUAL.value,
                            FieldRef=[
                                pml.FieldRef(field=_TAG),
                                pml.FieldRef(field=_TAG_UPPER_BOUNDARY)
                            ]
                        )
                    ]
                )
            )
            return is_inside_boundary

        def get_calculate_distance_function():

            value_for_true = pml.Constant(valueOf_=0)
            value_for_true.original_tagname_ = _CONSTANT

            calculate_distance = pml.DefineFunction(
                name=_CALCULATE_DISTANCE,
                optype=OPTYPE.CONTINUOUS.value,
                dataType=DATATYPE.DOUBLE.value,
                ParameterField=[
                    pml.ParameterField(
                        name=_TAG,
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value
                    ),
                    pml.ParameterField(
                        name=_TAG_UPPER_BOUNDARY,
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value
                    ),
                    pml.ParameterField(
                        name=_TAG_LOWER_BOUNDARY,
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value
                    )
                ],
                Apply=pml.Apply(
                    function=FUNCTION.IF.value,
                    Apply_member=[
                        pml.Apply(
                            function=_IS_INSIDE_BOUNDARY,
                            FieldRef=[
                                pml.FieldRef(field=_TAG),
                                pml.FieldRef(field=_TAG_UPPER_BOUNDARY),
                                pml.FieldRef(field=_TAG_LOWER_BOUNDARY)
                            ]
                        ),
                        value_for_true,
                        pml.Apply(
                            function=FUNCTION.IF.value,
                            Apply_member=[
                                pml.Apply(
                                    function=FUNCTION.LESS_OR_EQUAL.value,
                                    FieldRef=[
                                        pml.FieldRef(field=_TAG),
                                        pml.FieldRef(field=_TAG_LOWER_BOUNDARY)
                                    ]
                                ),
                                pml.Apply(
                                    function=FUNCTION.SUBSTRACTTION.value,
                                    FieldRef=[
                                        pml.FieldRef(field=_TAG_LOWER_BOUNDARY),
                                        pml.FieldRef(field=_TAG)
                                    ]
                                ),
                                pml.Apply(
                                    function=FUNCTION.SUBSTRACTTION.value,
                                    FieldRef=[
                                        pml.FieldRef(field=_TAG),
                                        pml.FieldRef(field=_TAG_UPPER_BOUNDARY)
                                    ]
                                )
                            ]
                        )
                    ]
                )
            )
            return calculate_distance

        def get_lagged_fields():
            derived_fields = []
            for idx in range(1, self._length_of_fingerprint):
                name = _TAG + _UNDERSCORE + str(idx - 1)
                derived_fields.append(
                    pml.DerivedField(
                        name=name,
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value,
                        Lag=pml.Lag(field=_TAG, n=self._length_of_fingerprint - idx)
                    )
                )
            last_derived_name = _TAG + _UNDERSCORE + str(self._length_of_fingerprint - 1)
            derived_fields.append(
                pml.DerivedField(
                    name=last_derived_name,
                    optype=OPTYPE.CONTINUOUS.value,
                    dataType=DATATYPE.DOUBLE.value,
                    FieldRef=pml.FieldRef(field=_TAG)
                )
            )
            return derived_fields

        def get_transformation_dictionary():
            trans_dict = pml.TransformationDictionary(
                DefineFunction=[
                    get_is_inside_boundary_function(),
                    get_calculate_distance_function()
                ],
                DerivedField=get_lagged_fields() if self._use_lag else []
            )
            return trans_dict

        def get_mining_fields(field_names):
            mining_fields = []
            for field in field_names:
                mining_fields.append(
                    pml.MiningField(
                        name=field
                    )
                )
            return mining_fields

        def get_normalization_function(index):
            constant_100 = pml.Constant(valueOf_=100)
            constant_100.original_tagname_ = "Constant"

            constant_max_distance = pml.Constant(valueOf_=self._max_distances[index])
            constant_max_distance.original_tagname_ = "Constant"

            substraction_function = pml.Apply(
                function=FUNCTION.MULTIPLICATION.value,
                Apply_member=[
                    pml.Apply(
                        function=FUNCTION.DIVISION.value,
                        Apply_member=[
                            pml.Apply(
                                function=FUNCTION.SUBSTRACTTION.value,
                                FieldRef=[
                                    constant_max_distance,
                                    pml.FieldRef(field="absolute_difference_" + str(index))
                                ]
                            )
                        ],
                        Constant=[
                            pml.Constant(valueOf_=self._max_distances[index])
                        ]
                    ),
                    constant_100
                ]
            )
            substraction_function.original_tagname_ = "Apply"

            equal_function = pml.Apply(
                function=FUNCTION.IF.value,
                Apply_member=[
                    pml.Apply(
                        function=FUNCTION.EQUAL.value,
                        FieldRef=[
                            pml.FieldRef(field="absolute_difference_" + str(index))
                        ],
                        Constant=[
                            pml.Constant(valueOf_=0)
                        ]
                    )
                ],
                Constant=[
                    pml.Constant(valueOf_=100),
                    substraction_function,
                ]
            )
            equal_function.original_tagname_ = "Apply"

            return pml.Apply(
                function=FUNCTION.IF.value,
                Apply_member=[
                    pml.Apply(
                        function=FUNCTION.GREATER_OR_EQUAL.value,
                        FieldRef=[
                            pml.FieldRef(field="absolute_difference_" + str(index))
                        ],
                        Constant=[
                            pml.Constant(valueOf_=self._max_distances[index])
                        ]
                    )
                ],
                Constant=[
                    pml.Constant(valueOf_=0),
                    equal_function
                ]
            )

        def get_output_for_mining_model():
            output_fields = []
            for idx, hull in enumerate(self._hulls):
                output_fields.extend([
                    pml.OutputField(
                        name="absolute_difference_" + str(idx),
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value,
                        feature=RESULT_FEATURE.PREDICTED_VALUE.value,
                        segmentId=str(idx)
                    ),
                    pml.OutputField(
                        name="normalized_score_" + str(idx),
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value,
                        feature=RESULT_FEATURE.TRANSFORMED_VALUE.value,
                        Apply=get_normalization_function(idx)
                    ),
                    pml.OutputField(
                        name="is_matched_" + str(idx),
                        optype=OPTYPE.CATEGORICAL.value,
                        dataType=DATATYPE.BOOLEAN.value,
                        feature=RESULT_FEATURE.TRANSFORMED_VALUE.value,
                        Apply=pml.Apply(
                            function=FUNCTION.GREATER_OR_EQUAL.value,
                            FieldRef=[
                                pml.FieldRef(field="normalized_score_" + str(idx)),
                            ],
                            Constant=[
                                pml.Constant(valueOf_=self._detection_threshold)
                            ]
                        )
                    )
                ])
            return pml.Output(OutputField=output_fields)

        def get_local_transformation(index):
            derived_fields = []
            derived_field_names = []
            hull = self._hulls[index]
            for idx, val in enumerate(hull["values"]):
                name = "distance_tag_" + str(idx)
                derived_field_names.append(name)
                derived_fields.append(
                    pml.DerivedField(
                        name=name,
                        optype=OPTYPE.CONTINUOUS.value,
                        dataType=DATATYPE.DOUBLE.value,
                        Apply=pml.Apply(
                            function=_CALCULATE_DISTANCE,
                            FieldRef=[pml.FieldRef(field=_TAG + _UNDERSCORE + str(idx))],
                            Constant=[
                                pml.Constant(valueOf_=val["maxValue"]),
                                pml.Constant(valueOf_=val["minValue"])
                            ]
                        )
                    )
                )
            derived_fields.append(
                pml.DerivedField(
                    name=_SUM_OF_DISTANCE,
                    optype=OPTYPE.CONTINUOUS.value,
                    dataType=DATATYPE.DOUBLE.value,
                    Apply=pml.Apply(
                        function=FUNCTION.SUM.value,
                        FieldRef=[
                            pml.FieldRef(
                                field=field
                            )
                            for field in derived_field_names
                        ]
                    )
                )
            )
            return pml.LocalTransformations(DerivedField=derived_fields)

        def get_output_for_regression_model(idx):
            output_fields = [
                pml.OutputField(
                    name="predicted_result_" + str(idx),
                    optype=OPTYPE.CONTINUOUS.value,
                    dataType=DATATYPE.DOUBLE.value
                )
            ]
            return pml.Output(OutputField=output_fields)

        def get_segments():
            segments = []
            for idx, hull in enumerate(self._hulls):
                segments.append(
                    pml.Segment(
                        id=str(idx),
                        True_=pml.True_(),
                        RegressionModel=pml.RegressionModel(
                            functionName=MINING_FUNCTION.REGRESSION.value,
                            MiningSchema=pml.MiningSchema(
                                MiningField=[pml.MiningField(name=_TAG)] if self._use_lag else
                                get_mining_fields(
                                    [_TAG + _UNDERSCORE + str(i) for i in range(self._length_of_fingerprint)])
                            ),
                            Output=get_output_for_regression_model(idx),
                            LocalTransformations=get_local_transformation(idx),
                            RegressionTable=[
                                pml.RegressionTable(
                                    intercept=0,
                                    NumericPredictor=[
                                        pml.NumericPredictor(name=_SUM_OF_DISTANCE, coefficient=1)
                                    ]
                                )
                            ]
                        )
                    )
                )
            return segments

        def get_mining_model():
            output = get_output_for_mining_model()
            mining_model = pml.MiningModel(
                functionName=MINING_FUNCTION.REGRESSION.value,
                modelName=self._fingerprint_name,
                MiningSchema=pml.MiningSchema(
                    MiningField=[pml.MiningField(name=_TAG)] if self._use_lag else
                    get_mining_fields([_TAG + _UNDERSCORE + str(i) for i in range(self._length_of_fingerprint)])
                ),
                Output=output,
                Segmentation=pml.Segmentation(
                    multipleModelMethod=MULTIPLE_MODEL_METHOD.MAX.value,
                    Segment=get_segments()
                )
            )
            return mining_model

        header = get_header()
        data_dict = get_data_dictionary()
        trans_dict = get_transformation_dictionary()
        mining_model = get_mining_model()
        pmml = pml.PMML(
            version=PMML_SCHEMA.VERSION.value,
            Header=header,
            DataDictionary=data_dict,
            TransformationDictionary=trans_dict,
            MiningModel=[mining_model]
        )
        return pmml
