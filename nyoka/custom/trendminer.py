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
import PMML44 as pml
from base.constants import *
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


class FingerprintToPmml:

    def __init__(self, fingerprint, model_name = None, use_lag=True, pmml_file_name="from_fingerprint.pmml"):
        """
        Converts a fingerprint into PMML

        params
        ------

        fingerprint : str or json

            - the fingerprint content or the path of the file

        model_name : str or None

            - the name of the model

        use_lag : boolean

            - If True, there will be only one tag for each hull and the remaining are generated using Lag.
              If False, all the tags will be present.

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
            raise ValueError("Invalid value for 'fingerprint'. Should be a dictionary object or path to a json file.")
        self._use_lag = use_lag
        self.pmml_file_name = pmml_file_name
        self._model_name = model_name
        self._extract_info()
        self._pmml_obj = self._generate_pmml()
        self._pmml_obj.export(open(pmml_file_name, "w"), 0)

    def _extract_info(self):
        if "data" not in self.content:
            raise AttributeError("Attribute 'data' not found in fingerprint.")
        self._fingerprint_name = self.content.get("name","fingerprint")
        self._fingerprint_description = self.content.get("description","Fingerprint in PMML")
        self._hulls = self.content["data"]["hulls"]
        self._length_of_fingerprint = len(self._hulls[0]["values"])
        # self._detection_threshold = self.content["data"]["detectionThreshold"]
        self._max_distances = []
        self._fp_ranges = []
        for hull in self._hulls:
            max_value = hull["values"][0]["maxValue"]
            min_value = hull["values"][0]["minValue"]
            for tag in hull["values"][1:]:
                if tag["maxValue"] > max_value:
                    max_value = tag["maxValue"]
                if tag["minValue"] < min_value:
                    min_value = tag["minValue"]
            range_of_fp = abs(max_value - min_value)
            self._fp_ranges.append(range_of_fp)
            self._max_distances.append(range_of_fp * len(hull["values"]))

    def _generate_pmml(self):

        def get_header():
            header = pml.Header(
                Application=pml.Application(
                    name=HEADER_INFO.APPLICATION_NAME,
                    version=HEADER_INFO.APPLICATION_VERSION
                ),
                description=self._fingerprint_description
            )
            return header

        def get_data_dictionary():
            data_fields = []
            if not self._use_lag:
                for i,hull in enumerate(self._hulls):
                    for j in range(self._length_of_fingerprint):
                        data_fields.append(
                            pml.DataField(
                                name=hull["name"] + _UNDERSCORE + str(j),
                                optype=OPTYPE.CONTINUOUS,
                                dataType=DATATYPE.DOUBLE
                            )
                        )
            else:
                for idx, hull in enumerate(self._hulls):
                    data_fields.append(
                        pml.DataField(
                            name=hull["name"],
                            optype=OPTYPE.CONTINUOUS,
                            dataType=DATATYPE.DOUBLE
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
                optype=OPTYPE.CATEGORICAL,
                dataType=DATATYPE.BOOLEAN,
                ParameterField=[
                    pml.ParameterField(
                        name=_TAG,
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE
                    ),
                    pml.ParameterField(
                        name=_TAG_UPPER_BOUNDARY,
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE
                    ),
                    pml.ParameterField(
                        name=_TAG_LOWER_BOUNDARY,
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE
                    )
                ],
                Apply=pml.Apply(
                    function=FUNCTION.AND,
                    Apply_member=[
                        pml.Apply(
                            function=FUNCTION.GREATER_THAN,
                            FieldRef=[
                                pml.FieldRef(field=_TAG),
                                pml.FieldRef(field=_TAG_LOWER_BOUNDARY)
                            ]
                        ),
                        pml.Apply(
                            function=FUNCTION.LESS_OR_EQUAL,
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
                optype=OPTYPE.CONTINUOUS,
                dataType=DATATYPE.DOUBLE,
                ParameterField=[
                    pml.ParameterField(
                        name=_TAG,
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE
                    ),
                    pml.ParameterField(
                        name=_TAG_UPPER_BOUNDARY,
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE
                    ),
                    pml.ParameterField(
                        name=_TAG_LOWER_BOUNDARY,
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE
                    )
                ],
                Apply=pml.Apply(
                    function=FUNCTION.IF,
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
                            function=FUNCTION.IF,
                            Apply_member=[
                                pml.Apply(
                                    function=FUNCTION.LESS_OR_EQUAL,
                                    FieldRef=[
                                        pml.FieldRef(field=_TAG),
                                        pml.FieldRef(field=_TAG_LOWER_BOUNDARY)
                                    ]
                                ),
                                pml.Apply(
                                    function=FUNCTION.SUBSTRACTTION,
                                    FieldRef=[
                                        pml.FieldRef(field=_TAG_LOWER_BOUNDARY),
                                        pml.FieldRef(field=_TAG)
                                    ]
                                ),
                                pml.Apply(
                                    function=FUNCTION.SUBSTRACTTION,
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

        def get_transformation_dictionary():
            trans_dict = pml.TransformationDictionary(
                DefineFunction=[
                    get_is_inside_boundary_function(),
                    get_calculate_distance_function()
                ]
            )
            return trans_dict

        def get_mining_fields_for_regression_model(index):
            mining_fields = []
            if not self._use_lag:
                for i in range(self._length_of_fingerprint):
                    mining_fields.append(
                        pml.MiningField(
                            name=self._hulls[index]["name"] + _UNDERSCORE + str(i),
                            usageType="active"
                        )
                    )
            else:
                mining_fields.append(
                    pml.MiningField(
                        name=self._hulls[index]["name"],
                        usageType="active"
                    )
                )

            return mining_fields

        def get_normalization_function():
            if len(self._hulls) == 1:
                max_distance = pml.Constant(valueOf_=self._max_distances[0])
            else:
                max_distance = pml.Constant(valueOf_=self._length_of_fingerprint * len(self._hulls))
            max_distance.original_tagname_ = "Constant"

            constant_100 = pml.Constant(valueOf_=100)
            constant_100.original_tagname_ = "Constant"

            # constant_max_distance = pml.Constant(valueOf_=self._max_distances[0])
            # constant_max_distance.original_tagname_ = "Constant"

            substraction_function = pml.Apply(
                function=FUNCTION.MULTIPLICATION,
                Apply_member=[
                    pml.Apply(
                        function=FUNCTION.DIVISION,
                        Apply_member=[
                            pml.Apply(
                                function=FUNCTION.SUBSTRACTTION,
                                FieldRef=[
                                    max_distance,
                                    pml.FieldRef(field="totalDistance")
                                ]
                            )
                        ],
                        Constant=[
                            max_distance
                        ]
                    ),
                    constant_100
                ]
            )
            substraction_function.original_tagname_ = "Apply"

            equal_function = pml.Apply(
                function=FUNCTION.IF,
                Apply_member=[
                    pml.Apply(
                        function=FUNCTION.EQUAL,
                        FieldRef=[
                            pml.FieldRef(field="totalDistance")
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
                function=FUNCTION.IF,
                Apply_member=[
                    pml.Apply(
                        function=FUNCTION.GREATER_OR_EQUAL,
                        FieldRef=[
                            pml.FieldRef(field="totalDistance")
                        ],
                        Constant=[
                            max_distance
                        ]
                    )
                ],
                Constant=[
                    pml.Constant(valueOf_=0),
                    equal_function
                ]
            )
            # else:
            #     calculated_d = pml.FieldRef(field="totalDistance")
            #     calculated_d.original_tagname_ = "FieldRef"
            #     k = self._length_of_fingerprint * len(self._hulls)
            #
            #     apply = pml.Apply(
            #         function="*",
            #         Apply_member = [
            #             pml.Apply(
            #                 function="/",
            #                 Apply_member=[
            #                     pml.Apply(
            #                         function="-",
            #                         Constant=[
            #                             pml.Constant(valueOf_=k),
            #                             calculated_d
            #                         ]
            #                     )
            #                 ],
            #                 Constant=[
            #                     pml.Constant(valueOf_=k)
            #                 ]
            #             )
            #         ],
            #         Constant=[
            #             pml.Constant(valueOf_=100)
            #         ]
            #     )
            #     return apply

        def get_output_for_mining_model():
            output_fields = [
                pml.OutputField(
                    name="totalDistance",
                    optype=OPTYPE.CONTINUOUS,
                    dataType=DATATYPE.DOUBLE,
                    feature=RESULT_FEATURE.PREDICTED_VALUE,
                ),
                pml.OutputField(
                    name="finalResult",
                    optype=OPTYPE.CONTINUOUS,
                    dataType=DATATYPE.DOUBLE,
                    feature=RESULT_FEATURE.TRANSFORMED_VALUE,
                    Apply=get_normalization_function()
                ),
            ]
            return pml.Output(OutputField=output_fields)

        def get_local_transformation(index):

            derived_fields = []
            derived_field_names = []
            hull = self._hulls[index]
            if self._use_lag:
                for i in range(1, self._length_of_fingerprint):
                    name = hull["name"] + _UNDERSCORE + str(i - 1)
                    derived_fields.append(
                        pml.DerivedField(
                            name=name,
                            optype=OPTYPE.CONTINUOUS,
                            dataType=DATATYPE.DOUBLE,
                            Lag=pml.Lag(field=hull["name"], n=self._length_of_fingerprint - i)
                        )
                    )
                last_derived_name = hull["name"] + _UNDERSCORE + str(self._length_of_fingerprint - 1)
                derived_fields.append(
                    pml.DerivedField(
                        name=last_derived_name,
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE,
                        FieldRef=pml.FieldRef(field=hull["name"])
                    )
                )
            for idx, val in enumerate(hull["values"]):
                name = "distance_tag_" + str(idx)
                derived_field_names.append(name)
                derived_fields.append(
                    pml.DerivedField(
                        name=name,
                        optype=OPTYPE.CONTINUOUS,
                        dataType=DATATYPE.DOUBLE,
                        Apply=pml.Apply(
                            function=_CALCULATE_DISTANCE,
                            FieldRef=[pml.FieldRef(field=hull["name"] + _UNDERSCORE + str(idx))],
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
                    optype=OPTYPE.CONTINUOUS,
                    dataType=DATATYPE.DOUBLE,
                    Apply=pml.Apply(
                        function=FUNCTION.SUM,
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

        def get_output_for_regression_model(index):
            output_fields = [
                pml.OutputField(
                    name="normalizedDistance"+_UNDERSCORE+str(index),
                    optype=OPTYPE.CONTINUOUS,
                    dataType=DATATYPE.DOUBLE
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
                            functionName=MINING_FUNCTION.REGRESSION,
                            MiningSchema=pml.MiningSchema(
                                MiningField=get_mining_fields_for_regression_model(idx)
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

        def get_mining_fields_for_mining_model():
            mining_fields = []
            for hull in self._hulls:
                mining_fields.append(
                    pml.MiningField(
                        name=hull["name"],
                        usageType="active"
                    )
                )
            return mining_fields

        def get_mining_model():
            output = get_output_for_mining_model()
            mining_model = pml.MiningModel(
                functionName=MINING_FUNCTION.REGRESSION,
                modelName=self._fingerprint_name if self._model_name is None else self._model_name,
                MiningSchema=pml.MiningSchema(
                    MiningField=get_mining_fields_for_mining_model()
                ),
                Output=output,
                Segmentation=pml.Segmentation(
                    multipleModelMethod=MULTIPLE_MODEL_METHOD.SUM,
                    Segment=get_segments()
                )
            )
            return mining_model

        header = get_header()
        data_dict = get_data_dictionary()
        trans_dict = get_transformation_dictionary()
        mining_model = get_mining_model()
        pmml = pml.PMML(
            version=PMML_SCHEMA.VERSION,
            Header=header,
            DataDictionary=data_dict,
            TransformationDictionary=trans_dict,
            MiningModel=[mining_model]
        )
        return pmml