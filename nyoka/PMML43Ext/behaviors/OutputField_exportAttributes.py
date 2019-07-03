        if self.name is not None and 'name' not in already_processed:
            already_processed.add('name')
            outfile.write(' name=%s' % (supermod.quote_attrib(self.name), ))
        if self.displayName is not None and 'displayName' not in already_processed:
            already_processed.add('displayName')
            outfile.write(' displayName=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.displayName), input_name='displayName')), ))
        if self.optype is not None and 'optype' not in already_processed:
            already_processed.add('optype')
            outfile.write(' optype=%s' % (supermod.quote_attrib(self.optype), ))
        if self.dataType is not None and 'dataType' not in already_processed:
            already_processed.add('dataType')
            outfile.write(' dataType=%s' % (supermod.quote_attrib(self.dataType), ))
        if self.targetField is not None and 'targetField' not in already_processed:
            already_processed.add('targetField')
            outfile.write(' targetField=%s' % (supermod.quote_attrib(self.targetField), ))
        if self.feature is not None and 'feature' not in already_processed:
            already_processed.add('feature')
            outfile.write(' feature=%s' % (supermod.quote_attrib(self.feature), ))
        if self.value is not None and 'value' not in already_processed:
            already_processed.add('value')
            outfile.write(' value=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.value), input_name='value')), ))
        if self.ruleFeature != "consequent" and 'ruleFeature' not in already_processed:
            already_processed.add('ruleFeature')
            outfile.write(' ruleFeature=%s' % (supermod.quote_attrib(self.ruleFeature), ))
        if self.algorithm != "exclusiveRecommendation" and 'algorithm' not in already_processed:
            already_processed.add('algorithm')
            outfile.write(' algorithm=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.algorithm), input_name='algorithm')), ))
        # if self.rank is not None and 'rank' not in already_processed:
            # already_processed.add('rank')
            # outfile.write(' rank=%s' % (supermod.quote_attrib(self.rank), ))
        if self.rankBasis != "confidence" and 'rankBasis' not in already_processed:
            already_processed.add('rankBasis')
            outfile.write(' rankBasis=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.rankBasis), input_name='rankBasis')), ))
        if self.rankOrder != "descending" and 'rankOrder' not in already_processed:
            already_processed.add('rankOrder')
            outfile.write(' rankOrder=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.rankOrder), input_name='rankOrder')), ))
        if self.isMultiValued != "0" and 'isMultiValued' not in already_processed:
            already_processed.add('isMultiValued')
            outfile.write(' isMultiValued=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.isMultiValued), input_name='isMultiValued')), ))
        if self.segmentId is not None and 'segmentId' not in already_processed:
            already_processed.add('segmentId')
            outfile.write(' segmentId=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.segmentId), input_name='segmentId')), ))
        if not self.isFinalResult and 'isFinalResult' not in already_processed:
            already_processed.add('isFinalResult')
            outfile.write(' isFinalResult="%s"' % self.gds_format_boolean(self.isFinalResult, input_name='isFinalResult'))
        if self.numTopCategories is not None and 'numTopCategories' not in already_processed:
            already_processed.add('numTopCategories')
            outfile.write(' numTopCategories=%s' % (supermod.quote_attrib(self.numTopCategories), ))
        if self.threshold is not None and 'threshold' not in already_processed:
            already_processed.add('threshold')
            outfile.write(' threshold=%s' % (supermod.quote_attrib(self.threshold), ))
