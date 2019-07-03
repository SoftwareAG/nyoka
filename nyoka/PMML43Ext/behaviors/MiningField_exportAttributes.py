        if self.name is not None and 'name' not in already_processed:
            already_processed.add('name')
            outfile.write(' name=%s' % (supermod.quote_attrib(self.name), ))
        if self.usageType is not None and 'usageType' not in already_processed:
            already_processed.add('usageType')
            outfile.write(' usageType=%s' % (supermod.quote_attrib(self.usageType), ))
        if self.optype is not None and 'optype' not in already_processed:
            already_processed.add('optype')
            outfile.write(' optype=%s' % (supermod.quote_attrib(self.optype), ))
        if self.importance is not None and 'importance' not in already_processed:
            already_processed.add('importance')
            outfile.write(' importance=%s' % (supermod.quote_attrib(self.importance), ))
        if self.outliers != "asIs" and 'outliers' not in already_processed:
            already_processed.add('outliers')
            outfile.write(' outliers=%s' % (supermod.quote_attrib(self.outliers), ))
        if self.lowValue is not None and 'lowValue' not in already_processed:
            already_processed.add('lowValue')
            outfile.write(' lowValue=%s' % (supermod.quote_attrib(self.lowValue), ))
        if self.highValue is not None and 'highValue' not in already_processed:
            already_processed.add('highValue')
            outfile.write(' highValue=%s' % (supermod.quote_attrib(self.highValue), ))
        if self.missingValueReplacement is not None and 'missingValueReplacement' not in already_processed:
            already_processed.add('missingValueReplacement')
            outfile.write(' missingValueReplacement=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.missingValueReplacement), input_name='missingValueReplacement')), ))
        if self.missingValueTreatment is not None and 'missingValueTreatment' not in already_processed:
            already_processed.add('missingValueTreatment')
            outfile.write(' missingValueTreatment=%s' % (supermod.quote_attrib(self.missingValueTreatment), ))
        if self.invalidValueTreatment != "returnInvalid" and 'invalidValueTreatment' not in already_processed:
            already_processed.add('invalidValueTreatment')
            outfile.write(' invalidValueTreatment=%s' % (supermod.quote_attrib(self.invalidValueTreatment), ))