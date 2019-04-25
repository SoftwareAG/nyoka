        from datetime import datetime
        
        if self.copyright is not None and 'copyright' not in already_processed:
            if not self.copyright.endswith("Software AG"):
                self.copyright += ", exported to PMML by Nyoka (c) " + str(datetime.now().year) + " Software AG"
            already_processed.add('copyright')
            outfile.write(' copyright=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.copyright), input_name='copyright')), ))
        if self.description is not None and 'description' not in already_processed:
            already_processed.add('description')
            outfile.write(' description=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.description), input_name='description')), ))
        if self.modelVersion is not None and 'modelVersion' not in already_processed:
            already_processed.add('modelVersion')
            outfile.write(' modelVersion=%s' % (self.gds_encode(self.gds_format_string(supermod.quote_attrib(self.modelVersion), input_name='modelVersion')), ))
