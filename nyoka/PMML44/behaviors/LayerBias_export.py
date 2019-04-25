        imported_ns_def_ = supermod.GenerateDSNamespaceDefs_.get('LayerBias')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None:
            name_ = self.original_tagname_
        supermod.showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespace_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespace_, name_='LayerBias')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            if not pretty_print:
                self.content_[0].value = self.content_[0].value.replace('\t', '').replace(' ', '')
                self.valueOf_ = self.valueOf_.replace('\t', '').replace(' ', '')
            self.exportChildren(outfile, level + 1, namespace_='', name_='LayerBias', pretty_print=pretty_print)
            outfile.write(eol_)
            supermod.showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespace_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))