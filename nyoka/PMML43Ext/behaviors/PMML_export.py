        imported_ns_def_ = supermod.GenerateDSNamespaceDefs_.get('Timestamp')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None:
            name_ = self.original_tagname_
        supermod.showIndent(outfile, level, pretty_print)
        outfile.write('<?xml version="1.0" encoding="UTF-8"?>' + eol_)
        outfile.write('<%s%s%s' % (namespace_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        outfile.write(' xmlns="http://www.dmg.org/PMML-4_3"')
        self.exportAttributes(outfile, level, already_processed, namespace_, name_='Timestamp')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            self.exportChildren(outfile, level + 1, namespace_='', name_='Timestamp', pretty_print=pretty_print)
            supermod.showIndent(outfile, 0, pretty_print)
            outfile.write('</%s%s>%s' % (namespace_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))
        