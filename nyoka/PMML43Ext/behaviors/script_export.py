        imported_ns_def_ = supermod.GenerateDSNamespaceDefs_.get('script')
        if imported_ns_def_ is not None:
            namespacedef_ = imported_ns_def_
        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        if self.original_tagname_ is not None:
            name_ = self.original_tagname_
        showIndent(outfile, level, pretty_print)
        outfile.write('<%s%s%s' % (namespace_, name_, namespacedef_ and ' ' + namespacedef_ or '', ))
        already_processed = set()
        self.exportAttributes(outfile, level, already_processed, namespace_, name_='script')
        if self.hasContent_():
            outfile.write('>%s' % (eol_, ))
            if pretty_print:
                lines = []
                code = self.valueOf_.lstrip('\n')
                leading_spaces = len(code) - len(code.lstrip(' '))
                for line in code.split('\n'):
                    lines.append(line[leading_spaces:])
                code = '\n'.join(lines)
                indent = "    " * (level + 1)
                count = code.count('\n')
                indented = indent + code.replace("\n", "\n" + indent, count - 1)
                self.content_ = [supermod.MixedContainer(1, 2, "", str(indented))]
                self.valueOf_ = str(indented)
            self.exportChildren(outfile, level + 1, namespace_='', name_='script', pretty_print=pretty_print)
            showIndent(outfile, level, pretty_print)
            outfile.write('</%s%s>%s' % (namespace_, name_, eol_))
        else:
            outfile.write('/>%s' % (eol_, ))