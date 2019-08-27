        if pretty_print:
            eol_ = '\n'
        else:
            eol_ = ''
        for obj_ in self.anytypeobjs_:
            try:
                obj_.export(outfile, level, namespace_, pretty_print=pretty_print)
            except:
                showIndent(outfile, level, pretty_print)
                outfile.write(str(obj_))
                outfile.write(eol_)
        for objName_ in self.elementobjs_:
            obj_ = eval("self." + objName_)
            if eval("isinstance(obj_, list)"):
                for s in obj_:
                    showIndent(outfile, level, pretty_print)
                    outfile.write("<" + objName_ + ">" + str(s) + "</" + objName_ + ">")
                    outfile.write(eol_)
            else:
                showIndent(outfile, level, pretty_print)
                outfile.write("<" + objName_ + ">" + str(obj_) + "</" + objName_ + ">")
                outfile.write(eol_)