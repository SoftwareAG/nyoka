        already_processed = set()
        self.buildAttributes(node, node.attrib, already_processed)
        for child in node:
            nodeName_ = supermod.Tag_pattern_.match(child.tag).groups()[-1]
            self.buildChildren(child, node, nodeName_)
        if self.anytypeobjs_ == []:
            if node.text is not None:
                self.anytypeobjs_ = list(filter(None, [obj_.lstrip(' ') for obj_ in node.text.split('\n')]))
        return self