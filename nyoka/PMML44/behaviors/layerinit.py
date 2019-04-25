        self.original_tagname_ = None
        self.src = supermod._cast(None, src)
        if Extension is None:
            self.Extension = []
        else:
            self.Extension = supermod.Extension
        self.valueOf_ = valueOf_
        if mixedclass_ is None:
            self.mixedclass_ = supermod.MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_
        self.valueOf_ = valueOf_