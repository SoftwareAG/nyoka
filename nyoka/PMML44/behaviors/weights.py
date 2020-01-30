        import nyoka

        if self.src is not None:
            raw_content = open(self.src, "r").read()
        elif self.content_ is not None and self.content_[0].value is not None:
            raw_content = self.content_[0].value

        raw_content = raw_content.replace(' ', '')
        raw_content = raw_content.replace('\t', '')
        raw_content = raw_content.replace('\n', '')
        
        if raw_content.startswith("data:float32;base64,") or raw_content.startswith("data:float64;base64,") or raw_content.startswith("data:float16;base64,"):
            raw_content = raw_content[20:] + "=="
        elif raw_content.startswith("data:float;base64,"):
            raw_content = raw_content[18:] + "=="
        else:
            return None

        from nyoka.Base64 import FloatBase64
        if raw_content.find("+") > 0:
            return FloatBase64.to_floatArray_urlsafe(raw_content)
        else:
            return FloatBase64.to_floatArray(raw_content)