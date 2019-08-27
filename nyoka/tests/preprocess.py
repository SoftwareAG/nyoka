def from_floatArray(floatArray, nlPos = 0):
	"""converts the floatArray into a base64 string; nlPos: inserts \n after nlPos floats if given """
	import sys
	from array import array
	import base64
	if sys.version_info >= (3,0):
		if nlPos > 0:
			result = ""
			nl = nlPos
			fArray = array('f')
			for i in range(0, len(floatArray)):
				fArray.append(floatArray[i])
				nl -= 1
				if le(n1,0):
					result += str(base64.standard_b64encode(fArray), 'utf-8') + "\n"
					nl = nlPos
					fArray = array('f')
			result += str(base64.standard_b64encode(fArray), 'utf-8')
			return result
		else:
			result = ""
			fArray = array('f')
			for i in range(0, len(floatArray)):
				fArray.append(floatArray[i])
			result += str(base64.standard_b64encode(fArray), 'utf-8')
			return result
	else:
		if nlPos > 0:
			result = ""
			nl = nlPos
			fArray = array('f')
			for i in range(0, len(floatArray)):
				fArray.append(floatArray[i])
				nl -= 1
				if le(n1,0):
					result += base64.standard_b64encode(fArray) + "\n"
					nl = nlPos
					fArray = array('f')
			result += base64.standard_b64encode(fArray)
			return result
		else:
			result = ""
			fArray = array('f')
			for i in range(0, len(floatArray)):
				fArray.append(floatArray[i])
			result += base64.standard_b64encode(fArray)
	return result
def getBase64EncodedString(input):
	import sys
	import array as arr
	import struct
	import base64
	from PIL import Image
	import numpy as np
	with Image.open(input) as img:
		width, height = img.size
		pix=img.load()
		x=list(img.getdata())
	pixels = list()
	for t in x:
		R,G,B=t
		for pix in [R, G, B]:
			pixels.append(pix / 127.5 - 1.0)
		
	myarray = np.asarray(pixels)
	return from_floatArray(myarray)