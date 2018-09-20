import struct
import base64
import sys
from array import array

#https://stackoverflow.com/questions/40914544/python-base64-float
#https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python

# >>> sys.float_info
# sys.float_info(max=1.7976931348623157e+308, max_exp=1024, max_10_exp=308, min=2.2250738585072014e-308, min_exp=-1021, min_10_exp=-307, dig=15, mant_dig=53, epsilon=2.220446049250313e-16, radix=2, rounds=1)

class FloatBase64:
	"""provide several conversions into a base64 encoded string"""

	def __init__(self, param):
		"""initializer"""
		if type(param) == str:	# type check is a bad idea!
			self.string = param
			self.float = self.to_float(param)
		else:
			self.float = param
			self.string = self.from_float(param)

	@classmethod
	def to_float(cls, base64String):
		"""convert string base64String into a float"""
		return struct.unpack('<d', base64.standard_b64decode(base64String)[:8])[0]

	@classmethod
	def from_float(cls, number):
		"""converts the float number into a base64 string"""
		return str(base64.standard_b64encode(struct.pack('<d', number)))

	@classmethod
	def to_floatArray(cls, base64String):
		"""converts the base64String into an array('f')"""
		base64String = base64String.replace("\n", "")
		base64String = base64String.replace("\t", "")
		base64String = base64String.replace(" ", "")
		data = base64.standard_b64decode(base64String)
		count = len(data) // 4
		result = array('f', struct.unpack('<{0}f'.format(count), data))	# one big structure of `count` floats
		return result

	@classmethod
	def to_floatArray_urlsafe(cls, base64String):
		"""converts the base64String into an array('f')"""
		data = base64.urlsafe_b64decode(base64String)
		count = len(data) // 4
		result = array('f', struct.unpack('<{0}f'.format(count), data))	# one big structure of `count` floats
		return result

	@classmethod
	def from_floatArray(cls, floatArray, nlPos = 0):
		"""converts the floatArray into a base64 string; nlPos: inserts \n after nlPos floats if given """
		import sys
		if sys.version_info >= (3,0):
			if nlPos > 0:
				result = ""
				nl = nlPos
				fArray = array('f')
				for i in range(0, len(floatArray)):
					fArray.append(floatArray[i])
					nl -= 1
					if nl <= 0:
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
					if nl <= 0:
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

	@classmethod
	def from_floatArray_urlsafe(cls, floatArray, nlPos = 0):
		"""converts the floatArray into a base64 string; nlPos: inserts \n after nlPos floats if given """
		if nlPos > 0:
			result = ""
			nl = nlPos
			fArray = array('f')
			for i in range(0, len(floatArray)):
				fArray.append(floatArray[i])
				nl -= 1
				if nl <= 0:
					result += base64.urlsafe_b64encode(fArray) + "\n"
					nl = nlPos
					fArray = array('f')
			result += base64.urlsafe_b64encode(fArray)
			return result
		else:
			return base64.urlsafe_b64encode(floatArray)
