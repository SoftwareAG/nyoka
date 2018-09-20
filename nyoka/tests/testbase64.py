from __future__ import print_function
import sys
from nyoka.Base64 import *

def test1():
	"""test static methods to_base64 and to_float"""
	print({ "Python version" : sys.version_info })
	pi = 3.1415926
	piBase64 = FloatBase64.from_float(pi)
	pi2 = FloatBase64.to_float(piBase64)
	print({ "pi" : pi, "piBase64" : piBase64, "pi2" : pi2})

def test2():
	"""test constructor"""
	f1 = FloatBase64(3.1415926)
	f2 = FloatBase64('StgSTfshCUA=\n')
	print([
			{ "name" : "f1", "float" : f1.float, "string" : f1.string },
			{ "name" : "f2", "float" : f2.float, "string" : f2.string }])

def test3():
	"""convert float[] to base64 and back"""
	e = 2.7182818284590452353602874713527
	eLow = 2.71
	pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286
	piLow = 3.14
	# https://docs.python.org/3/library/array.html https://docs.python.org/2/library/array.html
	src =array('f', [
		e,
		eLow,
		pi,
		piLow,
		e,
		eLow,
		pi,
		piLow
	])
	assert type(src) == array
	base64Array = FloatBase64.from_floatArray(src)
	dst = FloatBase64.to_floatArray(base64Array)
	print({ "src" : src, "dst" : dst })
	for i in range(0, 7):
		assert src[i] == dst[i]
	# with newline
	base64ArrayNewLine = FloatBase64.from_floatArray(src, 3)
	print({ "base64ArrayNewLine" : base64ArrayNewLine })
	print({ "base64Array" : base64Array })
	dst = FloatBase64.to_floatArray(base64ArrayNewLine)
	for i in range(0, 8):
		assert src[i] == dst[i]

def test4():
	e = 2.7182818284590452353602874713527
	eLow = 2.71
	pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286
	piLow = 3.14
	arr = array('d', [
		e,
		eLow,
		pi,
		piLow,
		e,
		eLow,
		pi,
		piLow
	])
	print(arr)

def run_tests():
	test4()
	test1()
	test2()
	test3()

run_tests()
#float_array = FloatBase64.to_floatArray(sys.argv[1])
#array = array('f', float_array)
#print FloatBase64.from_floatArray(array)