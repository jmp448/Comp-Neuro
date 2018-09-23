# Josh Popp
# BME 3300

import numpy

def calc_output(in1,in2,in3):
	weights = [[1.0,0.0,4.0],[-1.0,0.0,-4.0]]
	out = numpy.matmul(weights,[in1,in2,in3])
	return out

if __name__=="__main__":
	in1 = int(input("input 1:\n"))
	in2 = int(input("input 2:\n"))
	in3 = int(input("input 3:\n"))

	print(calc_output(in1,in2,in3))
