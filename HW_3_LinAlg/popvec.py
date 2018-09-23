# Josh Popp
# BME 3300

import numpy

N1MAX = 45
N2MAX = 90
N3MAX = 100
MAX_VEC = [N1MAX,N2MAX,N3MAX]

def calc_dir(r1,r2,r3):
	norm = r1+r2+r3
	direction = numpy.matmul(MAX_VEC,[r1/norm,r2/norm,r3/norm])
	return direction

if __name__=="__main__":
	r1 = int(input("r1: \n"))
	r2 = int(input("r2: \n"))
	r3 = int(input("r3: \n"))

	print(calc_dir(r1,r2,r3))