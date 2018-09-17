# Josh Popp
# BME 3300
# Python HW 2

# AND function
def and_fcn(in1,in2):
	thresh = 2
	if in1+in2>=thresh:
		return 1
	else:
		return 0

# OR function
def or_fcn(in1,in2):
	thresh = 1
	if in1+in2>=thresh:
		return 1
	else:
		return 0

# XOR function
def xor_fcn(in1,in2):
	thresh_i1 = 1
	if in1-in2 >= thresh_i1:
		out_i1 = 1
	else:
		out_i1 = 0

	thresh_i2 = 1
	if in2-in1 >= thresh_i2:
		out_i2 = 1
	else:
		out_i2 = 0

	thresh = 1
	if out_i1+out_i2 >= thresh:
		return 1
	else:
		return 0

if __name__=="__main__":

	# Ask user for function
	fcn = input("What function would you like to run?  Please enter AND,OR, or XOR: \n")
	if fcn not in ['AND','OR','XOR']:
		print("Uh oh, looks like that's not a function this neuron can compute! \n")
		fcn = input("What function would you like to run?  Please enter AND,OR, or XOR: \n")
	
	# Ask user for first neuron
	in1 = input("Please enter the activity of the first neuron (0 or 1): \n")
	if int(in1) not in [0,1]:
		print("Uh oh, looks like that's not a valid input value \n")
		in1 = input("Please enter the activity of the first neuron (0 or 1): \n")
	
	# Ask user for second neuron
	in2 = input("Please enter the activity of the second neuron (0 or 1): \n")
	if int(in2) not in [0,1]:
		print("Uh oh, looks like that's not a valid input value \n")
		in2 = input("Please enter the activity of the first neuron (0 or 1): \n")

	# Calculate and print neuron output
	if fcn == 'AND':
		print("The neuron's output is %s" % and_fcn(int(in1),int(in2)))
	elif fcn == 'OR':
		print("The neuron's output is %s" % or_fcn(int(in1),int(in2)))
	elif fcn == 'XOR':
		print("The neuron's output is %s" % xor_fcn(int(in1),int(in2)))


