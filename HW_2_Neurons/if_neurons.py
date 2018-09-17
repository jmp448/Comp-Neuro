# Josh Popp
# BME 3300
# HW 2
# IF and LIF neurons

import numpy
import math
import matplotlib.pyplot as plt

def if_neuron_voltage(time,curr_in,cm,threshold):

	t=1
	rp = -60
	v_if=numpy.full(len(time),rp)
	while t in range(1,len(time)):
		if v_if[t-1] >= threshold:
			v_if[t] = rp
		else:
			v_if[t] = v_if[t-1]+curr_in[t]/cm
		t+=1
	return v_if

def if_neuron_spikes(voltage):

	t=1
	out_if=numpy.zeros(len(time))
	while t in range(1,len(time)):
		if voltage[t] >= threshold:
			out_if[t]=1
		t+=1
	return out_if

def lif_neuron_voltage(time,curr_in,tau,threshold):

	t=1
	rp = -60
	v_lif = numpy.full(len(time),rp)
	while t in range(1,len(time)):
		if v_lif[t-1] >= threshold:
			v_lif[t] = rp
		else:
			v_lif[t]=v_lif[t-1]*math.exp(-1/tau)+(1-math.exp(-1/tau))*curr_in[t]
		t+=1
	return v_lif

def lif_neuron_spikes(voltage):

	t=1
	out_lif=numpy.zeros(len(time))
	while t in range(1,len(time)):
		if v_lif[t] >= threshold:
			out_lif[t]=1
		t+=1
	return out_lif

# Define LIF voltage over time
v_lif = numpy.zeros(500)
t = 1
# Plot 1
# tau = 75
# Plot 2
tau = 150

if __name__=="__main__":
	# Get timeframe
	time = input("Enter length of time in seconds: \n")
	if time == '':
		time = numpy.linspace(float(0),float(500),num=500,endpoint=False)
	else:
		time = numpy.linspace(float(0),float(time),num=500,endpoint=False)

	# Get threshold
	threshold = input("Enter threshold: \n")
	if threshold == '':
		threshold = 0
	else:
		threshold = float(threshold)

	# Get IF time constant
	cm = input("Enter IF time constant: \n")
	if cm == '':
		cm = 25
	else:
		cm = float(cm)

	# Get LIF time constant
	tau = input("Enter LIF time constant: \n")
	if tau == '':
		tau = 75
	else:
		tau = float(tau)

	# Get if time constant 
	# cm = 25

	# Get input format
	in_type = input("Enter input current, either 'step' or 'sine': \n")
	if in_type=='':
		in_type='step'
	elif in_type=='sine':
		freq = input("Enter frequency 1, 3, or 5: \n")

	curr_in = numpy.zeros(len(time))
	for i in range(len(time)):
		if in_type=='step':
			curr_in[i]=threshold*math.floor(time[i]/(len(time)/5))
		else:
			curr_in[i] = 45*math.sin(float(freq)*time[i]*math.pi/180)

	print(curr_in)
	plt.subplot(5,1,1,xticks=[],yticks=[],ylabel="Input")
	plt.plot(time,curr_in)

	v_if = if_neuron_voltage(time, curr_in, cm, threshold)

	plt.subplot(5,1,2,xticks=[],yticks=[],ylabel="IF Voltage")
	plt.plot(time,v_if)

	out_if = if_neuron_spikes(v_if)

	plt.subplot(5,1,3,xticks=[],yticks=[],ylabel="IF Output")
	plt.plot(time,out_if)

	v_lif = if_neuron_voltage(time, curr_in, tau, threshold)

	plt.subplot(5,1,4,xticks=[],yticks=[],ylabel="LIF Voltage")
	plt.plot(time,v_lif)

	out_lif = lif_neuron_spikes(v_lif)

	plt.subplot(5,1,5,xticks=[],yticks=[],ylabel="LIF Output")
	plt.plot(time,out_lif)

	print(v_if)
	print('LIF: \n')
	print(v_lif)
	plt.show()
