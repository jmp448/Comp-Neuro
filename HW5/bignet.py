# Josh Popp
# BME 3300
# HW 5
# 25-neuron net

import numpy
import math
import matplotlib.pyplot as plt


def lif_neuron_voltage(time, curr_in, tau, threshold, t_step):
    rp = -60.0
    v_lif = numpy.full(len(time), rp)
    for t in range(1, len(time)):
        if v_lif[t - 1] >= threshold:
            v_lif[t] = rp
        else:
            print(v_lif[t - 1] * math.exp(-t_step / tau))
            print((1 - math.exp(-t_step / tau)) * curr_in[t])
            print((v_lif[t - 1] * math.exp(-t_step / tau)) + ((1 - math.exp(-t_step / tau)) * curr_in[t]))
            v_lif[t] = (v_lif[t - 1] * math.exp(-t_step / tau)) + ((1 - math.exp(-t_step / tau)) * curr_in[t])
    return v_lif


def lif_neuron_spikes(voltage):
    t = 1
    out_lif = numpy.zeros(len(time))
    while t in range(1, len(time)):
        if voltage[t] >= threshold:
            out_lif[t] = 1
        t += 1
    return out_lif


if __name__ == "__main__":
    # Get timeframe
    time = numpy.linspace(float(0), float(500), num=500, endpoint=False)

    # Get threshold
    threshold = 0.0

    # Get LIF time constant
    tau = 10.0

    # Set number of spikes in the window
    spikes = 10

    # Get the time step
    t_step = (time[len(time)-1]-time[0]) / len(time)

    # curr_in = numpy.zeros(len(time))
    curr_in = numpy.zeros(len(time))

    if spikes != 0:
        for i in range(spikes):
            curr_in[i * int(len(time)/spikes)] = 1.0

    # plt.subplot(6, 5, 3, xticks=[], yticks=[])
    # plt.plot(time, curr_in)
    #
    # for i in range(5):
    #     for j in range(5):
    #         voltage = lif_neuron_voltage(time, curr_in, tau, threshold, t_step)
    #         plt.subplot(6, 5, (i+1)*5+j+1, xticks=[], yticks=[])
    #         plt.plot(time, voltage)

    plt.subplot(3, 2, 2, xticks=[], yticks=[])
    plt.plot(time, curr_in)

    for i in range(3):
        for j in range(3):
            voltage = lif_neuron_voltage(time, curr_in, tau, threshold, t_step)
            plt.subplot(3, 2, (i + 1) * 2 + j + 1, xticks=[], yticks=[])
            plt.plot(time, voltage)

    plt.show()
