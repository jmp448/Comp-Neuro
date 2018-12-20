# Josh Popp
# BME 3300
# HW 4 Part 1b
# Noisy LIF Neurons

import numpy as np
import math
import matplotlib.pyplot as plt


def def_volts(current, resistance):
    """
    :param current: the total input current into a neuron
    :param resistance: the neuron's membrane resistance (assumed constant)
    :return voltage: the neuron's voltage over the same time range as the input
    """
    voltage = np.zeros(len(current))
    for i in range(len(current)):
        voltage[i] = current[i] * resistance
    return voltage


def lif(time, volts_in, tau, threshold,
        spike_output=0.0012,
        noisy=False,
        noise_range=0.005):
    """
    :param time: the time range we're looking at
    :param volts_in: the input voltage received by the LIF neuron
    :param tau: the time constant of the LIF neuron, in seconds
    :param threshold: the threshold of the LIF neuron, in volts
    :param spike_output: the magnitude of the spike output by the neuron, in volts
    :param noisy: boolean for whether the neuron's voltage is determined randomly
    :param noise_range: the range over which threshold is allowed to vary

    :return voltage: the LIF neuron's voltage as a function of time
    :return spikes: the LIF neuron's spikes over time
    """
    # Define the time step
    t_step = np.ptp(time) / len(time)

    # Set resting potential
    rp = 0.0

    # Initialize arrays
    v = np.full(len(time), rp)  #voltage of the cell over time
    s = np.zeros(len(time))  #output of the cell over time

    # Fill in according to the input voltage
    for t in range(1, len(time)):
        if noisy:
            threshold_range = np.linspace(threshold - noise_range/2, threshold + noise_range/2, 10)
            threshold = np.random.choice(threshold_range)

        if v[t-1] >= threshold:
            v[t] = rp
            s[t] = spike_output
        else:
            v[t] = v[t-1]*math.exp(-t_step/tau) + volts_in[t]*(1-math.exp(-t_step/tau))
    return v, s


if __name__=="__main__":

    # Get timeframe, in seconds
    t_start = 0.0
    t_stop = 0.05
    mils = int(1000*(t_stop-t_start))  # ensures that time step is 1 millisecond
    time = np.linspace(t_start, t_stop, num=mils, endpoint=False)

    # Get threshold, in volts
    threshold = 0.005

    # Get LIF time constant, in seconds
    tau = 0.002

    # Set number of spikes
    spikes = 10
    # TODO figure out a better way to do this (ie input interspike delay)

    # Get LIF resistance, in ohms (?)
    R = 1.0

    # Define external current flowing into the network
    curr = np.zeros(len(time))
    if spikes != 0:
        for i in range(spikes):
            curr[i * int(len(time)/spikes)] = 0.012

    # Define the voltage getting to the neuron
    voltage = def_volts(curr, R)

    # Neuron 1
    voltage1, spikes1 = lif(time, voltage, tau, threshold, noisy=False)

    # Neuron 2
    voltage2 = def_volts(curr, R)
    voltage2, spikes2 = lif(time, voltage2, tau, threshold, noisy=False)

    plt.subplot(5, 1, 1, xticks=[], yticks=[], ylabel="Current")
    plt.plot(time, curr)

    plt.subplot(5, 1, 2, xticks=[], yticks=[], ylabel="Voltage 1")
    plt.plot(time, voltage1)

    plt.subplot(5, 1, 3, xticks=[], yticks=[], ylabel="Spikes 1")
    plt.plot(time, spikes1)

    plt.subplot(5, 1, 4, xticks=[], yticks=[], ylabel="Voltage 2")
    plt.plot(time, voltage2)

    plt.subplot(5, 1, 5, xticks=[], yticks=[], ylabel="Spikes 2")
    plt.plot(time, spikes2)

    plt.show()
