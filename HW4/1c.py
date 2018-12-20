# Josh Popp
# BME 3300
# HW 4 Part 1c
# Noisy LIF Neurons - Basic Network

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


def lif(time, volts_in,
        tau, threshold, rp=0.0,
        spike_output=0.0012,
        noisy=False,
        noise_range=0.005):
    """
    :param time: the time range we're looking at
    :param volts_in: the input voltage received by the LIF neuron
    :param tau: the time constant of the LIF neuron, in seconds
    :param threshold: the threshold of the LIF neuron, in volts
    :param rp: resting potential of the LIF neuron, in volts
    :param spike_output: the magnitude of the spike output by the neuron, in volts
    :param noisy: boolean for whether the neuron's voltage is determined randomly
    :param noise_range: the range over which threshold is allowed to vary

    :return voltage: the LIF neuron's voltage as a function of time
    :return spikes: the LIF neuron's spikes over time
    """
    # Define the time step
    t_step = np.ptp(time) / len(time)

    # Initialize arrays
    v = np.full(len(time), rp) #voltage of the cell over time
    s = np.zeros(len(time)) #output of the cell over time

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


def lif_network(weights, time, curr_in,
                r, tau, threshold, rp,
                spike_output,
                noisy,
                noise_range):
    """
    Calculates the output for an entire neural network of lif neurons
    :param weights: square matrix of synaptic weights for the neurons in the network
                    index 0 corresponds to the input current, so neurons are one-indexed
                    NOTE that in all other parameters neurons are zero-indexed
    :param time: array of time points over which sampling is to occur
    :param curr_in: the external input current, in amps, over time
    :param r: array of resistances for each neuron in the net
    :param tau: array of time constants for each neuron in the net
    :param threshold: array of (average) thresholds for each neuron in the net
    :param rp: array of resting potentials for each neuron in the net
    :param spike_output: array of the magnitudes of current released by an action potential for each neuron
    :param noisy: boolean array saying whether each neuron is noisy
    :param noise_range: noise range over which each neuron varies

    :return voltages: 2D array of the voltages for each neuron over time
                      position 0 is which neuron we're looking at, position 1 is time
    :return spikes: 2D array of the spikes for each neuron over time
                    position 0 is which neuron we're looking at, position 1 is time
    """
    # Define the time step
    t_step = np.ptp(time) / len(time)

    # Initialize arrays
    v = np.zeros([len(weights), len(time)])  # voltage of the cell over time
    for n in range(len(weights)):
        for t in range(len(time)):
            v[n, t] = rp[n]
    s = np.zeros([len(weights), len(time)])
    for n in range(len(weights)):
        s[n] = np.full([len(time)], rp[n])  # output of the cell over time

    # Fill in according to the input voltage
    for t in range(1, len(time)):
        for n in range(len(weights)):
            if noisy[n]:
                threshold_range = np.linspace(threshold[n] - noise_range[n] / 2, threshold[n] + noise_range[n] / 2, 10)
                threshold[n] = np.random.choice(threshold_range)
            if v[n, t-1] >= threshold[n]:
                v[n, t] = rp[n]
                s[n, t] = spike_output[n]
            else:
                volts_in = np.matmul(weights[n], np.concatenate([[curr_in[t-1]*r[n]], v[:, t-1]]))
                v[n, t] = v[n, t - 1] * math.exp(-t_step / tau[n]) + volts_in * (1 - math.exp(-t_step / tau[n]))
    return v, s


if __name__=="__main__":

    # Get timeframe, in seconds
    t_start = 0.0
    t_stop = 0.05
    mils = int(1000*(t_stop-t_start))  # ensures that time step is 1 millisecond
    time = np.linspace(t_start, t_stop, num=mils, endpoint=False)

    # Get threshold, in volts
    threshold = [0.005, 0.005]

    # Get LIF time constant, in seconds
    tau = [0.002, 0.002]

    # Set resting potentials
    rp = [0.0, 0.0]

    # Spike output
    so = [0.0012, 0.0012]

    # Noisy
    noisy = [True, False]

    # Set number of spikes
    spikes = 10
    # TODO figure out a better way to do this (ie input interspike delay)

    # Get LIF resistance, in ohms (?)
    r = [1.0, 1.0]

    # Set noise ranges
    noise_range = [0.005, 0.005]
    # Define external current flowing into the network
    curr = np.zeros(len(time))
    if spikes != 0:
        for i in range(spikes):
            curr[i * int(len(time)/spikes)] = 0.012

    weights = [[1, 0, -1], [1, -1, 0]]

    print(len(weights))

    v, s = lif_network(weights, time, curr,
        r, tau, threshold, rp, so,
        noisy, noise_range)
    print(v.shape)

    #
    plt.subplot(5, 1, 1, xticks=[], yticks=[], ylabel="Current")
    plt.plot(time, curr)

    plt.subplot(5, 1, 2, xticks=[], yticks=[], ylabel="Voltage 1")
    plt.plot(time, v[0, :])

    plt.subplot(5, 1, 3, xticks=[], yticks=[], ylabel="Spikes 1")
    plt.plot(time, s[0, :])

    plt.subplot(5, 1, 4, xticks=[], yticks=[], ylabel="Voltage 2")
    plt.plot(time, v[1, :])

    plt.subplot(5, 1, 5, xticks=[], yticks=[], ylabel="Spikes 2")
    plt.plot(time, s[1, :])

    plt.show()
