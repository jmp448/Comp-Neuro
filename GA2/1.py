import numpy as np
import matplotlib.pyplot as plt
import math


class LIFNet:
    def __init__(self, neurons, wiring):
        """
        :param neurons: list of neurons in the net
        :param wiring: map of maps, maps a neuron onto an array of its
                       inputs and the weight for each synapse
        """
        self.neurons = neurons
        self.wiring = wiring

    def update_network(self, time, curr_in=None):
        for n in self.neurons:
            if n.voltage is None:
                n.voltage = np.full(len(time), n.rp)
            if n.spikes is None:
                n.spikes = np.zeros(len(time))
            if n.output is None:
                n.output = np.zeros(len(time))

        tstep = np.ptp(time) / len(time)

        for tpoint in range(len(time)):
            for n in self.neurons:
                if len(self.wiring[n].keys()) > 0:
                    print(n.name)
                    print(n.voltage[tpoint - 1])
                    ins = self.wiring[n].keys()
                    in_volts = []
                    weights = []
                    for neuron in ins:
                        in_volts.append(neuron.output[tpoint - 1])
                        weights.append(self.wiring[n][neuron])
                    print(in_volts)
                    print(weights)
                    v_in = np.matmul(weights, in_volts)

                    if n.noisy:
                        threshold_range = np.linspace(n.threshold - n.noise_range / 2,
                                                      n.threshold + n.noise_range / 2, 10)
                        threshold = np.random.choice(threshold_range)
                    else:
                        threshold = n.threshold

                    if n.voltage[tpoint - 1] >= threshold:
                        n.voltage[tpoint] = n.rp
                        n.spikes[tpoint] = 1
                        n.output[tpoint] = n.spike_output
                    else:
                        n.voltage[tpoint] = n.voltage[tpoint - 1] * math.exp(-tstep / n.tc) \
                                            + v_in * (1 - math.exp(-tstep / n.tc))


class LIFNeuron:
    def __init__(self, name=None, threshold=0.005, tc=0.002, resist=1, rp=0.0,
                 inputs=None, stimulus=None,
                 spike_output=0.0012,
                 noisy=False, noise_range=0.005):
        self.name = name
        self.threshold = threshold
        self.tc = tc
        self.resist = resist
        self.rp = rp
        self.inputs = inputs
        self.stimulus = stimulus
        self.spike_output = spike_output
        self.noisy = noisy
        self.noise_range = noise_range
        self.voltage = None
        self.spikes = None
        self.output = None

    def update_voltage(self, time, v_in):
        self.voltage = np.full(len(time), self.rp)
        self.spikes = np.zeros(len(time))
        self.output = np.zeros(len(time))

        tstep = np.ptp(time) / len(time)

        for tpoint in range(1, len(time)):
            if self.noisy:
                threshold_range = np.linspace(self.threshold - self.noise_range / 2,
                                              self.threshold + self.noise_range / 2, 10)
                threshold = np.random.choice(threshold_range)
            else:
                threshold = self.threshold

            if self.voltage[tpoint-1] >= threshold:
                self.voltage[tpoint] = self.rp
                self.spikes[tpoint] = 1
                self.output[tpoint] = self.spike_output
            else:
                self.voltage[tpoint] = self.voltage[tpoint-1] * math.exp(-tstep/self.tc) \
                                       + v_in[tpoint] * (1-math.exp(-tstep/self.tc))

    # def update_voltage(self, time_point, inputs):
    #     # Set threshold
    #     if self.noisy:
    #         threshold_range = np.linspace(self.threshold - self.noise_range / 2, self.threshold + self.noise_range / 2, 10)
    #         threshold = np.random.choice(threshold_range)
    #     else:
    #         threshold = self.threshold
    #
    #     if self.voltage[time_point-1] >= threshold:
    #         self.voltage[time_point] = self.rp
    #         self.spikes[time_point] = self.spike_output
    #     else:
    #         used_inputs = self.inputs.keys()
    #         self.voltage[time_point] = np.matmul(weights[n], np.concatenate([[curr_in[t - 1] * r[n]], v[:, t - 1]]))
    #         v[n, t] = v[n, t - 1] * math.exp(-t_step / tau[n]) + volts_in * (1 - math.exp(-t_step / tau[n]))


def def_p_neurons():

    x = np.linspace(-180, 180, 1000)
    y1 = np.zeros(len(x))
    y2 = np.zeros(len(x))
    y3 = np.zeros(len(x))
    y4 = np.zeros(len(x))

    for i in range(len(x)):

        # Neuron 1
        y1[i] = p1_response(x[i])

        # Neuron 2
        y2[i] = p2_response(x[i])

        # Neuron 3
        y3[i] = p3_response(x[i])

        # Neuron 4
        y4[i] = p4_response(x[i])

    return x, y1, y2, y3, y4


def p1_response(theta):
    if theta < 0:
        response = -1 * (theta + 225.0) * (theta + 45.0) / 8100.0
    else:
        response = -1 * (theta - 135.0) * (theta - 315.0) / 8100.0
    if response < 0:
        response = 0.0
    return response


def p2_response(theta):
    response = -1 * (theta + 135.0) * (theta - 45.0) / 8100.0
    if response < 0:
        response = 0.0
    return response


def p3_response(theta):
    response = -1 * (theta - 135.0) * (theta + 45.0) / 8100.0
    if response < 0:
        response = 0.0
    return response


def p4_response(theta):
    if theta > 0:
        response = -1 * (theta - 45.0) * (theta - 225.0) / 8100.0
    else:
        response = -1 * (theta + 135.0) * (theta + 315.0) / 8100.0
    if response < 0:
        response = 0
    return response


def show_neurons(x, y):

    for i in range(len(y)):
        plt.plot(x, y[i])

    plt.xlim(left=-180, right=180)
    plt.xlabel('Stimulus location (angle in degrees)')
    plt.ylim(bottom=0, top=1)
    plt.ylabel('Normalized spike count')
    plt.title('Neuron Tuning Curves')
    plt.show()


def p_spike_train(t, theta, neuron, max_spikes=10):
    """
    This function creates a spike train for a specific neuron in response to a specific stimulus angle

    :param t: time range
    :param theta: angle of stimulus
    :param neuron: p neuron whose response we're looking at
    :param max_spikes: maximum number of spikes over t

    :return spikes: spike train for time range t
    """
    t_stop = t[len(t)-1]
    if neuron == 1:
        if p1_response(theta) > 0:
            interval = float(t_stop)/(p1_response(theta)*max_spikes)
        else:
            interval = np.inf
    elif neuron == 2:
        if p2_response(theta) > 0:
            interval = float(t_stop)/(p2_response(theta)*max_spikes)
        else:
            interval = np.inf
    elif neuron == 3:
        if p3_response(theta) > 0:
            interval = float(t_stop)/(p3_response(theta)*max_spikes)
        else:
            interval = np.inf
    else:
        if p4_response(theta) > 0:
            interval = float(t_stop)/(p4_response(theta)*max_spikes)
        else:
            interval = np.inf

    spikes = np.zeros(len(t))
    clock = 0
    for j in range(len(t)):
        if clock >= interval:
            clock = 0
            spikes[j] = 1
        else:
            spikes[j] = 0
            clock += float(t_stop)/1000

    return spikes


def plot_p_spike_trains(time, angles):
    spikes = np.zeros([len(angles), 4, len(time)])

    num_plots = 4 * len(angles)
    print(num_plots)
    for i in range(len(angles)):
        for j in range(4):
            if j == 0 and i == 0:
                plt.subplot(len(angles), 4, i*4+j+1, xticks=[], yticks=[], ylabel="%d" % angles[i],
                            title="Neuron %d" % (j + 1))
            elif i == 0:
                plt.subplot(len(angles), 4, i*4+j+1, xticks=[], yticks=[], title="Neuron %d" % (j + 1))
            elif j == 0:
                plt.subplot(len(angles), 4, i*4+j+1, xticks=[], yticks=[], ylabel="%d" % angles[i])
            else:
                plt.subplot(len(angles), 4, i*4+j+1, xticks=[], yticks=[])

            spikes[i, j] = p_spike_train(time, angles[i], j + 1)
            plt.plot(spikes[i, j])

    plt.show()

    return spikes


def spike_train(t, norm, max_spikes):

    t_stop = t[len(t)-1]

    if norm > 0:
        interval = float(t_stop) / (norm * max_spikes)
    else:
        interval = np.inf

    spikes = np.zeros(len(t))
    clock = 0
    for j in range(len(t)):
        if clock >= interval:
            clock = 0
            spikes[j] = 1
        else:
            spikes[j] = 0
            clock += float(t_stop) / 1000

    return spikes


def spike2curr(spikes, mag):
    curr = np.zeros(len(spikes))
    for i in range(len(spikes)):
        if spikes[i] == 1:
            curr[i] = mag
    return curr


# def update_network(net, time, curr_in=None):
#     for n in net.neurons:
#         if n.voltage is None:
#             n.voltage = np.full(len(time), n.rp)
#         if n.spikes is None:
#             n.spikes = np.zeros(len(time))
#         if n.output is None:
#             n.output = np.zeros(len(time))
#
#     tstep = np.ptp(time) / len(time)
#
#     for tpoint in range(len(time)):
#         for n in net.neurons:
#             if len(net.wiring[n].keys()) > 0:
#                 print(n.name)
#                 print(n.voltage[tpoint - 1])
#                 ins = net.wiring[n].keys()
#                 in_volts = []
#                 weights = []
#                 for neuron in ins:
#                     in_volts.append(neuron.output[tpoint - 1])
#                     weights.append(net.wiring[n][neuron])
#                 print(in_volts)
#                 print(weights)
#                 v_in = np.matmul(weights, in_volts)
#
#                 if n.noisy:
#                     threshold_range = np.linspace(n.threshold - n.noise_range / 2,
#                                                   n.threshold + n.noise_range / 2, 10)
#                     threshold = np.random.choice(threshold_range)
#                 else:
#                     threshold = n.threshold
#
#                 if n.voltage[tpoint - 1] >= threshold:
#                     n.voltage[tpoint] = n.rp
#                     n.spikes[tpoint] = 1
#                     n.output[tpoint] = n.spike_output
#                 else:
#                     n.voltage[tpoint] = n.voltage[tpoint - 1] * math.exp(-tstep / n.tc) \
#                                         + v_in * (1 - math.exp(-tstep / n.tc))
#

if __name__ == "__main__":

    # Create plot of response curves
    x, p1_pre, p2_pre, p3_pre, p4_pre = def_p_neurons()
    # show_neurons(x, [y1, y2, y3, y4])

    # Print spike trains for neurons over some range of angles
    # time = np.linspace(0, 1, 1000)
    # angles = [-150, -105, -60, -15, 30, 75]
    # # angles = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    # spikes = plot_p_spike_trains(time, angles)

    # # Create lif neurons that spike as desired
    # threshold = 0.005
    # r = 1.0
    # # for i in range(len(angles)):
    # curr_in = spike2curr(spikes, threshold)
    # volts_in = [curr_in[i] * r for i in range(len(curr_in))]
    # tau = 0.000001

    # # Test LIF Neuron class (WORKS)
    # # Get timeframe, in seconds
    # t_start = 0.0
    # t_stop = 0.05
    # mils = int(1000 * (t_stop - t_start))  # ensures that time step is 1 millisecond
    # time = np.linspace(t_start, t_stop, num=mils, endpoint=False)
    # #
    # # Define external current flowing into the network
    # curr = np.zeros(len(time))
    # spikes = 10
    # if spikes != 0:
    #     for i in range(spikes):
    #         curr[i * int(len(time) / spikes)] = 0.012
    #
    # testy = LIFNeuron(noisy=True)
    # testy.update_voltage(time, curr)
    #
    # plt.subplot(3, 1, 1)
    # plt.plot(time, curr)
    # plt.subplot(3, 1, 2)
    # plt.plot(time, testy.voltage)
    # plt.subplot(3, 1, 3)
    # plt.plot(time, testy.spikes)
    #
    # plt.show()

    # # Test LIFNet (WORKS)
    # # Get timeframe, in seconds
    # t_start = 0.0
    # t_stop = 0.05
    # mils = int(1000 * (t_stop - t_start))  # ensures that time step is 1 millisecond
    # time = np.linspace(t_start, t_stop, num=mils, endpoint=False)
    # #
    # # Define external current flowing into the network
    # curr = np.zeros(len(time))
    # spikes = 10
    # if spikes != 0:
    #     for i in range(spikes):
    #         curr[i * int(len(time) / spikes)] = 0.012
    # input_neuron = LIFNeuron(name="input current")
    # input_neuron.output = curr
    #
    # n1 = LIFNeuron(name="n1")
    #
    # net1_wiring = {
    #     input_neuron: {},
    #     n1: {input_neuron: 1.0}
    # }
    #
    # print(net1_wiring.keys())
    #
    # net1 = LIFNet([input_neuron, n1], net1_wiring)
    # net1.update_network(time)
    #
    # plt.subplot(4, 1, 1)
    # plt.plot(time, curr)
    # plt.subplot(4, 1, 2)
    # plt.plot(time, input_neuron.output)
    # plt.subplot(4, 1, 3)
    # plt.plot(time, n1.voltage)
    # plt.subplot(4, 1, 4)
    # plt.plot(time, n1.spikes)
    #
    # plt.show()

    # # Test LIFNet (WORKS)
    # # Get timeframe, in seconds
    # t_start = 0.0
    # t_stop = 0.05
    # mils = int(1000 * (t_stop - t_start))  # ensures that time step is 1 millisecond
    # time = np.linspace(t_start, t_stop, num=mils, endpoint=False)
    # #
    # # Define external current flowing into the network
    # curr = np.zeros(len(time))
    # spikes = 10
    # if spikes != 0:
    #     for i in range(spikes):
    #         curr[i * int(len(time) / spikes)] = 0.012
    # input_neuron = LIFNeuron(name="input current")
    # input_neuron.output = curr
    #
    # n1 = LIFNeuron(name="n1")
    # n2 = LIFNeuron(name="n2")
    #
    # net1_wiring = {
    #     input_neuron: {},
    #     n1: {input_neuron: 1.0, n2: 1.0},
    #     n2: {input_neuron: 1.0, n1: -10.0}
    # }
    #
    # print(net1_wiring.keys())
    #
    # net1 = LIFNet([input_neuron, n1, n2], net1_wiring)
    # net1.update_network(time)
    #
    # plt.subplot(5, 1, 1)
    # plt.plot(time, input_neuron.output)
    # plt.subplot(5, 1, 2)
    # plt.plot(time, n1.voltage)
    # plt.subplot(5, 1, 3)
    # plt.plot(time, n1.spikes)
    # plt.subplot(5, 1, 4)
    # plt.plot(time, n2.voltage)
    # plt.subplot(5, 1, 5)
    # plt.plot(time, n2.spikes)
    #
    # plt.show()

