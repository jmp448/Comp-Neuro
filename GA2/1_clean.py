import numpy as np
import matplotlib.pyplot as plt
import math


class NeuralNet:
    def __init__(self, neurons, wiring):
        """
        :param neurons: list of neurons in the net
        :param wiring: map of maps, maps a neuron onto an array of its
                       inputs and the weight for each synapse
        """
        self.neurons = neurons
        self.wiring = wiring
        self.weight_tracker = None

    def update_network(self, time, curr_in=None,
                       one_way_learning=False,
                       two_way_learning=False,
                       learning_rate=1,
                       tracking_weights=None):
        for n in self.neurons:
            if n.voltage is None:
                n.voltage = np.full(len(time), n.rp)
            if n.spikes is None:
                n.spikes = np.zeros(len(time))
            if n.output is None:
                n.output = np.zeros(len(time))

        tstep = np.ptp(time) / len(time)

        if tracking_weights is not None:
            self.weight_tracker = np.full(len(time), self.wiring[tracking_weights[0]][tracking_weights[1]])

        for tpoint in range(len(time)):
            for n in self.neurons:
                if len(self.wiring[n].keys()) > 0:
                    ins = self.wiring[n].keys()
                    in_volts = []
                    weights = []
                    for neuron in ins:
                        in_volts.append(neuron.output[tpoint - 1])
                        weights.append(self.wiring[n][neuron])
                    v_in = np.matmul(weights, in_volts)
                    n.update(time, tpoint, v_in)

            if one_way_learning:
                for n1 in self.neurons:
                    for n2 in self.wiring[n1]:
                        if sum(n1.output[tpoint-1:tpoint+1]) > 0 and sum(n2.output[tpoint-1:tpoint+1]) > 0:
                            print("it happened")
                            self.wiring[n1][n2] += learning_rate
                            self.weight_tracker[tpoint:] = self.wiring[n1][n2]

            if two_way_learning:
                for n1 in self.neurons:
                    for n2 in self.wiring[n1]:
                        if sum(n1.output[tpoint-1:tpoint+1]) > 0 and sum(n2.output[tpoint-1:tpoint+1]) > 0:
                            print("it happened")
                            self.wiring[n1][n2] += learning_rate
                            self.weight_tracker[tpoint:] = self.wiring[n1][n2]
                        else:
                            self.wiring[n1][n2] -= learning_rate
                            if self.wiring[n1][n2] < 0:
                                self.wiring[n1][n2] = 0
                            self.weight_tracker[tpoint:] = self.wiring[n1][n2]


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

    def initialize(self, time):
        self.voltage = np.full(len(time), self.rp)
        self.spikes = np.zeros(len(time))
        self.output = np.zeros(len(time))

    def update(self, time, tpoint, v_in):
        tstep = np.ptp(time) / len(time)

        # for tpoint in range(1, len(time)):
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
                                   + v_in * (1-math.exp(-tstep/self.tc))


class PNeuron:
    def __init__(self, name=None, max_spikes=10, spike_output=0.0012):
        self.name = name
        self.max_spikes = max_spikes
        self.spikes = None
        self.output = None
        self.spike_output = spike_output
        self.voltage = None
        self.rp = 0.0

    def response(self, time, angle, spike_output=0.0012):
        response = 0
        if self.name == "p1":
            if angle < 0:
                response = -1 * (angle + 225.0) * (angle + 45.0) / 8100.0
            else:
                response = -1 * (angle - 135.0) * (angle - 315.0) / 8100.0
            if response < 0:
                response = 0.0

        elif self.name == "p2":
            response = -1 * (angle + 135.0) * (angle - 45.0) / 8100.0
            if response < 0:
                response = 0.0

        elif self.name == "p3":
            response = -1 * (angle - 135.0) * (angle + 45.0) / 8100.0
            if response < 0:
                response = 0.0

        elif self.name == "p4":
            if angle > 0:
                response = -1 * (angle - 45.0) * (angle - 225.0) / 8100.0
            else:
                response = -1 * (angle + 135.0) * (angle + 315.0) / 8100.0
            if response < 0:
                response = 0

        tstep = np.ptp(time) / len(time)
        if response > 0:
            interval = np.ptp(time) / (response * self.max_spikes)
        else:
            interval = np.inf

        self.spikes = np.zeros(len(time))
        self.output = np.zeros(len(time))
        clock = 0
        for j in range(len(time)):
            if clock >= interval:
                clock = 0
                self.spikes[j] = 1
                self.output[j] = self.spike_output
            else:
                clock += tstep

    def update(self, time, tpoint, v_in):
        pass


class RewardNeuron:
    def __init__(self, spike_output=0.0012, sync=None):
        self.spike_output = spike_output
        self.voltage = None
        self.rp = 0.0
        self.spikes = None
        if sync is None:
            self.output = None
        else:
            self.output = sync.output

    def initialize(self, time):
        self.output = np.zeros(len(time))

    def update(self, time, tpoint, v_in):
        pass


class MCPNeuron:
    def __init__(self, name=None, threshold=0.005,
                 spike_output=0.0012,
                 noisy=False, noise_range=0.005):
        self.name = name
        self.threshold = threshold
        self.spike_output = spike_output
        self.noisy = noisy
        self.noise_range = noise_range
        self.spikes = None
        self.output = None
        self.voltage = None
        self.rp = None

    def initialize(self, time):
        self.spikes = np.zeros(len(time))
        self.output = np.zeros(len(time))

    def update(self, time, tpoint, v_in):
        if self.noisy:
            threshold_range = np.linspace(self.threshold - self.noise_range / 2,
                                          self.threshold + self.noise_range / 2, 10)
            threshold = np.random.choice(threshold_range)
        else:
            threshold = self.threshold

        if v_in >= threshold:
            self.spikes[tpoint] = 1
            self.output[tpoint] = self.spike_output


def display_response_curves(p1, p2, p3, p4,
                            angles=[-180, -135, -90, -45, 0, 45, 90, 135, 180]):
    # Get timeframe, in seconds
    t_start = 0.0
    t_stop = 0.05
    mils = int(1000 * (t_stop - t_start))  # ensures that time step is 1 millisecond
    time = np.linspace(t_start, t_stop, num=mils, endpoint=False)

    num_plots = 4 * len(angles)
    print(num_plots)
    for i in range(len(angles)):
        for j in range(4):
            if j == 0 and i == 0:
                plt.subplot(len(angles), 4, i * 4 + j + 1, xticks=[], yticks=[], ylabel="%d" % angles[i],
                            title="Neuron %d" % (j + 1))
            elif i == 0:
                plt.subplot(len(angles), 4, i * 4 + j + 1, xticks=[], yticks=[], title="Neuron %d" % (j + 1))
            elif j == 0:
                plt.subplot(len(angles), 4, i * 4 + j + 1, xticks=[], yticks=[], ylabel="%d" % angles[i])
            else:
                plt.subplot(len(angles), 4, i * 4 + j + 1, xticks=[], yticks=[])

            if j == 0:
                p1.response(time, angles[i])
                plt.plot(time, p1.output)
            elif j == 1:
                p2.response(time, angles[i])
                plt.plot(time, p2.output)
            elif j == 2:
                p3.response(time, angles[i])
                plt.plot(time, p3.output)
            else:
                p4.response(time, angles[i])
                plt.plot(time, p4.output)
    plt.show()


def test_LIFNeuron():
    # Get timeframe, in seconds
    t_start = 0.0
    t_stop = 0.05
    mils = int(1000 * (t_stop - t_start))  # ensures that time step is 1 millisecond
    time = np.linspace(t_start, t_stop, num=mils, endpoint=False)

    # Define external current flowing into the network
    curr = np.zeros(len(time))
    spikes = 10
    if spikes != 0:
        for i in range(spikes):
            curr[i * int(len(time) / spikes)] = 0.012

    testy = LIFNeuron(noisy=True)
    testy.initialize(time)
    for tpoint in range(1, len(time)):
        testy.update_voltage(time, tpoint, curr[tpoint])

    plt.subplot(3, 1, 1)
    plt.plot(time, curr)
    plt.subplot(3, 1, 2)
    plt.plot(time, testy.voltage)
    plt.subplot(3, 1, 3)
    plt.plot(time, testy.spikes)

    plt.show()


def test_LIFNet():
    # Get timeframe, in seconds
    t_start = 0.0
    t_stop = 0.05
    mils = int(1000 * (t_stop - t_start))  # ensures that time step is 1 millisecond
    time = np.linspace(t_start, t_stop, num=mils, endpoint=False)

    # Define external current flowing into the network
    curr = np.zeros(len(time))
    spikes = 10
    if spikes != 0:
        for i in range(spikes):
            curr[i * int(len(time) / spikes)] = 0.012
    input_neuron = LIFNeuron(name="input current")
    input_neuron.output = curr

    n1 = LIFNeuron(name="n1")

    net1_wiring = {
        input_neuron: {},
        n1: {input_neuron: 1.0}
    }

    print(net1_wiring.keys())

    net1 = LIFNet([input_neuron, n1], net1_wiring)
    net1.update_network(time)

    plt.subplot(4, 1, 1)
    plt.plot(time, curr)
    plt.subplot(4, 1, 2)
    plt.plot(time, input_neuron.output)
    plt.subplot(4, 1, 3)
    plt.plot(time, n1.voltage)
    plt.subplot(4, 1, 4)
    plt.plot(time, n1.spikes)

    plt.show()


if __name__ == "__main__":

    # test_LIFNeuron()
    # test_LIFNet()

    # Get timeframe, in seconds
    t_start = 0.0
    t_stop = 0.05
    mils = int(1000 * (t_stop - t_start))  # ensures that time step is 1 millisecond
    print(mils)
    time = np.linspace(t_start, t_stop, num=mils, endpoint=False)

    # Create plot of response curves
    p1 = PNeuron(name="p1")
    p2 = PNeuron(name="p2")
    p3 = PNeuron(name="p3")
    p4 = PNeuron(name="p4")

    # display_response_curves(p1, p2, p3, p4)

    p1.response(time, -175)
    p2.response(time, -175)
    p3.response(time, -175)
    p4.response(time, -175)

    x1 = MCPNeuron()
    reward = RewardNeuron(sync=p1)
    # reward.output[]

    synapses = {
        p1: {},
        p2: {},
        p3: {},
        p4: {},
        x1: {p1: 1, p2: 1, p3: 1, p4: 1, reward: 10},
        reward: {}
    }

    net = NeuralNet([p1, p2, p3, p4, x1, reward], synapses)
    net.update_network(time, one_way_learning=True, tracking_weights=(x1, p1))

    plt.subplot(6, 1, 1, xticks=[], yticks=[])
    plt.plot(time, p1.output)

    plt.subplot(6, 1, 2, xticks=[], yticks=[])
    plt.plot(time, p2.output)

    plt.subplot(6, 1, 3, xticks=[], yticks=[])
    plt.plot(time, p3.output)

    plt.subplot(6, 1, 4, xticks=[], yticks=[])
    plt.plot(time, p4.output)

    plt.subplot(6, 1, 5, xticks=[], yticks=[])
    plt.plot(time, reward.output)

    plt.subplot(6, 1, 6, xticks=[], yticks=[])
    plt.plot(time, x1.output)

    # plt.show()



    plt.plot(time, net.weight_tracker)

    # plt.show()

    # synapses = net.wiring
    
