import numpy as np
import matplotlib.pyplot as plt
import math


class NoisyNeuron:
    def __init__(self, name=None, freq=120):
        self.name = name
        self.freq = freq
        self.spikes = None
        self.spike_count = 0

    def initialize(self, time):
        self.spikes = np.zeros(len(time))

    def create_spike_train(self):
        t_range = np.ptp(time)
        mils = t_range * 1000
        for t_point in range(len(time)):
            chance = np.random.random_sample()
            if chance < self.freq / mils:
                self.spikes[t_point] = 1.0
                self.spike_count += 1


def run_7a(time):
    # Create the two neurons
    n1 = NoisyNeuron("n1")
    n2 = NoisyNeuron("n2")

    # Create spike trains (120 Hz frequency) for the neurons
    n1.initialize(time)
    n1.create_spike_train()
    spikes1 = n1.spikes

    n2.initialize(time)
    n2.create_spike_train()
    spikes2 = n2.spikes

    # 7a Noisy Neurons
    plt.subplot(2, 1, 1, xticks=[], yticks=[], ylabel="Neuron 1")
    plt.plot(time, spikes1)

    plt.subplot(2, 1, 2, xticks=[], yticks=[], xlabel="Time", ylabel="Neuron 2")
    plt.plot(time, spikes2)

    print("Neuron 1: %d\nNeuron 2: %d" % (n1.spike_count, n2.spike_count))
    plt.show()


def run_7b(time):
    # 7b Learning
    freqs = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    changes = []

    alpha = 0.15

    for f in freqs:
        n1 = NoisyNeuron(freq=f)
        n2 = NoisyNeuron(freq=f)

        n1.initialize(time)
        n1.create_spike_train()
        spikes1 = n1.spikes

        n2.initialize(time)
        n2.create_spike_train()
        spikes2 = n2.spikes

        w = 0.0
        for t in range(len(time)):
            w += alpha * spikes1[t] * spikes2[t]

        changes.append(w)

    plt.plot(freqs, changes)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Change in Synaptic Weight")
    plt.title("Learning vs Frequency (Learning Rate = 0.15)")
    plt.show()


def run_7c(time):
    # 7c Wider Windows
    windows = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    changes = []
    alpha = 0.15

    n1 = NoisyNeuron(freq=120)
    n1.initialize(time)
    n1.create_spike_train()
    spikes1 = n1.spikes
    n2 = NoisyNeuron(freq=120)
    n2.initialize(time)
    n2.create_spike_train()
    spikes2 = n2.spikes

    for w in windows:
        c = 0.0
        for t in range(len(time)):
            if spikes1[t] == 1:
                low = int(t - w / 2.0)
                high = int(t + w / 2.0)
                if t < low:
                    if sum(spikes2[0:t]) >= 1:
                        c += alpha
                elif low <= t <= high:
                    if sum(spikes2[low:high]) >= 1:
                        c += alpha
                elif t > len(time) - w / 2.0:
                    if sum(spikes2[high:len(time) - 1]) >= 1:
                        c += alpha
        changes.append(c)
    plt.plot(windows, changes)
    plt.xlabel("Window Range (ms)")
    plt.ylabel("Change in Synaptic Weight")
    plt.title("Learning vs Window Range (Learning Rate = 0.15)")
    plt.show()


def run_7d(time):
    # 7d STDP
    freqs = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    changes = []
    for f in freqs:
        c = 1.0
        n1 = NoisyNeuron(freq=f)
        n2 = NoisyNeuron(freq=f)

        n1.initialize(time)
        n1.create_spike_train()
        spikes1 = n1.spikes

        n2.initialize(time)
        n2.create_spike_train()
        spikes2 = n2.spikes
        for t in range(len(time)):
            if spikes2[t] == 1:
                # nearest neighbor before
                if sum(spikes1[max(0, t - 5):t]) >= 1:
                    c *= 1.4
                    print("biggest ups\n")
                elif sum(spikes1[max(0, t - 10):t]) >= 1:
                    c *= 1.2
                    print("bigger ups\n")
                elif sum(spikes1[max(0, t - 20):t]) >= 1:
                    c *= 1.1
                    print("big ups\n")

                # nearest neighbor after
                if sum(spikes1[t:min(len(time) - 1, t + 5)]) >= 1:
                    c *= 0.6
                    print("biggest downs\n")
                elif sum(spikes1[t:min(len(time) - 1, t + 10)]) >= 1:
                    c *= 0.8
                    print("bigger downs\n")
                elif sum(spikes1[t:min(len(time) - 1, t + 20)]) >= 1:
                    c *= 0.9
                    print("big downs\n")
        changes.append(c)

    plt.plot(freqs, changes)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Change in Synaptic Weight")
    plt.title("STDP (Initial Weight = 1.0)")
    plt.show()


if __name__ == "__main__":

    # Get timeframe, in seconds
    t_start = 0.0
    t_stop = 1.0
    mils = int(1000 * (t_stop - t_start))  # ensures that time step is 1 millisecond
    time = np.linspace(t_start, t_stop, num=mils, endpoint=False)

    run_7a(time)
    run_7b(time)
    run_7c(time)
    run_7d(time)
