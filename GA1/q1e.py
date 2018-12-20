import numpy as np
import matplotlib.pyplot as plt
import math
from q1a import *
from q1b import *


class PNeuron:
    def __init__(self, num, output_mag=0.1):
        self.num = num
        self.output_mag = output_mag
        self.output = None

    def respond(self, theta, max_spikes=10, time=np.linspace(0, 1, 1000)):
        self.output = self.output_mag * p_spike_train(time, theta, self.num, max_spikes=max_spikes)


class MNeuron:
    def __init__(self, wiring,
                 tau=30, threshold=0.00385,
                 spike_mag=0.12, rp=0.0):
        self.inputs = list(wiring)
        self.wiring = wiring
        self.tau = tau
        self.threshold = threshold
        self.rp = rp
        self.voltage = None
        self.output = None
        self.spike_mag = spike_mag
        self.spikes_per_sec = None

    def respond(self):
        time = np.linspace(0, 1, 1000)
        t_step = 0.001

        v_in = np.zeros(len(time))
        for n in self.inputs:
            w = self.wiring[n]
            v_in = np.add(v_in, w*n.output)

        v = np.full(len(time), self.rp)
        for t in range(1, len(time)):
            if v[t-1] >= self.threshold:
                v[t] = self.rp
            else:
                v[t] = v[t-1] * math.exp(-1.0/self.tau) + (1-math.exp(-1.0/self.tau)) * v_in[t]
        self.voltage = v

        self.output = np.zeros(len(time))
        for i in range(len(self.voltage)):
            if self.voltage[i] >= self.threshold:
                self.output[i] = self.spike_mag

        self.spikes_per_sec = sum(self.output*1.0/self.spike_mag)


def wire_the_brain(p):
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]

    wiring = {}

    w1 = 1.0
    w2 = 7.0/9.0
    w3 = 3.5/9.0

    wiring[1] = {
        p1: w2,
        p2: 0,
        p3: 0,
        p4: w3
    }
    m1 = MNeuron(wiring[1])

    wiring[2] = {
        p1: w1,
        p2: 0,
        p3: 0,
        p4: 0
    }
    m2 = MNeuron(wiring[2])

    wiring[3] = {
        p1: w2,
        p2: w3,
        p3: 0,
        p4: 0
    }
    m3 = MNeuron(wiring[3])

    wiring[4] = {
        p1: w3,
        p2: w2,
        p3: 0,
        p4: 0
    }
    m4 = MNeuron(wiring[4])

    wiring[5] = {
        p1: 0,
        p2: w1,
        p3: 0,
        p4: 0
    }
    m5 = MNeuron(wiring[5])

    wiring[6] = {
        p1: 0,
        p2: w2*1.2,
        p3: w3/1.2,
        p4: 0
    }
    m6 = MNeuron(wiring[6])

    wiring[7] = {
        p1: 0,
        p2: w3*1.2,
        p3: w2/1.2,
        p4: 0
    }
    m7 = MNeuron(wiring[7])

    wiring[8] = {
        p1: 0,
        p2: 0,
        p3: w1,
        p4: 0
    }
    m8 = MNeuron(wiring[8])

    wiring[9] = {
        p1: 0,
        p2: 0,
        p3: w2,
        p4: w3
    }
    m9 = MNeuron(wiring[9])

    wiring[10] = {
        p1: 0,
        p2: 0,
        p3: w3,
        p4: w2
    }
    m10 = MNeuron(wiring[10])

    wiring[11] = {
        p1: 0,
        p2: 0,
        p3: 0,
        p4: w1
    }
    m11 = MNeuron(wiring[11])

    wiring[12] = {
        p1: w3,
        p2: 0,
        p3: 0,
        p4: w2
    }
    m12 = MNeuron(wiring[12])

    m = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]
    return m, wiring


def main():
    p1 = PNeuron(1)
    p2 = PNeuron(2)
    p3 = PNeuron(3)
    p4 = PNeuron(4)
    p = [p1, p2, p3, p4]
    m, wiring = wire_the_brain(p)

    angles = np.linspace(-180, 180, 100)
    responses = np.zeros([len(m), len(angles)])
    for i in range(len(angles)):
        theta = angles[i]
        for n in p:
            n.respond(theta, max_spikes=20)
        for j in range(len(m)):
            m[j].respond()
            responses[j][i] = m[j].spikes_per_sec

    # Normalize spike count
    for i in range(len(m)):
        norm = max(responses[i])
        for j in range(len(responses[i])):
            responses[i][j] /= norm

    # Implement winner-takes-all inhibition
    for i in range(len(angles)):
        winner = 0
        for j in range(len(m)):
            if responses[j][i] > responses[winner][i]:
                winner = j
        print(winner)
        for j in range(len(m)):
            if j != winner:
                responses[j][i] = 0.0
    print(responses)

    for i in range(len(m)):
        plt.plot(angles, responses[i])
        plt.xlabel("Stimulus location (angle in degrees)")
        plt.ylabel("Normalized spike count")
    plt.show()


if __name__ == "__main__":
    main()
