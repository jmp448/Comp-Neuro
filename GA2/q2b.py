from q1 import p1, p2, p3, p4
import matplotlib.pyplot as plt
import numpy


class PNeuron:

    def __init__(self, num):
        self.num = num
        self.output = None

    def respond(self, theta):
        if self.num == 1:
            self.output = p1(theta)
        elif self.num == 2:
            self.output = p2(theta)
        elif self.num == 3:
            self.output = p3(theta)
        elif self.num == 4:
            self.output = p4(theta)


class RewardNeuron:

    def __init__(self):
        self.active = False
        self.output = 0.0

    def activate(self):
        self.active = True
        self.output = 1.0

    def inactivate(self):
        self.active = False
        self.output = 0.0


class MCPNeuron:

    def __init__(self, wiring, threshold=1.75,
                 learning=False, learning_rate=0.15):
        self.inputs = list(wiring)
        self.wiring = wiring
        self.input = 0
        self.output = 0.0
        self.threshold = threshold
        self.learning = learning
        self.learning_rate = learning_rate

    def respond(self, reward):
        for n in self.inputs:
            self.input += n.output * self.wiring[n]
        if self.input >= self.threshold:
            self.output = 1.0

        if self.learning:
            for n in self.inputs:
                if n != reward and n.output > 0 and reward.output > 0:
                    self.wiring[n] += self.learning_rate * n.output * reward.output


def main():

    p1 = PNeuron(1)
    p2 = PNeuron(2)
    p3 = PNeuron(3)
    p4 = PNeuron(4)
    p = [p1, p2, p3, p4]

    reward = RewardNeuron()

    wiring = {
        p1: 1,
        p2: 1,
        p3: 1,
        p4: 1,
        reward: 1.75
    }

    for n in p:
        n.respond(-45)

    reward.activate()

    np = MCPNeuron(wiring, threshold=1.75, learning=True, learning_rate=0.15)

    s1 = numpy.zeros(10)
    s2 = numpy.zeros(10)
    s3 = numpy.zeros(10)
    s4 = numpy.zeros(10)
    activity = numpy.zeros(10)

    s1[0] = np.wiring[p1]
    s2[0] = np.wiring[p2]
    s3[0] = np.wiring[p3]
    s4[0] = np.wiring[p4]
    activity[0] = np.output

    for epoch in range(1, 10):
        np.respond(reward)
        s1[epoch] = np.wiring[p1]
        s2[epoch] = np.wiring[p2]
        s3[epoch] = np.wiring[p3]
        s4[epoch] = np.wiring[p4]
        activity[epoch] = np.output

    plt.subplot(5, 1, 1, xticks=[], yticks=[])
    plt.ylabel("P1")
    plt.plot(range(10), s1)
    plt.subplot(5, 1, 2, xticks=[], yticks=[])
    plt.ylabel("P2")
    plt.plot(range(10), s2)
    plt.subplot(5, 1, 3, xticks=[], yticks=[])
    plt.ylabel("P3")
    plt.plot(range(10), s3)
    plt.subplot(5, 1, 4, xticks=[], yticks=[])
    plt.ylabel("P4")
    plt.plot(range(10), s4)
    plt.subplot(5, 1, 5, yticks=[])
    plt.xlabel("Iteration")
    plt.ylabel("Output")
    plt.plot(range(10), activity)

    plt.show()

    np.learning = False
    reward.inactivate()

    np.input = 0
    np.output = 0
    angles = numpy.linspace(-180, 180, 100)
    responses = numpy.zeros(len(angles))
    for i in range(len(angles)):
        theta = angles[i]
        for n in p:
            n.respond(theta)
        assert(reward.output == 0)
        np.respond(reward)
        print("for angle %f input is %f" % (theta, np.input))
        responses[i] = np.output
        np.input = 0
        np.output = 0

    plt.plot(angles, responses)
    plt.xlabel("Stimulus (angle in degrees)")
    plt.ylabel("Response")
    plt.show()


if __name__ == "__main__":
    main()
