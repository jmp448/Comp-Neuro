from q1 import p1, p2, p3, p4


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

    def __init__(self, wiring, threshold=1.5):
        self.inputs = list(wiring)
        self.wiring = wiring
        self.output = 0.0
        self.threshold = threshold

    def respond(self):
        input = 0
        for n in self.inputs:
            input += n.output * self.wiring[n]
        if input >= self.threshold:
            self.output = 1.0


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
        reward: 1.5
    }

    np = MCPNeuron(wiring, threshold=1.5)

    for n in p:
        n.respond(-45)

    np.respond()
    print(np.output)

    reward.activate()

    np.respond()
    print(np.output)


if __name__ == "__main__":
    main()
