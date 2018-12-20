import numpy as np
import matplotlib.pyplot as plt


def p1(theta):
    if theta < 0:
        response = -1 * (theta + 225.0) * (theta + 45.0) / 8100.0
    else:
        response = -1 * (theta - 135.0) * (theta - 315.0) / 8100.0
    if response < 0:
        response = 0.0
    return response


def p2(theta):
    response = -1 * (theta + 135.0) * (theta - 45.0) / 8100.0
    if response < 0:
        response = 0.0
    return response


def p3(theta):
    response = -1 * (theta - 135.0) * (theta + 45.0) / 8100.0
    if response < 0:
        response = 0.0
    return response


def p4(theta):
    if theta > 0:
        response = -1 * (theta - 45.0) * (theta - 225.0) / 8100.0
    else:
        response = -1 * (theta + 135.0) * (theta + 315.0) / 8100.0
    if response < 0:
        response = 0
    return response


def def_p_neurons():

    x = np.linspace(-180, 180, 1000)
    y1 = np.zeros(len(x))
    y2 = np.zeros(len(x))
    y3 = np.zeros(len(x))
    y4 = np.zeros(len(x))

    for i in range(len(x)):

        # Neuron 1
        y1[i] = p1(x[i])

        # Neuron 2
        y2[i] = p2(x[i])

        # Neuron 3
        y3[i] = p3(x[i])

        # Neuron 4
        y4[i] = p4(x[i])

    return x, y1, y2, y3, y4


def show_neurons():
    x, y1, y2, y3, y4 = def_p_neurons()
    for y in [y1, y2, y3, y4]:
        plt.plot(x, y)

    plt.xlim(left=-180, right=180)
    plt.xlabel('Stimulus location (angle in degrees)')
    plt.ylim(bottom=0, top=1)
    plt.ylabel('Normalized spike count')
    plt.title('Neuron Tuning Curves')
    plt.show()


def p_spike_train(time, theta, neuron, max_spikes=10, show=False):
    elapsed = np.ptp(time)
    if neuron == 1:
        if p1(theta) > 0:
            interval = elapsed / (p1(theta) * max_spikes)
        else:
            interval = np.inf
    elif neuron == 2:
        if p2(theta) > 0:
            interval = elapsed / (p2(theta) * max_spikes)
        else:
            interval = np.inf
    elif neuron == 3:
        if p3(theta) > 0:
            interval = elapsed / (p3(theta) * max_spikes)
        else:
            interval = np.inf
    else:
        if p4(theta) > 0:
            interval = elapsed / (p4(theta) * max_spikes)
        else:
            interval = np.inf

    spikes = np.zeros(len(time))
    clock = 0
    for i in range(len(time)):
        if clock >= interval:
            clock = 0
            spikes[i] = 1
        else:
            spikes[i] = 0
            clock += elapsed / 1000

    if show:
        plt.plot(time, spikes)
        plt.show()

    return spikes


def plot_spike_trains():
    time = np.linspace(0, 1, 1000)
    angles = [-150, -105, -60, -15, 30, 75]
    spikes = np.zeros([len(angles), len(time)])

    for i in range(6):
        plt.subplot(6, 1, i + 1, xticks=[], yticks=[], ylabel="%d" % angles[i])
        spikes[i] = p_spike_train(time, angles[i], 2)
        plt.plot(spikes[i])
    plt.show()


def main():
    show_neurons()
    plot_spike_trains()


if __name__ == "__main__":
    main()
