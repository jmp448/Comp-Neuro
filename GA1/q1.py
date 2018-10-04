import numpy as np
import matplotlib.pyplot as plt


def r1(theta):
    response = -1 * (theta + 225) * (theta + 45) / 8100
    if response < 0:
        response = 0
    return response


def r2(theta):
    response = -1 * (theta + 135) * (theta - 45) / 8100
    if response < 0:
        response = 0
    return response


def r3(theta):
    response = -1 * (theta - 135) * (theta + 45) / 8100
    if response < 0:
        response = 0
    return response


def r4(theta):
    response = -1 * (theta - 45) * (theta - 225) / 8100
    if response < 0:
        response = 0
    return response


def def_neurons():

    x = np.linspace(-180, 180, 1000)
    y1 = np.zeros(len(x))
    y2 = np.zeros(len(x))
    y3 = np.zeros(len(x))
    y4 = np.zeros(len(x))

    for i in range(len(x)):
        y1[i] = r1(x[i])

        # Neuron 2
        y2[i] = r2(x[i])

        # Neuron 3
        y3[i] = r3(x[i])

        # Neuron 4
        y4[i] = r4(x[i])

    return x, y1, y2, y3, y4


def show_neurons(x, y1, y2, y3, y4):

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.plot(x, y4)

    plt.xlim(left=-180, right=180)
    plt.xlabel('Stimulus location (angle in degrees)')
    plt.ylim(bottom=0, top=1)
    plt.ylabel('Normalized spike count')
    plt.title('Neuron Tuning Curves')
    plt.show()


def plot_spike_train(t, theta, neuron, max_spikes=10):
    if neuron == 1:
        if r1(theta) > 0:
            interval = float(t)/(r1(theta)*max_spikes)
        else:
            interval = np.inf
    elif neuron == 2:
        if r2(theta) > 0:
            interval = float(t)/(r2(theta)*max_spikes)
        else:
            interval = np.inf
    elif neuron == 3:
        if r3(theta) > 0:
            interval = float(t)/(r3(theta)*max_spikes)
        else:
            interval = np.inf
    else:
        if r4(theta) > 0:
            interval = float(t)/(r4(theta)*max_spikes)
        else:
            interval = np.inf

    print(interval)
    t_range = np.linspace(0, t, 1000)
    spikes = np.zeros(len(t_range))
    clock = 0
    for i in range(len(t_range)):
        if clock >= interval:
            clock = 0
            spikes[i] = 1
        else:
            spikes[i] = 0
            clock += float(t)/1000

    plt.plot(t_range, spikes)


def pop_vec(a):
    m2 = [-135, -45, 45, 135]

    return np.matmul(a, m2)


if __name__ == "__main__":

    # Q1
    # x, y1, y2, y3, y4 = def_neurons()
    # show_neurons(x, y1, y2, y3, y4)

    # plot_spike_train(1, -50, 2)

    # angles = [-150, -105, -60, -15, 30, 75]
    #
    # for i in range(6):
    #     plt.subplot(6, 1, i+1, xticks=[], yticks=[], ylabel="%d" % angles[i])
    #     plot_spike_train(1, angles[i], 2)
    #
    # plt.show()

    # Q2
    print(pop_vec([0, 0, 1, 0]))

