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


def main():
    show_neurons()


if __name__ == "__main__":
    main()
