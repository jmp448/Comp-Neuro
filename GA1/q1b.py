import numpy as np
import matplotlib.pyplot as plt
import math
from q1a import *


def pop_vec(response):
    """
    Combine the responses of 4 P neurons (sensory neurons) to calculate the angle that caused the response

    :param response: array of responses of 4 neurons p1, p2, p3, p4
    :return: the angle that caused the response, in degrees
    """

    # Normalize the measured spike values
    tot = 0.0
    for i in range(4):
        tot += response[i]
    for i in range(len(response)):
        response[i] /= tot

    # Encode the "preferences" of each individual neuron, aka their "vote"
    # Handle the complex situation where 1 and 4 are simultaneously active and 2 and 3 are not at all
    if (response[0] > 0 and response[3] > 0) and (response[1] == 0.0 and response[2] == 0.0):
        if response[0] > response[3]:
            prefs = [-135.0, -45.0, 45.0, -225.0]
        elif response[0] == response[3]:
            return 180.0
        else:
            prefs = [225.0, -45.0, 45.0, 135.0]

    # Simpler situation (everyone gets their normal vote)
    else:
        prefs = [-135.0, -45.0, 45.0, 135.0]

    return np.matmul(response, prefs)


def main():
    print(pop_vec([0.4784, 0, 0, 0.9228]))


if __name__ == "__main__":
    main()
