import math
import matplotlib.pyplot as plt
import numpy as np
import csv


class HopfieldNet:
    def __init__(self, n, w):
        self.neurons = range(n)
        self.voltages = np.zeros(n, int)
        self.weights = w

    def update(self, inputs):
        # # print(activity)
        self.voltages = np.matmul(inputs, self.weights)
        activity = np.zeros(len(inputs))
        # # print(self.voltages)
        # # print(self.voltages)
        for n in self.neurons:
            if self.voltages[n] > 0:
                activity[n] = 1
            elif self.voltages[n] < 0:
                activity[n] = -1
            else:
                activity[n] = inputs[n]
        return activity


def print_grid(g):
    for row in g:
        for e in row:
            if e == 1:
                print("o ", end="")
            else:
                print("- ", end="")
        print("", end="\n")


def print_array2grid(a):
    dim = int(np.sqrt(len(a)))

    for i in range(dim):
        for j in range(dim):
            pos = dim*i+j
            if a[pos] == 1:
                print("o ", end="")
            else:
                print("- ", end="")
        print("", end="\n")


def create_weights_matrix(n=49, filename="weights_matrix"):
    # Create upper right triangle of weights matrix
    weights = np.zeros([n, n], int)
    for i in range(n):
        for j in range(i + 1, n):
            weights[i][j] = np.random.choice([0, 1], p=[0.5, 0.5])

    # Fill in lower right triangle to make it symmetric
    for i in range(n):
        for j in range(0, i):
            weights[i][j] = weights[j][i]

    # Print to a space-delimited file
    with open(filename, "w") as f:
        for row in range(n):
            for col in range(n):
                f.write("%d " % weights[row][col])
            f.write("\n")


def create_starter(n=49, filename="initial_activity", smile=False):
    if smile:
        starter = np.full(49, -1, int)
        dots = [8, 13, 22, 28, 39]
        for d in dots:
            starter[d] = 1
    else:
        starter = np.random.choice([-1, 1], size=49)

    # Print to a space-delimited file
    with open(filename, "w") as f:
        for i in range(n):
            f.write("%d " % starter[i])
        f.write("\n")


def read_weights_matrix(filename="weights_matrix"):
    with open(filename) as f:
        weights = list(csv.reader(f, delimiter=' '))
        for row in range(len(weights)):
            for col in range(len(weights[row])):
                if weights[row][col] != "":
                    weights[row][col] = int(weights[row][col])
                else:
                    weights[row] = weights[row][:col]
    for r in range(len(weights)):
        weights[r] = np.array(weights[r], dtype=int)
    return weights


def read_starter(filename="initial_activity"):
    with open(filename) as f:
        starter = list(csv.reader(f, delimiter=' '))
        starter = starter[0]
        for pos in range(len(starter)):
            if starter[pos] != "":
                starter[pos] = int(starter[pos])
            else:
                starter = starter[:pos]
    starter = np.array(starter, dtype=int)
    return starter


def energy(activity, weights):
    e = 0
    volt = np.matmul(activity, weights)
    for k in range(49):
        for n in range(49):
            e = e + weights[n][k] * volt[n] * volt[k]
    e = -0.5 * e
    return e


def test_updates():
    file = "test_weights"
    weights = read_weights_matrix(file)
    test_net = HopfieldNet(3, weights)
    case1 = [1, 1, 1]
    case2 = [-1, 1, 1]
    case3 = [-1, -1, 1]
    case4 = [-1, -1, -1]

    if np.array_equal(test_net.update(case1), [1, 1, 1]):
        test1 = True
    if np.array_equal(test_net.update(case2), [1, 1, 1]):
        test2 = True
    if np.array_equal(test_net.update(case3), [-1, -1, -1]):
        test3 = True
    if np.array_equal(test_net.update(case4), [-1, -1, -1]):
        test4 = True

    if test1 and test2 and test3 and test4:
        print("The update function is working")


def run_1a():

    weights = read_weights_matrix()
    face_net = HopfieldNet(49, weights)
    initial = read_starter()

    distortion = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for d in range(len(distortion)):
        print("Distortion %.1f:\n" % distortion[d])
        distorted_start = initial
        changes = np.random.choice(range(49), size=int(distortion[d]*len(initial)),
                                   replace=False)
        for c in changes:
            if initial[c] == 1:
                distorted_start[c] = -1
            else:
                distorted_start[c] = 1
        starter = distorted_start

        # Update until reaching a stable state
        prev = np.zeros(49, int)
        curr = starter
        while not np.array_equal(prev, curr):
            prev = curr
            print("Energy level: %.4f" % energy(curr, weights))
            curr = face_net.update(curr)
            print("\nStarting from:\n")
            print_array2grid(prev)
            print("\nTransition to:\n")
            print_array2grid(curr)


def plot_energies():
    d = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    e = [734, -946, -10854, -2340, 222, -24048, -7038, -5340, -382, -6810]
    plt.plot(d, e)
    plt.xlabel("Distortion")
    plt.ylabel("Energy")
    plt.show()

def main():

    # create_starter()
    # create_weights_matrix()
    # test_updates()
    # run_1a()
    plot_energies()


if __name__ == "__main__":
    main()
