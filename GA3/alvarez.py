import numpy as np


def view(m):
    s = ''
    threshold = 0.95
    # for i in range(len(m)):
    #     if m[i] < threshold:
    #         s += 'o'
    #     elif m[i] >= threshold:
    #         s += 'x'
    #     else:
    #         s += '!'
    for i in range(len(m)):
        if m[i] == 0:
            s += 'o'
        elif m[i] == 1:
            s += 'x'
        else:
            s += '!'
    return s


def normalize(m):
    p = []
    for i in range(len(m)):
        p.append(max(m[i]))
    p = max(p)
    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] /= p
    return m


def initialize_network():

    # Create all-to-all symmetric synaptic weights
    nc1_nc2 = np.random.random([8, 8])
    mtl_nc1 = np.random.random([4, 8])
    mtl_nc2 = np.random.random([4, 8])

    return nc1_nc2, mtl_nc1, mtl_nc2


def train_model(mem1, mem2, mtl_init, mtl_nc1, mtl_nc2, nc1_nc2,
                alpha1=0.15, alpha2=0.015, decay=0.9, forget=0.01,
                presentations=2, tsteps=3):
    for e in range(presentations):
        nc1 = mem1.copy()
        nc2 = mem2.copy()
        mtl = mtl_init
        for t in range(tsteps):
            # Calculate inputs to mtl
            from_nc1 = np.matmul(mtl_nc1, nc1)
            from_nc2 = np.matmul(mtl_nc1, nc2)
            in_mtl = np.add(from_nc1, from_nc2)

            # Calculate mtl response
            for n in range(len(mtl)):
                # mtl[n] = decay*mtl[n] + in_mtl[n]
                mtl[n] = in_mtl[n]
            for n in range(len(mtl)):
                if mtl[n] < max(mtl):
                    mtl[n] = 0.0
                # elif mtl[n] > 1.0:
                #     mtl[n] = 1.0
                else:
                    mtl[n] = 1.0

            prev_1 = mtl_nc1.copy()

            # Update cortico-mtl weights
            for i in range(4):
                for j in range(8):
                    # mtl_nc1[i][j] += alpha1 * mtl[i] * (nc1[j]-np.average(mtl)) - forget*mtl_nc1[i][j]
                    # mtl_nc1[i][j] += alpha1 * mtl[i] * nc1[j]
                    if mtl[i] == 1 and nc1[j] == 1:
                        mtl_nc1[i][j] += alpha1
                    else:
                        mtl_nc1[i][j] -= alpha1
                    if mtl_nc1[i][j] > 1.0:
                        mtl_nc1[i][j] = 1.0
                    elif mtl_nc1[i][j] < 0.0:
                        mtl_nc1[i][j] = 0.0
                    # mtl_nc2[i][j] += alpha1 * mtl[i] * (nc2[j]-np.average(mtl)) - forget*mtl_nc2[i][j]
                    mtl_nc2[i][j] += alpha1 * mtl[i] * nc2[j]
                    if mtl_nc2[i][j] > 1.0:
                        mtl_nc2[i][j] = 1.0
                    elif mtl_nc2[i][j] < 0.0:
                        mtl_nc2[i][j] = 0.0
            # comp = np.add(mtl_nc1, -1*prev_1)
            # print("MTL to NC1 weight updates: ")
            # print(comp)

            # Calculate cortical inputs
            mtl_to_nc1 = np.matmul(mtl, mtl_nc1)
            nc2_to_nc1 = np.matmul(nc2, nc1_nc2.transpose())
            mtl_to_nc2 = np.matmul(mtl, mtl_nc2)
            nc1_to_nc2 = np.matmul(nc1, nc1_nc2)

            in_nc1 = np.add(mtl_to_nc1, nc2_to_nc1)
            in_nc2 = np.add(mtl_to_nc2, nc1_to_nc2)

            # Calculate cortical response
            for n in range(len(nc1)):
                nc1[n] = decay * nc1[n] + in_nc1[n]
                nc2[n] = decay * nc2[n] + in_nc2[n]

            for n in range(4):
                # NC1 Group 1
                if nc1[n] < max(nc1[0:4]):
                    nc1[n] = 0.0
                else:
                    nc1[n] = 1.0
                # NC2 Group 1
                if nc2[n] < max(nc2[0:4]):
                    nc2[n] = 0.0
                else:
                    nc2[n] = 1.0
                # NC1 Group 2
                if nc1[n+4] < max(nc1[4:8]):
                    nc1[n+4] = 0.0
                else:
                    nc1[n+4] = 1.0
                # NC2 Group 2
                if nc2[n+4] < max(nc2[4:8]):
                    nc2[n+4] = 0.0
                else:
                    nc2[n+4] = 1.0

            # Update cortico-cortical weights
            for i in range(8):
                for j in range(8):
                    # mtl_nc1[i][j] += alpha1 * mtl[i] * (nc1[j]-np.average(mtl)) - forget*mtl_nc1[i][j]
                    nc1_nc2[i][j] += alpha1 * nc1[i] * nc2[j]
                    if nc1_nc2[i][j] > 1.0:
                        nc1_nc2[i][j] = 1.0
                    elif nc1_nc2[i][j] < 0.0:
                        nc1_nc2[i][j] = 0.0

    return mtl_nc1, mtl_nc2, nc1_nc2


def update_activity(nc1, nc2, mtl_nc1, mtl_nc2, nc1_nc2,
                    decay=1.0):

    mtl = np.zeros(4)

    # Calculate inputs to mtl
    from_nc1 = np.matmul(mtl_nc1, nc1)
    from_nc2 = np.matmul(mtl_nc1, nc2)
    in_mtl = np.add(from_nc1, from_nc2)

    # Calculate mtl response
    for n in range(len(mtl)):
        # mtl[n] = decay*mtl[n] + in_mtl[n]
        mtl[n] = in_mtl[n]
    for n in range(len(mtl)):
        if mtl[n] < max(mtl):
            mtl[n] = 0.0
        # elif mtl[n] > 1.0:
        #     mtl[n] = 1.0
        else:
            mtl[n] = 1.0

    prev_1 = mtl_nc1.copy()

    # Calculate cortical inputs
    mtl_to_nc1 = np.matmul(mtl, mtl_nc1)
    nc2_to_nc1 = np.matmul(nc2, nc1_nc2.transpose())
    mtl_to_nc2 = np.matmul(mtl, mtl_nc2)
    nc1_to_nc2 = np.matmul(nc1, nc1_nc2)

    in_nc1 = np.add(mtl_to_nc1, nc2_to_nc1)
    in_nc2 = np.add(mtl_to_nc2, nc1_to_nc2)

    # Calculate cortical response
    for n in range(len(nc1)):
        nc1[n] = decay * nc1[n] + in_nc1[n]
        nc2[n] = decay * nc2[n] + in_nc2[n]

    for n in range(4):
        # NC1 Group 1
        if nc1[n] < max(nc1[0:4]):
            nc1[n] = 0.0
        else:
            nc1[n] = 1.0
        # NC2 Group 1
        if nc2[n] < max(nc2[0:4]):
            nc2[n] = 0.0
        else:
            nc2[n] = 1.0
        # NC1 Group 2
        if nc1[n + 4] < max(nc1[4:8]):
            nc1[n + 4] = 0.0
        else:
            nc1[n + 4] = 1.0
        # NC2 Group 2
        if nc2[n + 4] < max(nc2[4:8]):
            nc2[n + 4] = 0.0
        else:
            nc2[n + 4] = 1.0

    return mtl, nc1, nc2

def main():
    nc1_nc2, mtl_nc1, mtl_nc2 = initialize_network()

    # Create memories
    mem1 = [0, 1, 0, 0, 0, 0, 0, 1]
    mem2 = [0, 0, 1, 0, 1, 0, 0, 0]

    # Initialize random activity in MTL
    mtl_init = np.random.choice([0, 1], [4])
    # print("Initial MTL Activity: %s" % view(mtl_init))

    mtl_nc1, mtl_nc2, nc1_nc2 = train_model(mem1, mem2, mtl_init, mtl_nc1, mtl_nc2, nc1_nc2,
                                            presentations=100, tsteps=10)

    nc1 = [0, 1, 0, 0, 0, 0, 0, 0]
    nc2 = [0, 0, 0, 0, 1, 0, 0, 0]
    mtl, nc1, nc2 = update_activity(nc1, nc2, mtl_nc1, mtl_nc2, nc1_nc2)

    print("Memory 1: %s" % view(mem1))
    print("Reconstruction of memory 1: %s" % view(nc1))
    print("Memory 2: %s" % view(mem2))
    print("Reconstruction of memory 2: %s" % view(nc2))


if __name__ == "__main__":
    main()
