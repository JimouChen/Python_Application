import matplotlib.pyplot as plt

if __name__ == '__main__':

    RLIST = [1 / 3]
    DLIST = [1 / 3]
    ILIST = [1 / 3]
    for i in range(40):
        R = RLIST[i] * 0.75 + DLIST[i] * 0.20 + ILIST[i] * 0.40
        RLIST.append(R)
        D = RLIST[i] * 0.05 + DLIST[i] * 0.60 + ILIST[i] * 0.20
        DLIST.append(D)
        I = RLIST[i] * 0.20 + DLIST[i] * 0.20 + ILIST[i] * 0.40
        ILIST.append(I)
        plt.plot(RLIST)
        plt.plot(DLIST)
        plt.plot(ILIST)
        plt.xlabel('Time')
        plt.ylabel('Voting percent')
        plt.annotate('DemocraticParty', xy=(5, 0.2))
        plt.annotate('RepublicanParty', xy=(5, 0.5))
        plt.annotate('IndependentCandidate', xy=(5, 0.25))
        plt.show()
        print(RLIST, DLIST, ILIST)

    print('预测的最后一年：RLIST: {}, DLIST: {}, ILIST: {}'.format(RLIST[-1], DLIST[-1], ILIST[-1]))
