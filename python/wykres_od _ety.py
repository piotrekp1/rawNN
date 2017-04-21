import MLP
import pandas as pds
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    timeout = 600000

    # eta = [1.1, 1.2, 1.3, 1.35, 1.4, 1.5]
    #eta = [1.2, 1.3, 1.35]
    eta = [0.29, 0.295, 0.3, 0.31, 0.32] #dla ety stalej
    ilosc_prob = list(map(lambda eta: MLP.ilosc_prob(eta, timeout), eta))

    d = {'eta': eta, 'epochs needed': ilosc_prob}
    df = pds.DataFrame(data=d)

    sns.pointplot(x="eta", y="epochs needed", data=df)
    plt.show()


def probuj_jedno():
    timeout = 500000
    eta = 1.2
    MLP.ilosc_prob(eta, timeout)


main()
