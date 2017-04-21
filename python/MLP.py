import random

import numpy as np
import sys
from scipy.special import expit

# BETA = 2.6
BETA = 3.4


class Perceptron:
    def __init__(self, inputs_len, act_fun, beta):
        self.weights = [random.random() for i in range(0, inputs_len + 1)]
        self.act_fun = act_fun
        self.beta = beta

    def activate(self, inp):
        return self.act_fun(np.dot([1] + inp, self.weights) * self.beta)

    def update_weight(self, which_weight, added_weight):
        self.weights[which_weight] += added_weight

    def get_weights(self):
        return self.weights


class OneLayerNetwork:
    def __init__(self, inputs_len, layer_width, eta, beta, decrease=0):
        self.hidden_layer = [Perceptron(inputs_len, expit, beta) for i in range(0, layer_width)]
        self.output_neuron = Perceptron(layer_width, expit, beta)
        self.eta = eta
        self.beta = beta
        self.decrease = decrease

    def classify(self, input_vec):
        first_layer_results = list(map(lambda neuron: neuron.activate(input_vec), self.hidden_layer))
        return self.output_neuron.activate(first_layer_results)

    def delta_0(self, exp_result, output):
        return self.beta * (exp_result - output) * output * (1 - output)

    def delta_hi(self, exp_result, output, v_i, h_i):
        del_0 = self.delta_0(exp_result, output)
        return self.beta * del_0 * v_i * h_i * (1 - h_i)

    def dist(self, result, exp_resut):
        return (result - exp_resut) ** 2

    def dec_eta(self):
        self.eta -= self.decrease

    def learn(self, input_vec, exp_result, printing=False):
        output = self.classify(input_vec)
        if printing:
            print('input_vec:{}'.format(input_vec))
            print(self.dist(exp_result, output))

        H = list(map(lambda neur: neur.activate(input_vec), self.hidden_layer))  # vector of h_i s
        V = self.output_neuron.get_weights()

        # output layer
        self.output_neuron.update_weight(0, self.eta * self.delta_0(exp_result, output))
        for i in range(1, len(self.hidden_layer) + 1):
            update_val = self.eta * self.delta_0(exp_result, output) * H[i - 1]
            self.output_neuron.update_weight(i, update_val)

        # hidden layer
        for i in range(0, len(self.hidden_layer)):
            neuron = self.hidden_layer[i]
            eta_delta_hi = self.delta_hi(exp_result, output, V[i + 1], H[i]) * self.eta
            neuron.update_weight(1, eta_delta_hi * input_vec[0])
            neuron.update_weight(0, eta_delta_hi)

        output = self.classify(input_vec)
        if printing:
            print(self.dist(exp_result, output))

        self.dec_eta()


def f(x):
    retvals = [1, 0, 0, 1]
    return retvals[x]


def trained_network(eta, beta, training_sample_len):
    net = OneLayerNetwork(1, 8, eta, beta)
    upgrade_net(training_sample_len, net)
    return net


def print_classifyings(net):
    print("Wyniki:")
    for i in range(0, 4):
        print('x = {} , predicted f(x) = {}, expected f(x) = {}'.format(i, net.classify([i]), f(i)))
    print("\n\n")


def print_xor_classifyings(net):
    print("Wyniki:")
    for x in range(0, 2):
        for y in range(0, 2):
            print('x = {} ,y = {} predicted f(x) = {}, expected f(x) = {}'.format(x, y, net.classify([x, y]),
                                                                                  xor_fun(x, y)))
    print("\n\n")


def upgrade_net(skok, net):
    for i in range(0, skok):
        for inp in range(0, 4):
            # inp = random.randint(0, 3)
            net.learn([inp], f(inp))
        if i == 1000:
            sys.exit()

def upgrade_xor_net(skok, net):
    for i in range(0, skok):
        inp1 = random.randint(0, 1)
        inp2 = random.randint(0, 1)
        net.learn([inp1, inp2], xor_fun(inp1, inp2))


def upgrade_net_2(skok, net, printing=False):
    for i in range(0, skok):
        for i in range(0, 4):
            net.learn([i], f(i), printing)


def upgrade_net_3(skok, net, printing=False):
    for i in range(0, skok):
        for i in range(0, 4):
            net.learn([i], f(i), printing)
            res = net.classify([i])
            dist = net.dist(res, f(i))

            while dist > 0.20:
                net.learn([i], f(i), printing)
                res = net.classify([i])
                dist = net.dist(res, f(i))

                print_classifyings(net)
                if dist < 0.20:
                    print_classifyings(net)


def supremum_val(net):
    maks = 0
    for i in range(0, 4):
        delta = abs(net.classify([i]) - f(i))
        if delta > maks:
            maks = delta
    return maks


def progress(skok):
    net = trained_network(1.3, BETA, skok)
    while supremum_val(net) > 0.001:
        print_classifyings(net)
        upgrade_net(skok, net)

    print(supremum_val(net))
    print_classifyings(net)


def xor_fun(x1, x2):
    return int((x1 or x2) and (not (x1 and x2)))


def xor(eta):
    net = OneLayerNetwork(2, 4, eta, 0.10)
    upgrade_xor_net(10000, net)
    print_xor_classifyings(net)


def main():
    progress(10000)
    # xor(0.9)


def ilosc_prob(eta, timeout):
    skok = 10000
    i = skok
    net = trained_network(eta, BETA, skok)
    while supremum_val(net) > 0.001 and i < timeout:
        print_classifyings(net)
        upgrade_net(skok, net)
        i += skok

    print(supremum_val(net))
    print_classifyings(net)
    if i > timeout:
        return -1
    return i
