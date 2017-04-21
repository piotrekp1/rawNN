#include <iostream>
#include <vector>
#include <stdlib.h>
#include <random>
#include <math.h>
#include <algorithm>


std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> ret;
    for(int i = 0; i < a.size(); ++i){
        ret.push_back(a[i] + b[i]);
    }
    return ret;
}

/**
 * Assumes both vectors are of the same length
 */
double operator*(const std::vector<double>& fst,const std::vector<double>& snd){
    double res = 0;
    for(int i = 0; i < fst.size(); ++i){
        res += fst[i] * snd[i];
    }
    return res;
}

std::vector<double> multiply_coords(const std::vector<double>& a, const std::vector<double>& b){
    std::vector<double> ret;
    for(int i = 0; i < a.size(); ++i){
        ret.push_back(a[i] * b[i]);
    }
    return ret;
}

std::vector<double> multiply_by(const std::vector<double>& a, double multiplier){
    std::vector<double> res(a.size());
    std::transform(a.begin(), a.end(), res.begin(), [&](double el){ return el * multiplier;});
    return res;
}

class NN{
    std::vector<double> weights_1;
    std::vector<double> biases_1;
    std::vector<double> weights_2;
    double bias_2;

    int n;
    double static beta;
    double eta;

    double static activation_function(double arg){
        return 1/(1 + exp(-arg * beta));
    }

    std::vector<double> first_layer_result(int argument){
        std::vector<double> first_layer_arg = multiply_by(weights_1, argument);
        std::vector<double> first_layer_res((unsigned long) n);
        // launch activation function
        std::transform(first_layer_arg.begin(), first_layer_arg.end(), first_layer_res.begin(), activation_function);
        return first_layer_res;
    }

    double second_layer_result(const std::vector<double>& first_layer_result){
        double second_layer_arg = first_layer_result * weights_2 + bias_2;
        double second_layer_res = activation_function(second_layer_arg);

        return second_layer_res;
    }

public:
    NN(int n, double eta): n(n), eta(eta){
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double > dis(0,1);
        for(int i = 0; i < n; ++i){
            weights_1.push_back(dis(gen));
            biases_1.push_back(dis(gen));
            weights_2.push_back(dis(gen));
        }
        bias_2 = dis(gen);
    }

    double classify(int argument) {
        auto fst = first_layer_result(argument);
        return second_layer_result(fst);
    }

    double delta_0(double output, double exp_result){
        return beta * (exp_result - output) * output * (1 - output);
    }

    std::vector<double> delta_h(double exp_result, double output, const std::vector<double>& fst_layer_res){
        double del_0 = delta_0(exp_result, output);
        std::vector<double> res;
        for(int i = 0; i < weights_2.size(); ++i){
            res.push_back(beta * del_0 * weights_2[i] * fst_layer_res[i] * (1 - fst_layer_res[i]));
        }

        return res;
    }


    void learn(int input, double exp_result){
        auto fst = first_layer_result(input);
        auto output =  second_layer_result(fst);

        /**
         * v0 - bias in second layer
         * v - weights in second layer
         * u - biases in first layer
         * w - weights in first layer
         */

        double v0_plus = delta_0(output, exp_result) * eta;
        std::vector<double> v_plus = multiply_by(fst, delta_0(output,exp_result) * eta);
        std::vector<double> u_plus = multiply_by(delta_h(exp_result, output, fst), eta);
        std::vector<double> w_plus = multiply_by(u_plus, input);

        bias_2 = bias_2 + v0_plus;
        weights_2 = weights_2 + v_plus;
        biases_1 = biases_1 + u_plus;
        weights_1 = weights_1 + w_plus;
    }

};

double f(int x){
    return (double)(x % 3 == 0);
}

void print_results(NN& network){
    for(int i = 0; i < 4; ++i){
        std::cout << "i: " << i << " f(i): " << f(i) << " classified: " << network.classify(i) << std::endl;
    }
    std::cout << std::endl;
}

double sup_error(NN& network){
    double max_error = 0;
    for(int i = 0; i < 4; ++i){
        double error = abs(f(i) - network.classify(i));
        max_error = (max_error > error) ? max_error : error;
    }
    return max_error;
}

int train_network(int timeout_epochs, double eta){
    NN network(8, eta);
    for(int i = 0; i < timeout_epochs; ++i){
        for(int j = 0; j < 4; ++j){
            network.learn(j, f(j));
        }
        if(i % 30000 == 0){ print_results(network);}

        if(i % 2000 == 0 && sup_error(network) < 0.001){
            return i;
        }
    }
    return -1;

}

//double NN::beta = 2.65;
double NN::beta = 2.65;

int main() {
    std::cout << "wynik: " << train_network(4000000, 1.11) << std::endl;
    return 0;
}