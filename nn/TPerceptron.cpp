#include "TPerceptron.h"
#include <numeric>

using namespace std;

double frand(){
    return (2.*(double)rand() / RAND_MAX) - 1.;
}

TPerceptron::TPerceptron(int inputs, double bias) : m_bias(bias) {
    m_weights.resize(inputs+1);
    generate(m_weights.begin(), m_weights.end(), frand);
}

double TPerceptron::run(std::vector<double> x) const {
    x.push_back(m_bias);
    double sum = inner_product(x.begin(), x.end(), m_weights.begin(), (double)0.0);
    return sigmoid(sum);
}

void TPerceptron::set_weights(std::vector<double>& w_init) {
    if(w_init.size() != m_weights.size())
        throw std::bad_exception();
    
    m_weights = w_init;
}

double TPerceptron::sigmoid(double& x) const {
    return 1./(1.+exp(-x));
}


TPerceptron::~TPerceptron(){
    // nothing
}
