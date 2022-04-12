#include "TMultiLayerPerceptron.h"
#include <numeric>

using namespace std;

TMultiLayerPerceptron::TMultiLayerPerceptron(std::vector<int> layers, double bias, double eta) : m_bias(bias), m_eta(eta) {
    for (int i = 0; i < layers.size(); i++) {
        m_values.push_back(vector<double>(layers[i],0.));
        m_network.push_back(vector<TPerceptron>());
        m_d.push_back(vector<double>(layers[i],0.));
        if (i > 0){  // network[0] is the input layer, so it has no neurons
            for (int j = 0; j < layers[i]; j++)
                m_network[i].push_back(TPerceptron(layers[i-1],bias)); // uses default copy constructor
        }
    }
}

std::vector<double> TMultiLayerPerceptron::run(std::vector<double> x) {
    m_values[0] = x;
    for (int i = 1; i < m_network.size(); i++) { // loop through layers of network (each layer may have many perceptrons). Skips layer 0 (input)
        for (int j = 0; j < m_network[i].size(); j++) { // loop through perceptrons of a layer of the network
            m_values[i][j] = m_network[i][j].run(m_values[i-1]); // run perceptron j of layer i on input i-1. Output goes to values at layer i at position j
        }
    }
    return m_values.back();
}

void TMultiLayerPerceptron::set_weights(std::vector<std::vector<std::vector<double> > > w_init) {
    for (int i = 0; i < w_init.size(); i++) { // looping through layers of network
        for (int j = 0; j < w_init[i].size(); j++) { // looping through each perceptron
            // we start from i+1 because the input layer has no weights
            m_network[i+1][j].set_weights(w_init[i][j]); // each layer has one perceptron with a set of weights
        }
    }
}

void TMultiLayerPerceptron::print_weights() {
    cout << endl;
    for (int i = 0; i < m_network.size(); i++) {
        for (int j = 0; j < m_network[i].size(); j++) {
            cout << "Layer " << i+1 << " Neuron " << j << ": ";
            for (auto &it : m_network[i][j].weights()) {
                cout << it << "   ";
            }
            cout << endl;
        }
    }
    cout << endl;
}

double TMultiLayerPerceptron::bp(std::vector<double> x, std::vector<double> y) {
    
    // ======================> Feed sample to network
    std::vector o = run(x);
    
    // ======================> Calculate MSE
    if (o.size() != y.size())
        throw std::bad_exception();
    
    double mse = 0.;
    for (int i = 0; i < y.size(); i++) {
        mse += (y[i] - o[i]) * (y[i] - o[i]);
    }
    mse /= y.size();
        
    // ======================> Calculate the output errors
    const int nlayers = m_network.size();
    for (int i = 0; i < m_d.back().size(); i++) {
        m_d.back()[i] = delta(y[i],o[i]);
    }
    
    
    // ======================> Calculate the error term of each unit on each layer
    for (int i = nlayers-2; i > 0; i--) { // goes backwards through the layers skipping the output layer
        const int layeri_size = m_network[i].size();
        for (int h = 0; h < layeri_size; h++) { // loops over neurons of layer i
            double fwd_error = 0.;
            const int layerip1_size = m_network[i+1].size();
            for (int k = 0; k < layerip1_size; k++){  // loop over the layer ahead to get errors
                // neuron of layer i+1 has nweights as the size of layer i
                fwd_error += m_network[i+1][k].weights()[h] * m_d[i+1][k]; // updates sum(wk*deltak)
            }
            m_d[i][h] = dsig(m_values[i][h]) * fwd_error;
        }
    }
    
    // ======================> Calculate the deltas and update the weights
    for (int i = 1; i < nlayers; i++) { // loop over layers of network (skip layer 0 which is input)
        for (int j = 0; j < m_network[i].size(); j++) { // loop over neurons of layer
            const int nweights = m_network[i][j].weights().size();
            for (int k = 0; k < nweights; k++) { // number of weights in neuron j.
                double delta;
                if (k == nweights-1) // we are updating the delta of the bias
                    delta = m_eta * m_d[i][j] * m_bias;
                else
                    delta = m_eta * m_d[i][j] * m_values[i-1][k];
                m_network[i][j].weights()[k] += delta;                
            }
        }
    }
    
    return mse;
}

TMultiLayerPerceptron::~TMultiLayerPerceptron(){
    // nothing
}

const double TMultiLayerPerceptron::delta(const double y, const double o) const {
    return dsig(o) * (y - o);
}

const double TMultiLayerPerceptron::dsig(const double o) const {
    return o * (1. - o);
}
