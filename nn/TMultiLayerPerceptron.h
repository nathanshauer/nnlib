#ifndef TMULTILAYERPERCEPTRON_H
#define TMULTILAYERPERCEPTRON_H

#include "TPerceptron.h"

/**
 * @class TMultiLayerPerceptron manages a multi layered neural network
 */
class TMultiLayerPerceptron {
private:
    /// Bias can be used to shift the output of a neuron
    double m_bias = 1.0;
    
    /// Learning rate. Can change how fast the weights are changed in the backpropagation (bp) algorithm
    double m_eta;
    
    /// For each layer, the vector of perceptrons. Position 0 has nothing since it is the input layer
    std::vector<std::vector<TPerceptron> > m_network;
    
    /// For each layer, the values before the layer. The last position tells the values at the end. Position 0 is the input layer
    std::vector<std::vector<double> > m_values;
    
    /// Deltas for each neuron
    std::vector<std::vector<double> > m_d;
    
public:
    
    /// Constructor that receives number of layers and amount of neurons per layer.
    TMultiLayerPerceptron(std::vector<int> layers, double bias=1.0, double eta = 0.5);
    
    /// Desctructor (empty)
	~TMultiLayerPerceptron();
	    
    /// Sets weights based on an input w_init. w_init should have the correct size as initialized in the constructor
    void set_weights(std::vector<std::vector<std::vector<double> > > w_init);
    
    /// Prints the weights to screen
    void print_weights();
    
    /// Runs the neural net for an input layer x
    std::vector<double> run(std::vector<double> x);
    
    /// Runs the back propagation algorithm for an input x with expected results y. This function will modify the weights and spit a measure of the error (MSE).
    double bp(std::vector<double> x, std::vector<double> y);
    
    const double delta(const double y, const double o) const;
    
    const double dsig(const double o) const;
};

#endif
