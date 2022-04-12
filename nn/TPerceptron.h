#ifndef TPERCEPTRON_H
#define TPERCEPTRON_H

#include <iostream>
#include <vector>

class TPerceptron {
private:
    
    /// weights for this perceptron
    std::vector<double> m_weights;
    
    /// bias. it can shift the curve
    double m_bias = 1.0;
        
public:
    
    /// Constructor with number of weights (inputs) for this neuron
    TPerceptron(int inputs, double bias=1.0);
   
    /// Default constructor
    TPerceptron();
    
    /// Destructor
	~TPerceptron();
	
    /// Runs simulation (weights.x)
    double run(std::vector<double> x) const;
    
    /// Returns the weights for this neuron
    std::vector<double>& weights() {return m_weights;}
    
    /// Sets the weights of the perceptron
    void set_weights(std::vector<double>& w_init);
    
    /// Calculates the Sigmoid function at x
    double sigmoid(double& x) const;
};

#endif
