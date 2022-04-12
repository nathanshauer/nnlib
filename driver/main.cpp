#include <iostream>
#include "TMultiLayerPerceptron.h"

void logicGateExampleAND();
void logicGateExampleOR();
void logicGateExampleXOR();
void trainedXORExample();

using namespace std;
int main (int argc, char * const argv[]) {
    
    const int caseToSim = 3;
    if (caseToSim == 0)
        logicGateExampleAND();
    else if(caseToSim == 1)
        logicGateExampleOR();
    else if(caseToSim == 2)
        logicGateExampleXOR();
    else if(caseToSim == 3)
        trainedXORExample();
    else
        throw std::bad_exception();

    return 0;
}

void logicGateExampleAND() {
    cout << "--------- Logic Gate AND Example ---------" << endl;
    // In this example we use two weights and a bias to only generate a positive sum if
    // both inputs are 1
    
    const int ninput = 2;
    TPerceptron p(ninput);
    vector<double> w_init = {10,10,-15};
    p.set_weights(w_init);
    
    cout << "Gate: \n" << endl;
    cout << "inputs: 0,0\t\tgate val = " << p.run({0,0}) << endl;
    cout << "inputs: 0,1\t\tgate val = " << p.run({0,1}) << endl;
    cout << "inputs: 1,0\t\tgate val = " << p.run({1,0}) << endl;
    cout << "inputs: 1,1\t\tgate val = " << p.run({1,1}) << endl;
}

void logicGateExampleOR() {
    cout << "--------- Logic Gate OR Example ---------" << endl;
    
    const int ninput = 2;
    TPerceptron p(ninput);
    vector<double> w_init = {15,15,-10};
    p.set_weights(w_init);
    
    cout << "Gate: \n" << endl;
    cout << "inputs: 0,0\t\tgate val = " << p.run({0,0}) << endl;
    cout << "inputs: 0,1\t\tgate val = " << p.run({0,1}) << endl;
    cout << "inputs: 1,0\t\tgate val = " << p.run({1,0}) << endl;
    cout << "inputs: 1,1\t\tgate val = " << p.run({1,1}) << endl;
}

void logicGateExampleXOR() {
    cout << "--------- Logic Gate XOR Example ---------" << endl;
    // this neural network should give 1 for inputs 1,0 and 0,1. And give 0 for inputs 0,0 and 1,1
    // This can only be achieved by a multilayer neural network.
    // This NN has the first layer with a
        
    TMultiLayerPerceptron mlp({2,2,1}); // layer 0 has 2 inputs, layer 1 has 2 perceptrons, layer 2 has 1 perceptron
    
                       //layer 1 (2 percep)         layer 2 (1 percep)
    mlp.set_weights( { {{-10,-10,15},{15,15,-10}},  {{10,10,-15}}  } );
    
    cout << "Gate: \n" << endl;
    cout << "inputs: 0,0\t\tgate val = " << mlp.run({0,0})[0] << endl;
    cout << "inputs: 0,1\t\tgate val = " << mlp.run({0,1})[0] << endl;
    cout << "inputs: 1,0\t\tgate val = " << mlp.run({1,0})[0] << endl;
    cout << "inputs: 1,1\t\tgate val = " << mlp.run({1,1})[0] << endl;
}

void trainedXORExample() {
    cout << "--------- Trained XOR Example ---------" << endl;
    TMultiLayerPerceptron mlp({2,2,1}); // layer 0 has 2 inputs, layer 1 has 2 perceptrons, layer 2 has 1 perceptron
    double mse;
    for (int i = 0; i < 3000; i++) {
        mse = 0.;
        mse += mlp.bp({0,0},{0}); // inputs 0,0 should output 0
        mse += mlp.bp({0,1},{1}); // inputs 0,1 should output 1
        mse += mlp.bp({1,0},{1}); // inputs 1,0 should output 1
        mse += mlp.bp({1,1},{0}); // inputs 1,1 should output 0
        mse /= 4.;
        if (i % 100 == 0)
            cout << "MSE = " << mse << endl;
    }
    
    cout << "\n\nTrained weights:" << endl;
    mlp.print_weights();
    
    cout << "Gate: \n" << endl;
    cout << "inputs: 0,0\t\tgate val = " << mlp.run({0,0})[0] << endl;
    cout << "inputs: 0,1\t\tgate val = " << mlp.run({0,1})[0] << endl;
    cout << "inputs: 1,0\t\tgate val = " << mlp.run({1,0})[0] << endl;
    cout << "inputs: 1,1\t\tgate val = " << mlp.run({1,1})[0] << endl;
}
