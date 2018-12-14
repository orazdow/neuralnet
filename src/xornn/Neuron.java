package xornn;

import java.util.ArrayList;

public class Neuron {
    Double input;
    Double output;
    Double bias = 0.1;
    Double logit = 0.0;
    Net.NType type;  
    ArrayList<Neuron> inputs;
    ArrayList<Double> weights;
    
    Neuron(Net.NType type){
        this.type = type;
        inputs = new ArrayList<>();
        weights = new ArrayList<>();
    }
    
    void connect(Neuron neuron){
        neuron.inputs.add(this);
        if(type != Net.NType.BIAS)
            neuron.weights.add(Math.random());
        else
            neuron.weights.add(Net.initial_bias);
    }
    
    static double sigmoid(double in){
        return 1.0/(1+Math.exp(-1.0*in));
    }
    
    static double dsigmoid(double in){
        return (sigmoid(in)*(1.0-sigmoid(in)));
    }
    
    void compute(){
        // input node
        if(type == Net.NType.INPUT){
            output = input;
            return;
        }
        // bias node
        if(type == Net.NType.BIAS){
            output = 1.0;
            return;
        }
        // sum input logit
        logit = 0.0;
        int len = inputs.size();
        for(int i = 0; i < len; i++){
            logit += inputs.get(i).output*weights.get(i);
        }
        //hidden node
        if(type == Net.NType.HIDDEN)
            output = sigmoid(logit);
        else //output node
            output = sigmoid(logit); //just using sigmoid for now
    }
    
}
