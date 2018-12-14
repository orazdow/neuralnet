package xornn;

import java.util.ArrayList;

public class Neuron {
    Double input;
    Double output;
    Double bias = 0.1;
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
        neuron.weights.add(Math.random());
    }
    
    static double sigmoid(double in){
        return 1.0/(1+Math.exp(-1.0*in));
    }
    
    static double dsigmoid(double in){
        return (in*(1.0-in));
    }
    
    void compute(){
        if(type == Net.NType.INPUT){
            output = input;
            return;
        }
        double sum = 0;
        int len = inputs.size();
        
        for(int i = 0; i < len; i++){
            sum += inputs.get(i).output*weights.get(i);
        }
        sum += bias;
        if(type == Net.NType.HIDDEN)
            output = sigmoid(sum);
        else
            output = sigmoid(sum); //just using sigmoid for now
    }
    
}
