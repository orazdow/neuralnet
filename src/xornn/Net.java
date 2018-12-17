
package xornn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;


public class Net {
    
    ArrayList<ArrayList<Neuron>> layers = new ArrayList<>();
    int[] topo = {2,2,1};
    enum NType{INPUT, HIDDEN, OUTPUT, BIAS};
    double outError = 0;
    double learningRate = 0.3;
    static double initial_bias = 0.0;
    
    Net(){
        init(topo);
    }
    
    Net(int[] topo){
        this.topo = topo;
        init(topo);
    }
    
    
    void setInputs(Double[] data) throws Exception{
        ArrayList<Neuron> layer = layers.get(0);
        if(data.length != layer.size()-1) throw new Exception("data vector does not match input vector");
            for(int i = 0; i < data.length; i++){
                if(layer.get(i).type != NType.BIAS)
                layer.get(i).input = data[i];
                else{System.out.println("BIAS");}
            }
    }
    
    double predict(Data data, int index){
        Data.Frame frame = data.frames.get(index);
        double target = frame.targets.get(0);
        try {
            setInputs(frame.getFeatures());
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
        computeForward();
        return layers.get(layers.size()-1).get(0).output;
    }
 
    double predict(Double[] data, int targetIndex){
        Double[] features = Arrays.copyOfRange(data, 0, targetIndex);
        try {
            setInputs(features);
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
        computeForward();
        return layers.get(layers.size()-1).get(0).output;
    }
    
    double predict(Double[] features){
        try {
            setInputs(features);
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
        computeForward();
        return layers.get(layers.size()-1).get(0).output;
    }
    
    void reset(){
        for(ArrayList<Neuron> layer : layers){
            for(Neuron n : layer){
                if(n.type != NType.INPUT && n.type != NType.BIAS){
                    for(int i = 0; i < n.inputs.size(); i++) {
                        if(n.inputs.get(i).type != NType.BIAS){
                            n.weights.set(i, Math.random());
                        }else{
                            n.weights.set(i, Net.initial_bias);
                        }
                    }
                }
            }
        }
    }
    
    void preturb(double... params){
        Neuron n = layers.get(layers.size()-1).get(0);
        for(int i = 0; i < params.length; i++){
           if(n.inputs.get(i).type != NType.BIAS)
               n.weights.set(i, params[i]);
        }
    }
    
    double error(double y, double yhat){
        return (y-yhat)*(y-yhat);
    }

    double derror(double y, double yhat){
        return -2*(y-yhat);
    }
    
    void backProp(Neuron n, double indelta){
        if(n.type == NType.INPUT || n.type == NType.BIAS){return;}
        
        double nextdelta = 0.0;
        for(int i = 0; i < n.inputs.size(); i++){ 
            // sum new delta
            double w = n.weights.get(i);    
            nextdelta += indelta*w;
         
            // update: w_i = w_i-(lr * z_i * delta * dsig(a))
            double gradient = learningRate * indelta * n.inputs.get(i).output * Neuron.dsigmoid(n.logit); 
            w -= gradient;
            n.weights.set(i, w);            
        }
        //nextdelta *= Neuron.dsigmoid(n.logit);
        for(int i = 0; i < n.inputs.size(); i++){
            backProp(n.inputs.get(i), nextdelta);
        }
        
    }
    
    // stochastic gradient descent, fwd/bkwd for each sample
    double iterate(Data data, int index){
        
        Data.Frame frame = data.frames.get(index);
        try {
            setInputs(frame.getFeatures());
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
        double error = 0.0;
        computeForward();
        double target = frame.targets.get(0);
        
        ArrayList<Neuron> layer = layers.get(layers.size()-1);
 
        // backprop with error derivative on output layer
        for(Neuron n : layer){
            double out = n.output;
            error += error(target, out);
            backProp(n, derror(target, out));
        }
        
        return error;
    }
    
    //backward for n forward passes in batch
    double miniBatch(Data data, int index, int stride, double errlimit){
        return 0.0;
    }
    
    // forward pass
    void computeForward(){        
        for(ArrayList<Neuron> layer : layers){
            for(Neuron n : layer){
                n.compute();
            }
        }
    }    
    void init(int[] topo){
        for(int i = 0; i < topo.length; i++){
            NType type = NType.INPUT;
            if(i > 0){type = NType.HIDDEN;} 
            if(i == topo.length-1){type = NType.OUTPUT;}
            layers.add(initLayer(topo[i], type));
        }
        
        for(int i = 0; i < layers.size()-1; i++){
            connect(layers.get(i), layers.get(i+1));
        }
        
    } 
    
    ArrayList<Neuron> initLayer(int n, NType type){
        ArrayList<Neuron> layer = new ArrayList<>();
        for(int i = 0; i < n; i++)
            layer.add(new Neuron(type));

        if(type != NType.OUTPUT){layer.add(new Neuron(NType.BIAS));}
        
        return layer;
    }
     
    void connect( ArrayList<Neuron> layerA,  ArrayList<Neuron> layerB){
        for(Neuron b : layerB){
            for(Neuron a : layerA){
                a.connect(b);                     
            }
        }
    }
    
}
