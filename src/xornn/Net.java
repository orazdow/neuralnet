
package xornn;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;


public class Net {
    
    ArrayList<ArrayList<Neuron>> layers = new ArrayList<>();
    int[] topo = {2,2,1};
    enum NType{INPUT, HIDDEN, OUTPUT};
    double outError = 0;
    double learningRate = 0.1;
    
    Net(){
        init(topo);
    }
    
    Net(int[] topo){
        this.topo = topo;
        init(topo);
    }
    
    
    void setInputs(Double[] data) throws Exception{
        ArrayList<Neuron> layer = layers.get(0);
        if(data.length != layer.size()) throw new Exception("data vector does not match input vector");
            for(int i = 0; i < data.length; i++){
                layer.get(i).input = data[i];
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
    
    void backwardError(Neuron n, double target, double error){
        if(n.type == NType.INPUT){return;}
        double e = 0.0;
        double err = (n.type == NType.OUTPUT) ? error : target;

        for(int i = 0; i < n.inputs.size(); i++){
            double g = gradient(n.inputs.get(i).output, n.output, err, (n.type == NType.OUTPUT));
            double w = n.weights.get(i);
            w -= g;
            n.weights.set(i, w);
            e += g;
        }
            double b = gradient(1, n.output,err, (n.type == NType.OUTPUT));
            n.bias -= b;
            e+= b;
            
        for(int i = 0; i < n.inputs.size(); i++)
            backwardError(n.inputs.get(i), target, err);
    }
        
    double gradient(double logitInput, double neuronOutput, double target){
        return learningRate* logitInput * Neuron.dsigmoid(neuronOutput) * (target);
    }

    double gradient(double logitInput, double neuronOutput, double target, boolean outputLayer){
      if(outputLayer){
          return -target*learningRate*Neuron.dsigmoid(neuronOutput);
      }else{
          return -target*logitInput*learningRate*Neuron.dsigmoid(neuronOutput);
      }
    }
    
    boolean iterate(Data data, int index, double limit){
        outError = 0;
       
        Data.Frame frame = data.frames.get(index);
        double target = frame.targets.get(0);
        try {
            setInputs(frame.getFeatures());
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
        computeForward();
        double out = layers.get(layers.size()-1).get(0).output;
        outError +=(target-out);
        System.out.println(out+" : "+target);
           
        System.out.println("error: "+outError);
        
        boolean rtn = (Math.abs(outError) < limit);
        
        backwardError(layers.get(layers.size()-1).get(0), target, outError);
        
        return rtn;
    }
    
    boolean forwardBatch(Data data, int index, int num, double limit){
        outError = 0;
        double target = 0;
        // compute error over batch
        for(int i = index; i < index+num; i++) {
            Data.Frame frame = data.frames.get(i);
            target = frame.targets.get(0);
            try {
                setInputs(frame.getFeatures());
            } catch (Exception ex) {
                System.out.println(ex.getMessage());
            }
            computeForward();
            double out = layers.get(layers.size()-1).get(0).output;
//            outError +=0.5*Math.pow((target-out),2);
            outError +=(target-out);
            System.out.println(out+" : "+target);
        }    
        System.out.println("error: "+outError);
        //start backprop on output
        backwardError(layers.get(layers.size()-1).get(0), target, outError);
        
        return (Math.abs(outError) < limit);
        

    }
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
