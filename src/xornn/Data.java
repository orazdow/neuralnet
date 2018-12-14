package xornn;

import java.util.ArrayList;

public class Data {
    
    ArrayList<Frame> frames = new ArrayList<>();
    
    void addFrame(double[] X, double[] Y){
        frames.add(new Frame(X,Y));
    }
    void addFrame(double[] X, double y){
        frames.add(new Frame(X,y));
    }
 
    class Frame{
            
        ArrayList<Double> features = new ArrayList<>();
        ArrayList<Double> targets = new ArrayList<>();
        
        Double[] getFeatures(){
            return features.toArray(new Double[features.size()]);
        }
        
        Frame(){}
        Frame(double[] X, double[] Y){
            for(Double x : X){
                features.add(x);
            }
            
            for(Double y : Y){
                targets.add(y);
            }
        }
        Frame(double[] X, double y){
            for(Double x : X){
                features.add(x);
            } 
            targets.add(y);
        }
        
    }
    
}
