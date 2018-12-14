package xornn;

import processing.core.PApplet;


public class Main extends PApplet{

    public static void main(String[] args) {
      //  PApplet.main("xornn.Main"); 
        Net nn = new Net();
        Data d = new Data();
        Util.genXOR(d, 10000);
        double limit = 0.0013;
        int index = 0;
                
        for (int i = 0; i < d.frames.size(); i++) {
             double error = nn.iterate(d, i);
             System.out.println("error: "+error);
             if(error <= limit){
                 System.out.println("error limit reached at "+i+": "+error);
                 break;
             }
        }

        System.out.println("predictions:");
        d = new Data();
        Util.genXOR(d, 10);
        for(int i = 0; i < 10; i++){
           double x1 = d.frames.get(i).features.get(0);
           double x2 = d.frames.get(i).features.get(1);
           double y =  d.frames.get(i).targets.get(0);
           double p = nn.predict(d, i);
           System.out.println(x1+", "+x2+": "+y+" ("+p+")");
        }

    }
    
}
