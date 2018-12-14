package xornn;

import processing.core.PApplet;


public class Main extends PApplet{


    public static void main(String[] args) {
      //  PApplet.main("xornn.Main"); 
        Net nn = new Net();
        Data d = new Data();
        Util.genXOR(d, 10000);
        double limit = 0.02;
        int index = 0;
        
        System.out.println("");
        
        for (int i = 0; i < 1000; i++) {
            boolean reached = nn.forwardBatch(d, index, 10, limit);
            index += 10;
            if(reached){
                System.out.println("reached "+limit+" at iteration "+i);
                break;
            }
//           double x1 = d.frames.get(i).features.get(0);
//           double x2 = d.frames.get(i).features.get(1);
//           double y =  d.frames.get(i).targets.get(0);
//           double p = nn.predict(d, i);
//            System.out.println(x1+", "+x2+": "+y+" ("+p+")");
        }
        


    }
    
}
