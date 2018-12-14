package xornn;

import processing.core.PApplet;


public class Main extends PApplet{

    Net nn = new Net();
    double errlimit = 0.0013;

    public static void main(String[] args) {
        PApplet.main("xornn.Main"); 
    }
    
    void train(Net nn){
        Data d = new Data();
        Util.genXOR(d, 10000);
                
        for (int i = 0; i < d.frames.size(); i++) {
             double error = nn.iterate(d, i);
            // System.out.println("error: "+error);
             if(error <= errlimit){
                 System.out.println("error limit reached at "+i+": "+error);
                 break;
             }
        }   
    }
    
    void test(Net nn, int num){
        System.out.println("predictions:");
        Data d = new Data();
        Util.genXOR(d, num);
        for(int i = 0; i < num; i++){
           double x1 = d.frames.get(i).features.get(0);
           double x2 = d.frames.get(i).features.get(1);
           double y =  d.frames.get(i).targets.get(0);
           double p = nn.predict(d, i);
           System.out.println(x1+", "+x2+": "+y+" ("+p+")");
        }    
    }
    
    void errorMap(int samples, Net nn, int size){
        int wh = size/samples;
        for(int i = 0; i < samples; i++){
            for(int j = 0; j < samples; j++){
                double y = i/(double)samples;
                double x = j/(double)samples;
                Double[] f = {y, x};
                double p = nn.predict(f, 2);
                fill((float)p*255);
                rect(j*wh, i*wh, wh, wh);
            }
        }
        fill(50, 200, 255);
        text("click to retrain network", 15, 20);
    }
    
    public void settings(){
      size(600,600);
    }
    
    public void mousePressed(){
        nn.reset();
        train(nn);
        test(nn, 10);
        redraw();
    }
       
    public void setup(){
        noFill();  
        noStroke();
        textSize(16);
        background(120);  
        noLoop();
        train(nn);
        test(nn, 10);
        
    }

    
    public void draw(){
        errorMap(100, nn, 600);
    }   
}
