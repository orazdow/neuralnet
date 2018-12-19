package xornn;

import processing.core.PApplet;


public class Main extends PApplet{

    Net nn = new Net();
    double errlimit = 0.0013;
    Data d;
    double[][] X = new double[][]{{0,0}, {0,1}, {1,0}, {1,1}};
    double[] Y = new double[]{0, 1, 1, 0};
    double error = 5.0;
    int n = 0;
    
    int size = 600;
    int buttonw = 70;
    int buttonh = 20;
    int buttonm = 10;
    boolean viewErr = false;
    
    void init(){
        d = new Data( X, Y);
        //Util.genXOR(d, 10000);
    }
    
    
    public static void main(String[] args) {
        PApplet.main("xornn.Main"); 
    }
    
    // n iterations
    void train(Net nn, Data d, int num){
        for (int i = 0; i < num; i++) {
             double error = nn.iterate(d, i%d.frames.size());
             if(error <= errlimit){
                 System.out.println("error limit reached at "+i+": "+error);
                 break;
             }
        }   
    }  
    
    // 1 iteration
    double trainFrame(Net nn, Data d, int num){
        for(int i = 0; i < d.frames.size()*num; i++)
          error = nn.iterate(d, i%d.frames.size());
         
        return error;
    }
    
    // test random data
    void test(Net nn, int num){
        System.out.println("predictions:");
        Data d = new Data();
        Util.genXOR(d, num);
        for(int i = 0; i < num; i++){
           double x1 = d.frames.get(i).features.get(0);
           double x2 = d.frames.get(i).features.get(1);
           double y =  d.frames.get(i).targets.get(0);
           float p = (float)nn.predict(d, i);
           System.out.println(x1+", "+x2+": "+y+" ("+p+")");
        }    
    }
    
    // map predictions
    void predictMap(int samples, Net nn){
        int wh = size/samples;
        for(int i = 0; i < samples; i++){
            for(int j = 0; j < samples; j++){
                double y = i/(double)samples;
                double x = j/(double)samples;
                Double[] f = {y, x};
                double p = nn.predict(f);
                fill((float)p*295);
                rect(j*wh, i*wh, wh, wh);
            }
        }
    }
   
    // map error over output weights
    void _errorMap(int samples, Net nn){
        Double[][] X = {{0.0, 0.0},{0.0, 1.0}, {1.0, 0.0},{1.0, 1.0}};
        Double Y[] = {0.0, 1.0, 1.0, 0.0};
        int wh = size/samples;
        double e = 0.0;
        for(int i = -samples; i < samples; i++){
            for(int j = -samples; j < samples; j++){
                double y = i/(double)samples;
                double x = j/(double)samples;
                nn.preturb(x*10, y*10);
                //double p = nn.predict(X[1]);
               // e = (1-p)*(1-p);
               //try mapping gradient: derror(target, out)*output
                for(int _i = 0; _i < 4; _i++){
                    double p = nn.predict(X[_i]);
                    e += (Y[_i]-p)*(Y[_i]-p);
                }
                e *= 0.25;
                fill(map((float)e, 0, 1, 0, 255));
                rect(j*wh, i*wh, wh, wh);
            }
        }

    }
    
    // map error
    void errorMap(int samples, Net nn){
        Neuron n = nn.layers.get(nn.layers.size()-1).get(0);
        double w1 = n.weights.get(0);
        double w2 = n.weights.get(1);
        _errorMap(samples, nn); 
        n.weights.set(0, w1);
        n.weights.set(1, w2);
    }
    
    // display output weight cooridantes
    void outCoord(Net nn){
        Neuron n = nn.layers.get(nn.layers.size()-1).get(0);
        float x = n.weights.get(0).floatValue();
        float y = n.weights.get(1).floatValue();
        float b = n.weights.get(2).floatValue();
        x = map(x, -10, 10, 0, size);
        y = map(y, -10, 10, 0, size);
        fill(255,0,0);
        ellipse(x,y,20,20);
    }

    void buttons(){
        if(!viewErr){fill(100,200,200);}else{fill(100,255,100);}
        rect(size-(buttonw+buttonm), buttonm, buttonw, buttonh, 20);
        if(viewErr){fill(100,200,200);}else{fill(100,255,100);}
        rect(size-(buttonw+buttonm)*2, buttonm, buttonw, buttonh, 20);
        fill(0);
        text("error", (size-(buttonw+buttonm))+5, 5+buttonm+buttonh/2);
        text("predict", (size-(buttonw+buttonm)*2)+5, 5+buttonm+buttonh/2);
    }
    
    int bcoords(float x, float y){
        if(y > buttonm && y < buttonm+buttonh)
        if(x > size-(buttonw+buttonm) && x <  size-(buttonm)){
            return 1;
        }else if(x > size-(buttonw+buttonm)*2 && x < size-(buttonm*2+buttonw)){
            return 2;
        }  
        return 0;    
    }
    

    
    public void settings(){
      size(size,size);
      init();
    }
       
    public void setup(){
        noStroke();
        textSize(16);
        background(120);  
        noLoop();
        train(nn, d, d.frames.size()*1000);
        test(nn, 10);  
    }
    
    public void mousePressed(){
        int b = bcoords(mouseX, mouseY);
        if(b == 1){viewErr = true;}else if(b == 2){viewErr = false;}
        if(b != 0){redraw(); return;}
        noLoop();
        nn.reset();
        n = 0;
        error = 5.0;
        loop();
    }
    
    public void draw(){
       // 1 training iteration 
       error = trainFrame(nn, d, 10);
       n+=10;
       
       //display
       if(viewErr){
            errorMap(100, nn);
            outCoord(nn);
       }else{
            predictMap(100, nn); 
       }
       // stop training
       if(n > 1500 || error <= 0.004){
           Neuron out = nn.layers.get(nn.layers.size()-1).get(0);
           double w1 = out.weights.get(0); double w2 = out.weights.get(1); double w3 = out.weights.get(2);
           System.out.println("\niter: "+n+", w1: "+w1+", w2: "+w2+", bias: "+w3+", error: "+error);
           noLoop();
           n = 0;
           test(nn, 10);
       }
       
        fill(50, 200, 255);
        text("click to retrain network", 15, 20); 
        text("error: "+(float)error, 15, 40);       
        buttons();

    }   
}
