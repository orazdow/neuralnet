
package xornn;


public class Util {
    
    static int[] XOR(){
        int a = (int)Math.round(Math.random());
        int b = (int)Math.round(Math.random());
        int c = a != b ? 1 : 0;    
        return new int[]{a,b,c};
    }
 
    static void genXOR(Data d, int num){
        for (int i = 0; i < num; i++) {
            int[] xor = XOR();
            d.addFrame(new double[]{xor[0], xor[1]}, xor[2]);
        }
    }
    
}
