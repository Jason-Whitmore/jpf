


public class Tanh extends ActivationFunction{

    public float f(float x){
        return (float)Math.tanh(x);
    }

    public float fPrime(float x){
        float t = (float)Math.tanh(x);

        return 1 - (t * t);
    }


    public String toString(){
        return "TANH";
    }
}