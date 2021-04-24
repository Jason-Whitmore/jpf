


public class Sigmoid extends ActivationFunction{

    public Sigmoid(){

    }


    public float f(float x){
        float xExp = (float)(Math.exp((double)x));

        return xExp / (xExp + 1);
    }

    public float fPrime(float x){
        return f(x) * (1 - f(x));
    }
}