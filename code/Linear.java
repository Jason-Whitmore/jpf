
/**
 * Class that defines the linear activation function, often used as an
 * output layer's activation function.
 */
public class Linear extends ActivationFunction{

    public float f(float x){
        Utility.checkReal(x);

        return x;
    }

    public float fPrime(float x){
        Utility.checkReal(x);
        
        return 1;
    }

    public String toString(){
        return "LINEAR";
    }
}