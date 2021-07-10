/**
 * Defines the Sigmoid activation function which is derived from the abstract ActivationFunction class.
 * The Sigmoid function is used when the output of a layer should be in the range (0,1)
 */
public class Sigmoid extends ActivationFunction{

    /**
     * The basic constructor for the sigmoid activation function.
     * There are no user defined parameters for this function.
     */
    public Sigmoid(){

    }

    /**
     * The main activation function. sigmoid(x) outputs numbers in range (0, 1).
     * @param x The input number.
     * @return The output number.
     */
    public float f(float x){
        //Check parameter
        Utility.checkReal(x);

        float xExp = (float)(Math.exp((double)x));

        return xExp / (xExp + 1);
    }

    public float fPrime(float x){
        Utility.checkReal(x);
        
        return f(x) * (1 - f(x));
    }
}