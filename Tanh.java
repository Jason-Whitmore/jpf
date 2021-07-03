/**
 * Defines the Tanh (hyperbolic tangent) activation function which is derived
 * from the abstract ActivationFunction class.
 */
public class Tanh extends ActivationFunction{

    /**
     * The main activation function. Tanh(x) outputs numbers in range (-1, 1).
     * @param x The input number.
     * @return The output number.
     */
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