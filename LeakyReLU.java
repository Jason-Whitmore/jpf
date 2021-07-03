/**
 * Class definition for the LeakyReLU activation function which is derived from the abstract ActivationFunction class.
 */
public class LeakyReLU extends ActivationFunction{

    /**
     * Determines the slope of the function on inputs less than 0.
     * Setting alpha to 0 creates the normal ReLU function.
     */
    private float alpha;

    /**
     * Creates the LeakyReLU activation function with a user selected alpha.
     * @param alpha The slope of the function on inputs less than 0.
     * A selection of 0 turns this function into the normal ReLU function.
     */
    public LeakyReLU(float alpha){
        this.alpha = alpha;
    }

    /**
     * The main activation function. LeakyReLU outputs numbers in range (-inf, inf) for alpha < 0.
     * @param x The input number.
     * @return The output number.
     */
    public float f(float x){
        if(x > 0){
            return x;
        } else {
            return x * alpha;
        }
    }

    public float fPrime(float x){
        if(x > 0){
            return 1;
        } else {
            return alpha;
        }
    }


    public String toString(){
        return "LEAKYRELU(" + this.alpha + ")";
    }
}