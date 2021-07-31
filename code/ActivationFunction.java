

/**
 * Defines the ActivationFunction abstract class for use in neural networks
 */
public abstract class ActivationFunction{

    /**
     * The activation function itself. Should be differentiable.
     * @param x The input scalar
     * @return The output scalar
     */
    public abstract float f(float x);

    /**
     * The activation function derivative. Used to find gradients.
     * @param x The input scalar.
     * @return The output scalar.
     */
    public abstract float fPrime(float x);


    /**
     * Applies the activation function on each component of the x vector and places the result in dest vector
     * @param x The vector to apply the activation function.
     * @param dest The vector to place the results.
     */
    public void f(float[] x, float[] dest){
        //Check parameters
        Utility.checkNotNull((Object)x, (Object)dest);
        Utility.checkArrayLengthsEqual(x, dest);

        for(int i = 0; i < x.length; i++){
            dest[i] = this.f(x[i]);
        }
    }

    /**
     * Applies the activation function on each component of the x vector.
     * @param x The vector to apply the activation function to.
     * @return A newly allocated vector of the x vector after applying the activation function.
     */
    public float[] f(float[] x){
        //Check parameter
        Utility.checkNotNull((Object)x);

        float[] r = new float[x.length];

        this.f(x, r);

        return r;
    }


    /**
     * Calculates the derivative of the activation function on each component of the x vector.
     * @param x The input vector where each component is where the derivative is calculated at.
     * @param dest The vector where the derivatives will be placed.
     */
    public void fPrime(float[] x, float[] dest){
        //Check parameters
        Utility.checkNotNull((Object)x, (Object)dest);
        Utility.checkArrayLengthsEqual(x, dest);

        for(int i = 0; i < x.length; i++){
            dest[i] = this.fPrime(x[i]);
        }
    }

    /**
     * Calculates the derivative of the activation function on each component of the x vector.
     * @param x The input vector where each component is where the derivative is calculated at.
     * @return A newly allocated vector of the derivative calculated at each component of the x vector.
     */
    public float[] fPrime(float[] x){
        //Check parameter
        Utility.checkNotNull((Object)x);

        float[] r = new float[x.length];

        this.fPrime(x, r);

        return r;
    }


    /**
     * Constructs the activation function from the string. Used to help construct neural networks from strings/text files.
     * @param s The string containing the activation function name. Check the toString() functions.
     * @return The activation function object described in the input string.
     */
    public static ActivationFunction constructFromString(String s){
        //Check parameter
        Utility.checkNotNull(s);

        if(s.equals("TANH")){
            return new Tanh();
        } else if(s.contains("LEAKYRELU")){
            String alphaString = s.replace("LEAKYRELU(", "").replace(")", "");
            float alpha = Float.parseFloat(alphaString);
            return new LeakyReLU(alpha);
        } else if(s.equals("LINEAR")){
            return new Linear();
        } else if(s.equals("SIGMOID")){
            return new Sigmoid();
        }

        return null;
    }
}