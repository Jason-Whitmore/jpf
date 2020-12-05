

/**
 * Defines the ActivationFunction class for use in neural networks
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
        for(int i = 0; i < x.length; i++){
            dest[i] = f(dest[i]);
        }
    }

    /**
     * Applies the activation function on each component of the x vector.
     * @param x The vector to apply the activation function to.
     * @return A newly allocated vector of the x vector after applying the activation function.
     */
    public float[] f(float[] x){
        float[] r = new float[x.length];

        f(x, r);

        return r;
    }


    /**
     * Calculates the derivative of the activation function on each component of the x vector.
     * @param x The input vector where each component is where the derivative is calculated at.
     * @param dest The vector where the derivatives will be placed.
     */
    public void fPrime(float[] x, float[] dest){
        for(int i = 0; i < x.length; i++){
            dest[i] = fPrime(x[i]);
        }
    }

    /**
     * Calculates the derivative of the activation function on each component of the x vector.
     * @param x The input vector where each component is where the derivative is calculated at.
     * @return A newly allocated vector of the derivative calculated at each component of the x vector.
     */
    public float[] fPrime(float[] x){
        float[] r = new float[x.length];

        fPrime(x, r);

        return r;
    }
}