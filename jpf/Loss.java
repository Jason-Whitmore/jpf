package jpf;

/**
 * Interface that defines the functions used for loss functions.
 */
public interface Loss{

    /**
     * Calculates the loss for each component in the output vector
     * @param yTrue The ground truth, or training output vector
     * @param yPredicted The predicted output vector from a model.
     * @return The loss calculation for each component in the output vector
     */
    public float[] calculateLossVector(float[] yTrue, float[] yPredicted);

    /**
     * Calculates the gradient (direction of steepest ascent) of the loss function for each component
     * in the vector
     * @param yTrue The ground truth, or training output vector
     * @param yPredicted The predicted output vector from a model.
     * @return The gradient vector for each component in the output vector.
     */
    public float[] calculateLossVectorGradient(float[] yTrue, float[] yPredicted);

    /**
     * Calculates the scalar loss.
     * @param yTrue The ground truth, or training output vector.
     * @param yPredicted The predicted output vector from a model.
     * @return The scalar loss.
     */
    public float calculateLossScalar(float[] yTrue, float[] yPredicted);
}