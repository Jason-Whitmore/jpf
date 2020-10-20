
/**
 * Definition for the Mean Squared Error loss function which is commonly used on regression tasks.
 */
public class MSE implements Loss{

    /**
     * Calculates the loss for each component in the output vector
     * @param yTrue The ground truth, or training output vector
     * @param yPredicted The predicted output vector from a model.
     * @return The loss calculation for each component in the output vector
     */
    public float[] calculateLossVector(float[] yTrue, float[] yPredicted){
        float[] r = new float[yTrue.length];

        for(int i = 0; i < r.length; i++){
            float difference = yTrue[i] - yPredicted[i];

            r[i] = difference * difference;
        }

        return r;
    }

    /**
     * Calculates the gradient (direction of steepest ascent) of the loss function
     * @param yTrue The ground truth, or training output vector
     * @param yPredicted The predicted output vector from a model.
     * @return The gradient vector for each component in the output vector.
     */
    public float[] calculateLossVectorGradient(float[] yTrue, float[] yPredicted){
        float[] r = new float[yTrue.length];

        for(int i = 0; i < r.length; i++){
            float difference = yPredicted[i] - yTrue[i];

            r[i] = 2 * difference;
        }

        return r;
    }

    /**
     * Calculates the scalar loss.
     * @param yTrue The ground truth, or training output vector.
     * @param yPredicted The predicted output vector from a model.
     * @return The scalar loss.
     */
    public float calculateLossScalar(float[] yTrue, float[] yPredicted){
        //TODO: Enforce array sizes

        float[] lossVector = calculateLossVector(yTrue, yPredicted);

        float sum = 0;

        for(int i = 0; i < yTrue.length; i++){
            sum += lossVector[i];
        }

        return sum / yTrue.length;
    }

}