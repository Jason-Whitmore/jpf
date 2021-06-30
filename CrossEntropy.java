/**
 * Class that defines the CrossEntropy loss function, which is a concrete class derived from the Loss abstract class.
 * This loss is used for classification tasks 
 */
public class CrossEntropy implements Loss{

    /**
     * Constructor that simply initializes the loss function object.
     */
    public CrossEntropy(){

    }

    /**
     * The cross entropy loss function for single variable input and output
     * @param yTrue The label variable, which should be either 1 or 0
     * @param yPredicted The predicted output from a model, should be in range (0, 1)
     * @return The scalar loss
     */
    private float CELoss(float yTrue, float yPredicted){
        if(yTrue == 1){
            return (float)(-Math.log((yPredicted)));
        } else {
            return (float)(-Math.log((1f - yPredicted)));
        }
    }

    //TODO: complete this
    private float CELossPrime(float yTrue, float yPredicted){
        return 0;
    }


    public float[] calculateLossVector(float[] yTrue, float[] yPredicted){
        float[] r = new float[yTrue.length];

        for(int i = 0; i < r.length; i++){
            r[i] = this.CELoss(yTrue[i], yPredicted[i]);
        }

        return r;
    }

    public float[] calculateLossVectorGradient(float[] yTrue, float[] yPredicted){
        float[] r = new float[yTrue.length];

        for(int i = 0; i < r.length; i++){
            r[i] = this.CELossPrime(yTrue[i], yPredicted[i]);
        }

        return r;
    }

    public float calculateLossScalar(float[] yTrue, float[] yPredicted){
        float[] lossVector = this.calculateLossVector(yTrue, yPredicted);

        return Utility.mean(lossVector);
    }
}