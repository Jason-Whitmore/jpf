package jpf;

/**
 * Class that defines the CrossEntropy loss function, which is a concrete class derived from the Loss abstract class.
 * This loss is used for classification tasks 
 */
public class CrossEntropy implements Loss{

    /**
     * Very small value > 0 that prevents log(0) from being calculated
     */
    private float epsilon;

    /**
     * Constructor that simply initializes the loss function object.
     */
    public CrossEntropy(){
        this.epsilon = 0.0001f;
    }

    /**
     * Constructor that creates the loss object with a specified epsilon.
     * @param epsilon User defined epsilon. Should be > 0. Prevents loss from exploding.
     */
    public CrossEntropy(float epsilon){
        if(epsilon <= 0){
            throw new AssertionError("epsilon argument must be > 0.");
        }

        this.epsilon = epsilon;
    }

    /**
     * The cross entropy loss function for single variable input and output
     * @param yTrue The label variable, which should be either 1 or 0
     * @param yPredicted The predicted output from a model, should be in range (0, 1)
     * @return The scalar loss
     */
    private float CELoss(float yTrue, float yPredicted){
        if(yTrue == 1){
            return (float)(-Math.log((yPredicted + this.epsilon)));
        } else {
            return (float)(-Math.log((1f - yPredicted + this.epsilon)));
        }
    }

    /**
     * The cross entropy loss function derivative for a single component
     * @param yTrue The label or ground truth. Should be either 1 or 0.
     * @param yPredicted The predicted output from a model, should be in range (0,1)
     * @return The derivative of the loss function.
     */
    private float CELossPrime(float yTrue, float yPredicted){

        if(yTrue == 1){
            return -1f / (yPredicted + this.epsilon);
        } else {
            return 1f / (1 - yPredicted + this.epsilon);
        }
    }


    public float[] calculateLossVector(float[] yTrue, float[] yPredicted){
        //Check parameters
        Utility.checkArrayLengthsEqual(yTrue, yPredicted);

        float[] r = new float[yTrue.length];

        for(int i = 0; i < r.length; i++){
            //Check to see if floats are real
            Utility.checkReal(yTrue[i], yPredicted[i]);

            r[i] = this.CELoss(yTrue[i], yPredicted[i]);
        }

        return r;
    }

    public float[] calculateLossVectorGradient(float[] yTrue, float[] yPredicted){
        //Check parameters
        Utility.checkArrayLengthsEqual(yTrue, yPredicted);

        float[] r = new float[yTrue.length];

        for(int i = 0; i < r.length; i++){
            //Check to see if floats are real
            Utility.checkReal(yTrue[i], yPredicted[i]);

            r[i] = this.CELossPrime(yTrue[i], yPredicted[i]);
        }

        return r;
    }

    public float calculateLossScalar(float[] yTrue, float[] yPredicted){
        //Check parameters
        Utility.checkArrayLengthsEqual(yTrue, yPredicted);

        float[] lossVector = this.calculateLossVector(yTrue, yPredicted);
        return Utility.mean(lossVector);
    }
}