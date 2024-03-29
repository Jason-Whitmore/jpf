package jpf;

import java.util.ArrayList;


/**
 * Class for the classic Stochastic Gradient Descent optimizer which performs
 * no processing to the input gradients.
 */
public class SGD implements Optimizer{

    /**
     * The scalar to multiply the gradient by before applying to the model's parameters.
     * Should be greater than 0.
     */
    private float learningRate;

    /**
     * Default constructor for SGD. Sets a learning rate of 0.0001.
     */
    public SGD(){
        this.learningRate = 0.0001f;
    }

    /**
     * Constructor for SGD that uses a user defined learning rate
     * @param learningRate The scalar to multiply the gradient by.
     * Smaller values converge slowly, but with more stability.
     */
    public SGD(float learningRate){
        //Check parameter
        Utility.checkReal(learningRate);
        if(learningRate <= 0){
            throw new AssertionError("Learning rate should be > 0");
        }

        this.learningRate = learningRate;
    }


    /**
     * Performs no adjustment to the raw gradient other than the typical scaling
     * by the learning rate
     * @param rawGradient A raw, unprocessed gradient of the model's parameters evaluated at a data point.
     * @return The processed gradient, ready to be applied to the model.
     */
    public ArrayList<float[][]> processGradient(ArrayList<float[][]> rawGradient){
        //Check parameter
        Utility.checkNotNull(rawGradient);
        
        Utility.scaleList(rawGradient, this.learningRate);

        return rawGradient;
    }
}