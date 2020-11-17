import java.util.ArrayList;


/**
 * Class which implements the RMSProp adaptive learning rate optimizer.
 * It adapts the learning rate on a per parameter basis using exponentially decaying memory.
 */
public class RMSProp implements Optimizer{

    /**
     * The scalar learning rate to multiply the processed gradient by.
     */
    private float learningRate;

    /**
     * The exponential decay factor on the internal memory. If rho = 1, then no changes will be made to the memory.
     * If rho = 0, then the memory is never written to.
     */
    private float rho;

    /**
     * The small scalar value used to prevent divide by zero errors.
     */
    private float epsilon;

    /**
     * The exponentially decaying memory of each gradient component squared.
     */
    ArrayList<float[][]> gradSquare;

    /**
     * Default constructor for the RMSProp optimizer. Sets the learning rate to 0.0001, rho to 0.9, epsilon to 0.000001.
     */
    public RMSProp(){
        learningRate = 0.0001f;

        rho = 0.9f;

        epsilon = 0.000001f;

        gradSquare = null;
    }

    /**
     * Constructs an RMSProp optimizer with user specified settings
     * @param learningRate The scalar learning rate. Should be greater than 0.
     * @param rho The exponential decay factor in the internal memory. Should be in range of (0, 1), closer to 1. Recommended as 0.9
     * @param epsilon Small constant used to avoid divide by zero errors. Should be greater than 0.
     */
    public RMSProp(float learningRate, float rho, float epsilon){
        this.learningRate = learningRate;

        this.rho = rho;

        this.epsilon = epsilon;

        gradSquare = null;
    }


    public ArrayList<float[][]> processGradient(ArrayList<float[][]> rawGradient){

        //Create internal state if it isn't already there.
        if(gradSquare == null){
            ArrayList<float[][]> newState = Utility.cloneArrays(rawGradient);
            Utility.clearArrays(newState);
            gradSquare = newState;
        }



        //Update the state of the optimizer (estimation for avg squared gradient)
        for(int i = 0; i < gradSquare.size(); i++){
            for(int r = 0; r < gradSquare.get(i).length; r++){
                for(int c = 0; c < gradSquare.get(i)[r].length; c++){

                    //maintain exponentially decaying average
                    gradSquare.get(i)[r][c] = (rho * gradSquare.get(i)[r][c]) + ((1 - rho) * ((float)Math.pow(rawGradient.get(i)[r][c], 2)));
                }
            }
        }


        ArrayList<float[][]> grad = Utility.cloneArrays(rawGradient);
        Utility.clearArrays(grad);
        
        //populate gradient
        for(int i = 0; i < grad.size(); i++){
            for(int r = 0; r < grad.get(i).length; r++){
                for(int c = 0; c < grad.get(i)[r].length; c++){
                    grad.get(i)[r][c] = learningRate * (1f / (float)Math.sqrt(gradSquare.get(i)[r][c] + epsilon)) * rawGradient.get(i)[r][c];
                }
            }
        }

        return grad;
    }

    /**
     * Resets the optimizer state. Useful when retraining a model on new data.
     */
    public void resetState(){
        if(gradSquare == null){
            return;
        }

        ArrayList<float[][]> newState = Utility.cloneArrays(gradSquare);
        Utility.clearArrays(newState);

        gradSquare = newState;
    }
}