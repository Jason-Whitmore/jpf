package jpf;

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
        this.setLearningRate(0.0001f);

        this.setRho(0.9f);

        this.setEpsilon(0.000001f);

        this.gradSquare = null;
    }

    /**
     * Constructs an RMSProp optimizer with user specified settings
     * @param learningRate The scalar learning rate. Should be greater than 0.
     * @param rho The exponential decay factor in the internal memory. Should be in range of (0, 1), closer to 1. Recommended as 0.9
     * @param epsilon Small constant used to avoid divide by zero errors. Should be greater than 0.
     */
    public RMSProp(float learningRate, float rho, float epsilon){
        this.setLearningRate(learningRate);

        this.setRho(rho);

        this.setEpsilon(epsilon);

        this.gradSquare = null;
    }

    /**
     * @return The learning rate used for this optimizer.
     */
    public float getLearningRate(){
        return learningRate;
    }

    /**
     * Sets this optimizer's learning rate to the user specified value. Should be greater than 0.
     * @param newLr The new learning rate to set this optimizer to.
     */
    public void setLearningRate(float newLr){
        if(newLr <= 0){
            throw new AssertionError("newLR should be > 0");
        }

        this.learningRate = newLr;
    }


    /**
     * @return This optimizer's rho value, which controls the speed which the internal memory decays.
     */
    public float getRho(){
        return rho;
    }

    /**
     * Sets this optimizer's rho value, which controls the speed which the internal memory decays.
     * 
     * @param newRho The new rho value to set this optimizer to. This value should be in range (0,1),
     * where 0 is instant decay, and 1 is no decay. Recommended as 0.9
     */
    public void setRho(float newRho){
        if(newRho < 0 || newRho >= 1){
            throw new AssertionError("newRho should be in range [0, 1)");
        }
        this.rho = newRho;
    }

    /**
     * @return Returns the epsilon value used to prevent divide by zero errors.
     */
    public float getEpsilon(){
        return epsilon;
    }

    /**
     * Sets the epsilon value used to prevent divide by zero errors.
     * @param eps The new epsilon value. Should be > 0.
     */
    public void setEpsilon(float eps){
        if(eps <= 0){
            throw new AssertionError("eps should be > 0");
        }

        this.epsilon = eps;
    }


    /**
     * Updates the internal state and applies the adaptive learning rate to each component of the raw gradient.
     * @param rawGradient The unprocessed gradient from a model.
     * @return The processed model with the applied per-parameter adaptive learning rate.
     */
    public ArrayList<float[][]> processGradient(ArrayList<float[][]> rawGradient){
        //Check parameters
        Utility.checkNotNull(rawGradient);

        //Create internal state if it isn't already there.
        if(gradSquare == null){
            ArrayList<float[][]> newState = Utility.cloneArrays(rawGradient);
            Utility.clearArrays(newState);
            this.gradSquare = newState;
        }



        //Update the state of the optimizer (estimation for avg squared gradient)
        for(int i = 0; i < gradSquare.size(); i++){
            for(int r = 0; r < gradSquare.get(i).length; r++){
                for(int c = 0; c < gradSquare.get(i)[r].length; c++){
                    //maintain exponentially decaying average
                    this.gradSquare.get(i)[r][c] = (rho * gradSquare.get(i)[r][c]) + ((1 - rho) * ((float)Math.pow(rawGradient.get(i)[r][c], 2)));
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
     * Resets the optimizer internal memory. Useful when retraining a model on new data.
     */
    public void resetState(){
        this.gradSquare = null;
    }
}