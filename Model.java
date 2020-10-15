import java.util.ArrayList;


/**
 * Abstract class that defines parametric predictive models and their behaviors.
 */
public abstract class Model{

    /**
     * The collection of parameters used by the model for prediction.
     */
    private ArrayList<float[][]> parameters;

    /**
     * Sets the model parameters to a new collection of parameters.
     * Time complexity: O(1)
     * @param newParameters The new collection of parameters. To avoid issues, it's recommended that
     * The arraylist size, in addition to the dimensions of each 2d array, match the original parameter set.
     */
    public void setParameters(ArrayList<float[][]> newParameters){
        parameters = newParameters;
    }

    /**
     * Retrieves the model's current collection of parameters.
     * Time complexity: O(1)
     * @return The model's current collection of parameters
     */
    public ArrayList<float[][]> getParameters(){
        return parameters;
    }

    /**
     * Returns the number of parameters in the model.
     * Time complexity: O(n) where n is the size of the parameter arraylist
     * @return The number of parameters in the model.
     */
    public int getParameterCount(){
        int count = 0;
        
        for(int i = 0; i < parameters.size(); i++){
            count += parameters.get(i).length * parameters.get(i)[0].length;
        }

        return count;
    }

    /**
     * Makes predictions with multiple input vectors (mainly for multi-vector input models). If the model
     * does not support multiple input vectors (like LinearModel and PolynomialModel), then both  the input
     * and output ArrayLists should be of size 1.
     */
    public abstract ArrayList<float[]> predict(ArrayList<float[]> inputVectors);

    /**
     * Fits the model based on training parameters
     * @param x The ArrayList of input data. Each ArrayList element is for each input vector. For models with
     * one input vector, this ArrayList should be of size 1. x[i] should be the ith input vector in the
     * training dataset.
     * @param y The ArrayList of output data. Each ArrayList element is for each output vector. For models with
     * one output vector, this ArrayList should be of size 1. y[i] should be the ith output vector in the
     * training dataset.
     * @param epochs The number of complete passes over the dataset during training.
     * @param minibatchSize The number of training examples used in a single update of model parameters. A higher number
     * takes longer to compute, but is more representative of the dataset and produces a "smoother" gradient.
     * @param valueClip Elementwise clip to the gradient vector to prevent exploding gradients
     * @param opt The optimizer to use during training.
     * @param loss The loss function used to calculate the loss at the output vector(s)
     */
    public abstract void fit(ArrayList<float[][]> x, ArrayList<float[][]> y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss);

    /**
     * Calculates scalar loss based on provided data
     * @param x The ArrayList of test data inputs. If the model requires only 1 input vector, the ArrayList should be of size 1.
     * x[i] should be the ith test input vector.
     * @param y The ArrayList of test data outputs. If the model requires only 1 input vector, the ArrayList should be of size 1.
     * y[i] should be the ith test output vector.
     * @param loss The loss function to calculate the scalar loss with
     * @return The average scalar loss based on the provided data.
     */
    public abstract float calculateLoss(ArrayList<float[][]> x, ArrayList<float[][]> y, Loss loss);

}