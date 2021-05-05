import java.util.ArrayList;


/**
 * Abstract class that defines parametric predictive models and their behaviors.
 */
public abstract class Model{

    /**
     * The collection of parameters used by the model for prediction.
     */
    protected ArrayList<float[][]> parameters;

    /**
     * Retrieves the model's current collection of parameters.
     * Time complexity: O(1)
     * @return The model's current collection of parameters
     */
    public ArrayList<float[][]> getParameters(){
        return this.parameters;
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
     * Saves the model to disc using a human readable format.
     * @param filePath The file path to save the model to.
     */
    public abstract void saveModel(String filePath);

}