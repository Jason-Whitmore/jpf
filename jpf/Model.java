package jpf;

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
     * The total number of parameters, represented as floating point numbers, used in the model.
     */
    protected int parameterCount;


    /**
     * Initializes the parameter and parameter count fields in the abstract class.
     */
    public Model(){
        this.parameters = new ArrayList<float[][]>();

        this.parameterCount = -1;
    }

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
     * Time complexity: O(n) where n is the size of the parameter arraylist. O(1) on subsequent calls.
     * @return The number of parameters in the model.
     */
    public int getParameterCount(){


        if(this.parameterCount != -1){
            return this.parameterCount;
        }

        int count = 0;
        
        for(int i = 0; i < parameters.size(); i++){
            count += parameters.get(i).length * parameters.get(i)[0].length;
        }

        this.parameterCount = count;

        return this.parameterCount;
    }


    /**
     * Saves the model to disc using a human readable format.
     * @param filePath The file path to save the model to.
     */
    public abstract void saveModel(String filePath);

}