import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Class for the structure and behavior of Linear Models, which inherit from the abstract Model class.
 * This type of model is good for modeling data relationships that fit into the form y = Ax + b
 */
public class LinearModel extends Model{

    /**
     * The transformation, or weight matrix of the model. This is the matrix A in y = Ax + b
     */
    private float[][] transformationMatrix;

    /**
     * The bias vector of the model. This is the vector b in y = Ax + b
     */
    private float[][] biasVector;

    /**
     * The number of components in the input vector
     */
    private int numInputs;

    /**
     * The number of components in the output vector
     */
    private int numOutputs;

    
    /**
     * Constructor for a linear model of the form y = Ax + b
     * @param numInputs The number of components in the input vector
     * @param numOutputs The number of components in the output vector
     */
    public LinearModel(int numInputs, int numOutputs){
        transformationMatrix = LinearAlgebra.initializeRandomUniformMatrix(numOutputs, numInputs, -1f, 1f);

        biasVector = new float[numOutputs][1];

        ArrayList<float[][]> params = new ArrayList<float[][]>();

        params.add(transformationMatrix);
        params.add(biasVector);

        this.numInputs = numInputs;

        this.numOutputs = numOutputs;

        setParameters(params);
    }

    /**
     * Constructs a linear model based on the human readable format in saveModel()
     * @param filePath The file to read the model data from.
     */
    public LinearModel(String filePath){
        
        String fileContents = Utility.getTextFileContents(filePath);

        if(fileContents == null){
            //TODO: Error here
        }

        fileContents = fileContents.replace("[\n", "");
        fileContents = fileContents.replace("\n]", "");



        //Parse the transformation matrix

        int transformationStartIndex = 0;
        int transformationEndIndex = 0;
        String transformationString = "";
        try {
            transformationStartIndex = fileContents.indexOf("[[") + 0;
            transformationEndIndex = fileContents.indexOf("]]") + 2;
        
            transformationString = fileContents.substring(transformationStartIndex, transformationEndIndex);

            float[][] transformationMatrix = LinearAlgebra.initializeFromString(transformationString);

            this.transformationMatrix = transformationMatrix;

            //System.out.println(transformationString);
            //System.out.println(Utility.arrayToString(this.transformationMatrix));
        } catch(Exception e){
            System.err.println("Exception caught while parsing a LinearModel's transformation matrix from a file: " + e.getMessage());
            System.exit(1);
        }

        
        //Attempt to parse bias vector

        int biasStartIndex = 0;
        int biasEndIndex = 0;
        String biasString = "";

        try{
            biasStartIndex = fileContents.indexOf("[[", transformationEndIndex);
            biasEndIndex = fileContents.indexOf("]]", biasStartIndex) + 2;

            biasString = fileContents.substring(biasStartIndex, biasEndIndex);

            float[][] bias = LinearAlgebra.initializeFromString(biasString);

            
            this.biasVector = bias;
        } catch(Exception e){
            System.err.println("Exception caugh while parsing a LinearModel's bias matrix from a file: " + e.getMessage());
            System.exit(1);
        }
        
        numInputs = LinearAlgebra.getNumColumns(transformationMatrix);
        numOutputs = LinearAlgebra.getNumRows(biasVector);
    }

    /**
     * Changes the parameters of the LinearModel
     * @param newParams The new parameters to apply to this model. Must be of size 2 and each matrix must be
     * compatible with model's input and output counts.
     */
    public void setParameters(ArrayList<float[][]> newParams){
        if(newParams.size() != 2){
            //TODO: Not correct number of parameter matricies
        }

        //Validate parameter shapes

        if(LinearAlgebra.getNumRows(newParams.get(0)) != numOutputs || LinearAlgebra.getNumColumns(newParams.get(0)) != numInputs){
            //TODO: Bad transformation matrix dimensions
        }

        if(LinearAlgebra.getNumColumns(newParams.get(1)) != 1 || LinearAlgebra.getNumRows(newParams.get(1)) != numOutputs){
            //TODO: Bad bias matrix dimensions
        }

        super.setParameters(newParams);

        transformationMatrix = newParams.get(0);

        biasVector = newParams.get(1);
    }

    /**
     * Calculates gradient of Loss function with respect to the model parameters
     * @param inputVector The data sample's input used to calculate the gradient
     * @param outputVector The data sample's associated output used to calculate the gradient
     * @param loss The loss function used to calculate the gradient
     * @return The gradient as an ArrayList of matricies. These are in the same order as getParameters (transformation, bias)
     */
    private ArrayList<float[][]> calculateGradient(float[] inputVector, float[] outputVector, Loss loss){
        ArrayList<float[][]> gradient = new ArrayList<float[][]>(2);

        float[] yPredicted = this.predict(inputVector);

        float[] errorArray = loss.calculateLossVectorGradient(outputVector, yPredicted);

        //Calculate the error for the final step (adding the bias)
        float[][] biasGradient = LinearAlgebra.arrayToMatrix(errorArray);
        
        //Calculate error for the transformation matrix
        float[][] transformationGradient = new float[LinearAlgebra.getNumRows(transformationMatrix)][LinearAlgebra.getNumColumns(transformationMatrix)];

        for(int r = 0; r < LinearAlgebra.getNumRows(transformationGradient); r++){
            for(int c = 0; c < LinearAlgebra.getNumColumns(transformationGradient); c++){
                transformationGradient[r][c] = inputVector[c] * biasGradient[r][0];
            }
        }

        gradient.add(0, transformationGradient);
        gradient.add(1, biasGradient);

        return gradient;
    }


    /**
     * Makes predictions on multiple input vectors
     * @param inputVectors The set of input vectors to make predictions on
     * @return An ArrayList of prediction/output vectors associated with the ArrayList of input vectors
     */
    public ArrayList<float[]> predict(ArrayList<float[]> inputVectors){
        ArrayList<float[]> r = new ArrayList<float[]>();

        for(int i = 0; i < inputVectors.size(); i++){
            r.add(predict(inputVectors.get(i)));
        }

        return r;
    }

    /**
     * Makes a prediction from an input vector
     * @param inputVector The vector used to make predictions
     * @return The output or prediction vector
     */
    public float[] predict(float[] inputVector){
        float[] result = new float[numOutputs];
        float[][] resultMatrix = new float[numOutputs][1];

        float[][] matrixB = new float[inputVector.length][1];

        for(int i = 0; i < inputVector.length; i++){
            matrixB[i][0] = inputVector[i];
        }

        LinearAlgebra.matrixMultiply(transformationMatrix, matrixB, resultMatrix);

        //Add the bias
        LinearAlgebra.matrixAdd(resultMatrix, biasVector, resultMatrix);

        //Turn back into a vector
        for(int i = 0; i < result.length; i++){
            result[i] = resultMatrix[i][0];
        }

        return result;
    }



    /**
     * Saves the model to disk with a human readable format.
     * @param filePath The file path to save the model to.
     */
    public void saveModel(String filePath){
        //TODO: Check to see if parameter arraylist is good

        StringBuilder sb = new StringBuilder();

        //append the parameters

        sb.append(Utility.arraysToString(getParameters()));

        //write the string to disk

        Utility.writeStringToFile(filePath, sb.toString());
        
    }
}