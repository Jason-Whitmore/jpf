import java.util.ArrayList;

/**
 * Class for the structure and behavior of Linear Models, which inherit from the abstract Model class.
 * This type of model is good for modeling data relationships that fit into the form f(x) = Wx + b
 */
public class LinearModel extends SimpleModel{

    /**
     * The transformation or weight matrix of the model. This is the matrix A in y = Ax + b
     */
    private float[][] transformationMatrix;

    /**
     * The bias vector of the model represented as a 1 column matrix. This is the vector b in y = Ax + b
     */
    private float[][] biasMatrix;

    
    /**
     * Constructor for a linear model of the form y = Ax + b
     * @param numInputs The number of components in the input vector
     * @param numOutputs The number of components in the output vector
     */
    public LinearModel(int numInputs, int numOutputs){
        //Invalid parameters already checked in the super constructor.
        super(numInputs, numOutputs);

        this.transformationMatrix = LinearAlgebra.initializeRandomUniformMatrix(numOutputs, numInputs, -1f, 1f);

        this.biasMatrix = new float[numOutputs][1];

        this.parameters.add(this.transformationMatrix);
        this.parameters.add(this.biasMatrix);
    }

    /**
     * Constructs a linear model based on the human readable format in saveModel()
     * @param filePath The file to read the model data from.
     */
    public LinearModel(String filePath){
        
        String fileContents = Utility.getTextFileContents(filePath);

        fileContents = fileContents.replace("LINEARMODEL\n", "");

        ArrayList<float[][]> params = Utility.stringToMatrixList(fileContents);

        this.parameters = params;

        this.transformationMatrix = params.get(0);
        this.biasMatrix = params.get(1);
        
        this.numInputs = LinearAlgebra.getNumColumns(transformationMatrix);
        this.numOutputs = LinearAlgebra.getNumRows(biasMatrix);

        //Check if the num input/output fields are not valid
        if(this.numInputs <= 0){
            throw new AssertionError("Size of input vector for model is invalid.");
        }
        if(this.numOutputs <= 0){
            throw new AssertionError("Size of output vector for model is invalid.");
        }
    }

    /**
     * Calculates gradient of Loss function with respect to the model parameters
     * @param inputVector The data sample's input used to calculate the gradient
     * @param outputVector The data sample's associated output used to calculate the gradient
     * @param loss The loss function used to calculate the gradient
     * @return The gradient as an ArrayList of matricies. These are in the same order as getParameters (transformation, bias)
     */
    protected ArrayList<float[][]> calculateGradient(float[] inputVector, float[] outputVector, Loss loss){
        //Check parameters
        Utility.checkNotNull((Object)inputVector, (Object)outputVector, loss);
        Utility.checkEqual(inputVector.length, this.numInputs);
        Utility.checkEqual(outputVector.length, this.numOutputs);

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
     * Makes a prediction from an input vector
     * @param inputVector The vector used to make predictions
     * @return The output or prediction vector
     */
    public float[] predict(float[] inputVector){
        //Check parameters
        Utility.checkNotNull((Object)inputVector);
        Utility.checkEqual(inputVector.length, this.numInputs);

        
        float[] result = new float[numOutputs];
        float[][] resultMatrix = new float[numOutputs][1];

        float[][] matrixB = new float[inputVector.length][1];

        for(int i = 0; i < inputVector.length; i++){
            matrixB[i][0] = inputVector[i];
        }

        LinearAlgebra.matrixMultiply(transformationMatrix, matrixB, resultMatrix);

        //Add the bias
        LinearAlgebra.matrixAdd(resultMatrix, biasMatrix, resultMatrix);

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
        //Construct the string
        String contents = Utility.arraysToString(this.getParameters());

        contents = "LINEARMODEL\n" + contents; 

        //write the string to disk
        boolean success = Utility.writeStringToFile(filePath, contents);

        if(!success){
            System.err.println("Error: Linear model could not be saved.");
        }
        
    }
}