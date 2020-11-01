import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
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

        //Parse the header (contains input, output vector sizes)
        int numInputs = 0;
        int numOutputs = 0;

        try{
            String headerLine = fileContents.substring(0, fileContents.indexOf("\n"));

            headerLine.replace("LinearModel(", "");
            headerLine.replace(")", "");

            String[] digitStrings = headerLine.split(",");

            numInputs = Integer.parseInt(digitStrings[0]);
            numOutputs = Integer.parseInt(digitStrings[1]);
        } catch(Exception e){
            System.err.println("Exception caught while parsing a LinearModel's saved model's header: " + e.getMessage());
            System.exit(1);
        }

        //Parse the transformation matrix

        int transformationStartIndex = 0;
        int transformationEndIndex = 0;
        String transformationString = "";
        try {
            transformationStartIndex = fileContents.indexOf("[[");
            transformationEndIndex = fileContents.indexOf("]]");
        
            transformationString = fileContents.substring(transformationStartIndex, transformationEndIndex);


        } catch(Exception e){
            System.err.println("Exception caught while parsing a LinearModel's transformation matrix from a file: " + e.getMessage());
            System.exit(1);
        }

        //Attempt to convert to a matrix
        try {
            float[][] transformationMatrix = LinearAlgebra.initializeFromString(transformationString);

            this.transformationMatrix = transformationMatrix;
        } catch(Exception e){
            System.err.println("Exception caught while parsing a LinearModel's transformation string from a file (check formatting): " + e.getMessage());
            System.exit(1);
        }
        
        

        

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
     * Fits the model's parameters to minimize the loss on the training dataset
     * @param x The training inputs. Should be of size 1 since there is only 1 input vector to this model.
     * @param y The training outputs. Should be of size 1 since there is only 1 output vector to this model.
     * @param epochs The number of epochs (or complete training passes over the training dataset). Should be greater than 0.
     * @param minibatchSize The number of training data examples used when making a single update to the model's parameters.
     * A higher number will take longer to compute, but the updates will be less noisy. Should be greater than 0.
     * @param valueClip Clips the gradient's components to be in the range [-valueClip, valueClip]. Helps to stop exploding gradients.
     * @param opt The optimizer to use during fitting.
     * @param Loss The loss function used to make the model more accurate to the training dataset.
     */
    public void fit(ArrayList<float[][]> x, ArrayList<float[][]> y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){
        if(x.size() != 1 || y.size() != 1){
            //TODO: Throw exception here
        }

        fit(x.get(0), y.get(0), epochs, minibatchSize, valueClip, opt, loss);
    }


    /**
     * Fits the model's parameters to minimize the loss on the training dataset.
     * @param x The training inputs. Number of columns should match the model's input size, and rows should be the number of data points.
     * @param y The training outputs. Number of columns should match the model's output size, and rows should be the number of data points.
     * @param epochs The number of epochs (or complete training passes over the training dataset). Should be greater than 0.
     * @param minibatchSize The number of training data examples used when making a single update to the model's parameters.
     * A higher number will take longer to compute, but the updates will be less noisy. Should be greater than 0.
     * @param valueClip Clips the gradient's components to be in the range [-valueClip, valueClip]. Helps to stop exploding gradients.
     * @param opt The optimizer to use during fitting.
     * @param Loss The loss function used to make the model more accurate to the training dataset.
     */
    public void fit(float[][] x, float[][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){


        for(int e = 0; e < epochs; e++){
            //calculate minibatch indicies
            ArrayList<ArrayList<Integer>> indicies = Utility.getMinibatchIndicies(x.length, minibatchSize);
            //for each minibatch...
            for(int mb = 0; mb < indicies.size(); mb++){

                ArrayList<float[][]> minibatchGradient = Utility.cloneArrays(getParameters());

                //for each data point in the minibatch
                for(int i = 0; i < indicies.get(mb).size(); i++){
                    int index = indicies.get(mb).get(i);

                    //calculate the gradient
                    ArrayList<float[][]> rawGradient = calculateGradient(x[index], y[index], loss);

                    //clip the gradient
                    Utility.clip(rawGradient, -valueClip, valueClip);

                    //add it to the minibatch pool
                    Utility.addGradient(minibatchGradient, rawGradient, 1.0f / indicies.get(mb).size());
                }

                minibatchGradient = opt.processGradient(minibatchGradient);

                //minibatch gradient is calculated. Add to the model's parameters
                Utility.addGradient(getParameters(), minibatchGradient, -1f);
            }

        }
    }

    /**
     * Calculates average scalar loss over multiple data points.
     * This overloaded function is not recommended to use with LinearModel.
     * @param x The inputs. ArrayList should be of size 1 since the model only has 1 input vector.
     * @param y The outputs. ArrayList should be of size 1 since the model only has 1 output vector.
     * @param loss The loss function to use when calculating loss on the datapoints.
     * @return The scalar loss
     */
    public float calculateLoss(ArrayList<float[][]> x, ArrayList<float[][]> y, Loss loss){
        //TODO: Enforce arraylist sizes

        return calculateLoss(x.get(0), y.get(0), loss);
    }

    /**
     * Calculates average scalar loss over multiple data points.
     * @param x The inputs. Number of columns should match the input vector size, number of rows is the number of data points to use.
     * @param y The outputs. Number of columns should match the output vector size, number of rows is the number of data points to use.
     * @param loss The loss function to use when calculating loss on the datapoints.
     * @return The scalar loss.
     */
    public float calculateLoss(float[][] x, float[][] y, Loss loss){
        //TODO: Enforce x, y shapes

        float sum = 0;
        for(int i = 0; i < x.length; i++){
            sum += calculateLoss(x[i], y[i], loss);
        }

        return sum / x.length;
    }

    /**
     * Calculates the loss on a single data point.
     * @param x The input vector.
     * @param y The output vector.
     * @param loss The loss function to use when calculating the data point.
     * @return The scalar loss.
     */
    public float calculateLoss(float[] x, float[] y, Loss loss){
        float[] yPred = predict(x);

        return loss.calculateLossScalar(y, yPred);
    }

    /**
     * Saves the model to disk with a human readable format.
     * @param filePath The file path to save the model to.
     */
    public void saveModel(String filePath){
        //TODO: Check to see if parameter arraylist is good

        StringBuilder sb = new StringBuilder();

        //create the header
        sb.append("LinearModel(");
        sb.append(this.numInputs);
        sb.append(this.numOutputs);
        sb.append(")\n");

        //Get the weight/transformation matrix
        sb.append(Utility.arrayToString(getParameters().get(0)));
        sb.append("\n");

        //get the bias matrix/vector
        sb.append(Utility.arrayToString(getParameters().get(1)));

        //write the string to disk

        try{
            FileWriter f = new FileWriter(filePath);
            f.write(sb.toString());

        } catch(IOException e){
            System.err.println("Exception occured: " + e.getMessage());
        }
        
    }
}