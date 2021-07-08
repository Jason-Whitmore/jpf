import java.util.ArrayList;

/**
 * Defines abstract class for simple models which have one input vector and one output vector.
 * Derived from the Model abstract class.
 */
public abstract class SimpleModel extends Model{

    /**
     * The length of the input vector
     */
    protected int numInputs;

    /**
     * The length of the output vector.
     */
    protected int numOutputs;

    /**
     * Constructor for the SimpleModel abstract class. Calls constructor for Model class
     * and populates the input and output vector size fields.
     * @param numInputs The number of components in the input vector. Should be >= 1.
     * @param numOutputs The number of components in the output vector. Should be >= 1.
     */
    public SimpleModel(int numInputs, int numOutputs){
        super();

        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
    }

    /**
     * Basic constructor for the SimpleModel abstract class. Calls constructor for Model class
     * but does not populate the input and output vector size fields. This is useful when
     * the input and output vector size fields are unknown from constructor parameters,
     * such as when a model is being created from a file name only.
     */
    public SimpleModel(){
        super();
    }

    /**
     * @return The length of the input vector.
     */
    public int getNumInputs(){
        return this.numInputs;
    }

    /**
     * @return The length of the output vector.
     */
    public int getNumOutputs(){
        return this.numOutputs;
    }

    /**
     * Makes a prediction on a single input vector.
     * @param inputVector The vector to use as input for the function.
     * @return The output vector produced from the model and input vector.
     */
    public abstract float[] predict(float[] inputVector);


    /**
     * Makes predictions on multiple input vectors.
     * @param inputVectors The input vectors, where the first index specifies the input vector,
     * and the second index specifies the component of the input vector.
     * @return The predicted output vectors, corresponding to the input vectors.
     */
    public float[][] predict(float[][] inputVectors){
        //Check parameters
        Utility.checkNotNull((Object)inputVectors);
        Utility.checkMatrixRectangle(inputVectors);
        Utility.checkArrayNotEmpty(inputVectors);

        if(inputVectors[0].length != this.numInputs){
            throw new AssertionError("Predict input vector length does not match model input length");
        }

        //Make predictions
        float[][] outputVectors = new float[inputVectors.length][];

        for(int i = 0; i < outputVectors.length; i++){
            outputVectors[i] = predict(inputVectors[i]);
        }

        return outputVectors;
    }

    /**
     * Calculates the gradient of the loss function with a given input and output vector.
     * @param x The input vector.
     * @param y The target output vector.
     * @param loss The loss function which is to be minimized during training.
     * @return The gradient as an ArrayList of matricies of the same shape as the model's parameters.
     */
    protected abstract ArrayList<float[][]> calculateGradient(float[] x, float[] y, Loss loss);

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
        this.fitParameterCheck(x, y, epochs, minibatchSize, valueClip, opt, loss);

        for(int e = 0; e < epochs; e++){
            //calculate minibatch indicies
            ArrayList<ArrayList<Integer>> indicies = Utility.getMinibatchIndicies(x.length, minibatchSize);
            //for each minibatch...
            for(int mb = 0; mb < indicies.size(); mb++){

                ArrayList<float[][]> minibatchGradient = Utility.cloneArrays(getParameters());
                Utility.clearArrays(minibatchGradient);

                //for each data point in the minibatch
                for(int i = 0; i < indicies.get(mb).size(); i++){
                    int index = indicies.get(mb).get(i);

                    //calculate the gradient
                    ArrayList<float[][]> rawGradient = calculateGradient(x[index], y[index], loss);

                    //clip the gradient if applicable
                    if(valueClip > 0){
                        Utility.clip(rawGradient, -valueClip, valueClip);
                    }

                    //add it to the minibatch pool
                    Utility.addList(minibatchGradient, rawGradient, 1.0f / indicies.get(mb).size());
                }

                minibatchGradient = opt.processGradient(minibatchGradient);

                //minibatch gradient is calculated. Add to the model's parameters
                Utility.addList(getParameters(), minibatchGradient, -1f);
            }

        }

    }

    /**
     * Checks the parameters for the fit function for validity. This is a separate method to avoid wasting space in an already long function.
     * 
     * @param x The training inputs
     * @param y The training outputs
     * @param epochs The number of epochs to train for. Should be > 0.
     * @param minibatchSize The number of training examples used in a parameter update. Should be > 0, but less than number of examples.
     * @param valueClip The maximum absolute value a gradient component can be. Should be > 0
     * @param opt The optimizer used for training.
     * @param loss The loss function to minimize during training.
     */
    private void fitParameterCheck(float[][] x, float[][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){
        Utility.checkNotNull(x, y, opt, loss);
        Utility.checkArrayNotEmpty(x);
        Utility.checkArrayNotEmpty(y);
        Utility.checkMatrixRectangle(x);
        Utility.checkMatrixRectangle(y);
        Utility.checkArrayLengthsEqual(x, y);
        
        //Check x and y to see if the number of columns match the model's input and output sizes.
        
    }


    /**
     * Calculates scalar loss on one data sample.
     * @param x The model input.
     * @param y The model output.
     * @param loss The loss function to use to calculate the scalar loss.
     * @return The scalar loss
     */
    public float calculateLoss(float[] x, float[] y, Loss loss){
        Utility.checkNotNull((Object)x, (Object)y, loss);

        if(x.length != this.numInputs){
            throw new AssertionError("Input vector length incorrect for model input.");
        }

        if(y.length != this.numOutputs){
            throw new AssertionError("Output vector length incorrect for model output.");
        }

        float[] yPred = predict(x);
        return loss.calculateLossScalar(y, yPred);
    }
    
    /**
     * Calculates the average scalar loss on multiple data samples.
     * @param x The model inputs.
     * @param y The model ouputs corresponding with inputs.
     * @param loss The loss function to use to calculate the scalar losses.
     * @return The average scalar loss.
     */
    public float calculateLoss(float[][] x, float[][] y, Loss loss){
        Utility.checkNotNull((Object)x, (Object)y, loss);
        Utility.checkMatrixRectangle(x);
        Utility.checkMatrixRectangle(y);
        Utility.checkArrayLengthsEqual(x, y);

        float sum = 0;

        for(int i = 0; i < x.length; i++){
            sum += calculateLoss(x[i], y[i], loss);
        }

        return sum / x.length;
    }
}