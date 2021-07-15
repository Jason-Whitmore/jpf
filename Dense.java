
/**
 * Class that defines the Dense layer that makes up much of a typical neural network.
 */
public class Dense extends Layer {

    /**
     * The activation function of the layer, which may provide nonlinearities
     * that improve the capacity of the layer and model.
     */
    private ActivationFunction activationFunction;

    /**
     * The weight, or kernel matrix, which is multiplied with the input vector.
     * These parameters are used to determine how much influence the inputs have
     * on the layer's output.
     */
    private float[][] weightMatrix;

    /**
     * The biasMatrix is a representation of a bias vector which influence the
     * layer's outputs without using the layer's inputs.
     */
    private float[][] biasMatrix;

    /**
     * The sum matrix is a representation of the sum vector which is expressed as
     * Wx + b where W is the weight matrix, x is the input vector, and b is the bias
     * vector. This matrix is used for both the forward and backward passes.
     */
    private float[][] sumMatrix;

    /**
     * The standard Dense layer constructor with user specified properties.
     * 
     * @param numUnits The number of units or nodes in the layer. Should be >= 1.
     * 
     * @param f The activation function that will be applied to the layer sum.
     * A nonlinear activation function increases the capacity of the layer and model.
     * 
     * @param inputLayer The input layer which connects to this layer. Outputs
     * from the input layer become inputs to this layer.
     */
    public Dense(int numUnits, ActivationFunction f, Layer inputLayer){
        super();
        //Check parameters
        Utility.checkNotNull(f, inputLayer);

        if(numUnits <= 0){
            throw new AssertionError("numUnits must be >= 1");
        }

        //Initialize the activation function and input layer
        this.activationFunction = f;
        this.inputLayers.add(inputLayer);

        //Connect the input layer with this layer
        this.connectInputAndOutputLayers();

        //Determine the size of the input vector by looking at the input layer's output vector.
        int inputLayerOutputSize = inputLayer.getOutputVector().length;

        //Initialize the input and output vectors with correct lengths.
        this.inputVector = new float[inputLayerOutputSize];
        this.outputVector = new float[numUnits];

        //Initialize the weight and bias parameter matricies. Randomize the weight matrix entries.
        this.weightMatrix = new float[numUnits][inputLayerOutputSize];
        Utility.initializeUniform(weightMatrix, -1f, 1f);

        this.biasMatrix = new float[numUnits][1];

        //Add matricies to parameter list
        this.parameters.add(weightMatrix);
        this.parameters.add(biasMatrix);

        //Set up gradients
        this.gradient = Utility.cloneArrays(getParameters());
        Utility.clearArrays(gradient);

        //Set up backprop arrays
        this.dLdX = new float[inputLayerOutputSize];
        this.dLdY = new float[numUnits];
    }


    /**
     * Constructs a dense layer from a string description. Check toString().
     * This string should start with "DENSE(" and ends with "\n]".
     * @param layerInfoString The layer info string returned by the toString() method.
     */
    public Dense(String layerInfoString){
        super();
        //Check param string
        Utility.checkNotNull(layerInfoString);

        //Isolate the activation function string and initialize
        this.initializeActivationFunctionFromString(layerInfoString);

        String paramString = layerInfoString.substring(layerInfoString.indexOf(")\n") + 2, layerInfoString.lastIndexOf("\n"));
        this.parameters = Utility.stringToMatrixList(paramString);

        
        this.weightMatrix = this.parameters.get(0);
        this.biasMatrix = this.parameters.get(1);

        //From the parsed parameters, get the layer size and the input vector size
        int numUnits = this.biasMatrix.length;
        if(numUnits <= 0){
            throw new AssertionError("numUnits must be >= 1");
        }

        int inputSize = this.weightMatrix[0].length;

        //Allocate and initialize vectors
        this.inputVector = new float[inputSize];
        this.outputVector = new float[numUnits];

        this.dLdX = new float[inputSize];
        this.dLdY = new float[numUnits];

        //Allocate and initialize gradients
        this.gradient = Utility.cloneArrays(getParameters());
        Utility.clearArrays(this.gradient);

        
    }

    /**
     * Initializes the activation function from the layer info string from saving the model to disk.
     * 
     * @param layerInfoString The layer info string read from disk. This layer info string should 
     * only be the text from saving the layer, not the entire model. Check the toString() method
     * for specific syntax.
     */
    private void initializeActivationFunctionFromString(String layerInfoString){
        String headerInfo = layerInfoString.substring(0, layerInfoString.indexOf(")") + 1);
        headerInfo = headerInfo.replace("(", "");
        headerInfo = headerInfo.replace(")", "");

        String[] headerInfoSplit = headerInfo.split(",");
        headerInfoSplit[1] = headerInfoSplit[1].trim();
        this.activationFunction = ActivationFunction.constructFromString(headerInfoSplit[1]);
    }

    public void forwardPass(){
        //Bring over the previous layer's output into this layer's input vector
        this.initializeInputVectorCopy();

        //Do matrix multiplication on the input vector: Wx
        float[][] wx = LinearAlgebra.matrixMultiply(weightMatrix, LinearAlgebra.arrayToMatrix(inputVector));

        //add bias
        this.sumMatrix = LinearAlgebra.matrixAdd(wx, biasMatrix);

        //use as input to activation function.
        this.outputVector = activationFunction.f(LinearAlgebra.matrixToArray(sumMatrix));
    }

    public void backwardPass(){
        //Determine the error vector from the next layers
        this.initializedLdY();

        //Determine the dLdS vector by applying the chain rule: dLdS = dLdY * dYdS
        float[] dLdS = LinearAlgebra.elementwiseMultiply(this.dLdY, activationFunction.fPrime(LinearAlgebra.matrixToArray(this.sumMatrix)));

        //In a dense layer, the bias gradient is just the dLdS vector.
        float[] biasGradient = dLdS.clone();
        this.gradient.set(1, LinearAlgebra.arrayToMatrix(biasGradient));


        //Determine the weight gradient, dLdW by applying the chain rule: dLdW = dLdS * dSdW
        float[][] weightGradient = new float[weightMatrix.length][weightMatrix[0].length];

        for(int r = 0; r < weightMatrix.length; r++){
            for(int c = 0; c < weightMatrix[0].length; c++){
                /**
                 * Note: dSdW is just the components of the input vector since the input vector multiplied
                 * by the weight matrix influences the sum (multiplication rule).
                 */
                weightGradient[r][c] = dLdS[r] * this.inputVector[c];
            }
        }

        this.gradient.set(0, weightGradient);


        //Finally, populate dLdX for the input layer's backprop step using the chain and addition rules: dLdX = dLdS * dSdX
        for(int i = 0; i < this.inputVector.length; i++){
            for(int j = 0; j < this.outputVector.length; j++){
                //dLdX is the weight matrix because of the multiplication rule.
                this.dLdX[i] += weightMatrix[j][i] * dLdS[j];
            }
        }
    }

    /**
     * Converts the Dense layer into a text format that can be written to disk. The resulting string contains
     * all of the information needed to reconstruct the layer and it's parameters.
     */
    public String toString(){
        StringBuilder sb = new StringBuilder();

        int layerSize = biasMatrix.length;

        //Provide basic constructor information
        sb.append("DENSE(" + layerSize + ", " + activationFunction.toString() + ")\n");

        //Provide parameter information
        sb.append(Utility.arraysToString(this.getParameters()));

        return sb.toString();
    }
}