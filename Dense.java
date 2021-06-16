public class Dense extends Layer {

    private ActivationFunction activationFunction;

    private float[][] weightMatrix;

    private float[][] biasMatrix;



    //for backward prop purposes
    private float[][] sum;

    public Dense(int numUnits, ActivationFunction f, Layer inputLayer){
        super();

        this.activationFunction = f;

        this.inputLayers.add(inputLayer);

        //connect input layer's output to this layer
        this.connectInputAndOutputLayers();

        int inputLayerOutputSize = inputLayer.getOutputVector().length;

        this.inputVector = new float[inputLayerOutputSize];
        this.outputVector = new float[numUnits];



        this.weightMatrix = new float[numUnits][inputLayerOutputSize];
        Utility.initializeUniform(weightMatrix, -0.1f, 0.1f);

        this.biasMatrix = new float[numUnits][1];

        //Add matricies to parameter list
        this.parameters.add(weightMatrix);
        this.parameters.add(biasMatrix);

        //set up gradients
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

        String paramString = layerInfoString.substring(layerInfoString.indexOf("[\n"), layerInfoString.indexOf("]\n") + 2);
        this.parameters = Utility.stringToMatrixList(paramString);

        //From the parsed parameters, get the layer size and the input vector size
        int numUnits = this.biasMatrix.length;
        int inputSize = this.weightMatrix[0].length;

        //Allocate and initialize vectors
        this.inputVector = new float[inputSize];
        this.outputVector = new float[numUnits];

        this.dLdX = new float[inputSize];
        this.dLdY = new float[numUnits];

        //Allocate and initialize gradients
        this.gradient = Utility.cloneArrays(getParameters());
        Utility.clearArrays(this.gradient);

        //Isolate the activation function string and initialize
        this.initializeActivationFunctionFromString(layerInfoString);
    }

    private void initializeActivationFunctionFromString(String layerInfoString){
        String headerInfo = layerInfoString.substring(0, layerInfoString.indexOf(")") + 1);
        headerInfo.replace("(", "");
        headerInfo.replace(")", "");

        String[] headerInfoSplit = headerInfo.split(",");
        this.activationFunction = ActivationFunction.constructFromString(headerInfoSplit[1]);
    }

    public void forwardPass(){

        //Bring over the previous layer's output into this layer's input vector
        this.initializeInputVectorCopy();

        //Do matrix multiplication on the input vector: Wx
        float[][] wx = LinearAlgebra.matrixMultiply(weightMatrix, LinearAlgebra.arrayToMatrix(inputVector));

        //add bias
        sum = LinearAlgebra.matrixAdd(wx, biasMatrix);

        //use as input to activation function.
        outputVector = activationFunction.f(LinearAlgebra.matrixToArray(sum));
    }

    public void backwardPass(){
        //Determine the error vector from the next layers
        initializedLdY();

        //Determine the error vector of this layer's output wrt the sum
        float[] dLdS = LinearAlgebra.elementwiseMultiply(dLdY, activationFunction.fPrime(LinearAlgebra.matrixToArray(sum)));

        //In a dense layer, the bias gradient is just the error vector
        float[] biasGradient = dLdS.clone();
        gradient.set(1, LinearAlgebra.arrayToMatrix(biasGradient));


        //The weight gradient is just the error multiplied by the input vector component
        float[][] weightGradient = new float[weightMatrix.length][weightMatrix[0].length];

        float[] inputVector = getInputVector();

        for(int r = 0; r < weightMatrix.length; r++){
            for(int c = 0; c < weightMatrix[0].length; c++){
                weightGradient[r][c] = dLdS[r] * inputVector[c];
            }
        }

        gradient.set(0, weightGradient);


        //determine dLdX

        for(int i = 0; i < inputVector.length; i++){
            for(int j = 0; j < outputVector.length; j++){
                getdLdX()[i] += weightMatrix[j][i] * dLdS[j];
            }
        }
    }


    public String toString(){
        StringBuilder sb = new StringBuilder();

        int layerSize = biasMatrix.length;

        //Provide basic constructor information
        sb.append("DENSE(" + layerSize + ", " + activationFunction.toString() + ")\n");

        //Provide weight information:
        sb.append(Utility.arraysToString(getParameters()));

        return sb.toString();
    }
}