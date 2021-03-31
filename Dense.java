import java.util.ArrayList;
//import java.util.Arrays;

public class Dense extends Layer {

    private ActivationFunction activationFunction;

    private float[][] weightMatrix;

    private float[][] biasMatrix;



    //for backward prop purposes
    private float[][] wxPlusBias;

    public Dense(int numUnits, ActivationFunction f, Layer inputLayer){
        super();

        activationFunction = f;

        ArrayList<Layer> inputLayers = new ArrayList<Layer>(1);
        inputLayers.add(inputLayer);
        setInputLayers(inputLayers);

        //connect input layer's output to this this layer
        connectInputLayers();

        int inputLayerOutputSize = inputLayer.getOutputVector().length;

        inputVector = new float[inputLayerOutputSize];
        outputVector = new float[numUnits];



        weightMatrix = new float[numUnits][inputLayerOutputSize];
        Utility.Initializers.initializeUniform(weightMatrix, -0.1f, 0.1f);

        biasMatrix = new float[numUnits][1];

        ArrayList<float[][]> params = new ArrayList<float[][]>(2);
        params.add(weightMatrix);
        params.add(biasMatrix);

        setParameters(params);

        //set up gradients
        gradient = Utility.cloneArrays(getParameters());
        Utility.clearArrays(gradient);

        //Set up backprop arrays
        dLdX = new float[inputLayerOutputSize];
        dLdY = new float[numUnits];
    }

    public void forwardPass(){

        //Bring over the previous layer's output into this layer's input vector
        Utility.copyArrayContents(getInputLayers().get(0).getOutputVector(), getInputVector());

        //Do matrix multiplication on the input vector: Wx

        float[][] wx = LinearAlgebra.matrixMultiply(weightMatrix, LinearAlgebra.arrayToMatrix(inputVector));

        //add bias
        wxPlusBias = LinearAlgebra.matrixAdd(wx, biasMatrix);

        //use as input to activation function.

        outputVector = activationFunction.f(LinearAlgebra.matrixToArray(wxPlusBias));

        //distribute the outputvector to the next layers
    }

    public void backwardPass(){
        //Determine the error vector from the next layers
        initializedLdY();

        

        //Determine the error vector of this layer's output wrt the sum
        float[] dLdS = LinearAlgebra.elementwiseMultiply(dLdY, activationFunction.fPrime(LinearAlgebra.matrixToArray(wxPlusBias)));


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

        sb.append(")");

        return sb.toString();
    }
}