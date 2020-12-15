import java.util.ArrayList;
import java.util.Arrays;

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
        for(int i = 0; i < inputLayers.size(); i++){
            inputLayers.get(i).getOutputLayers().add(this);
        }

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


    }

    public void forwardPass(){

        //Do matrix multiplication on the input vector: Wx

        float[][] wx = LinearAlgebra.matrixMultiply(weightMatrix, LinearAlgebra.arrayToMatrix(inputVector));

        //add bias
        wxPlusBias = LinearAlgebra.matrixAdd(wx, biasMatrix);

        //use as input to activation function.

        outputVector = activationFunction.f(LinearAlgebra.matrixToArray(wxPlusBias));

        //distribute the outputvector to the next layers

        ArrayList<Layer> outputLayers = getOutputLayers();

        for(int i = 0; i < outputLayers.size(); i++){
            float[] nextInputVector = outputLayers.get(i).getInputVector();

            Utility.copyArrayContents(outputVector, nextInputVector);
        }
    }

    public void backwardPass(){
        //Determine the error vector from the next layer
        float[] dLdO = getOutputLayers().get(0).getLayerError();

        //Determine the error vector of this layer's output wrt the sum
        float[] dOdS = LinearAlgebra.elementwiseMultiply(dLdO, activationFunction.fPrime(LinearAlgebra.matrixToArray(wxPlusBias)));


        //In a dense layer, the bias gradient is just the error vector
        float[] biasGradient = dOdS.clone();
        gradient.set(1, LinearAlgebra.arrayToMatrix(biasGradient));


        //The weight gradient is just the error multiplied by the input vector component
        float[][] weightGradient = new float[weightMatrix.length][weightMatrix[0].length];

        float[] inputVector = getInputVector();

        for(int r = 0; r < weightMatrix.length; r++){
            for(int c = 0; c < weightMatrix[0].length; c++){
                weightMatrix[r][c] = dOdS[r] * inputVector[c];
            }
        }

        gradient.set(0, weightGradient);
    }
}