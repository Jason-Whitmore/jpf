import java.util.ArrayList;
import java.util.Arrays;

public class Dense extends Layer {

    private ActivationFunction activationFunction;

    private float[][] weightMatrix;

    private float[][] biasMatrix;

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
        float[][] wxPlusBias = LinearAlgebra.matrixAdd(wx, biasMatrix);

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
        //Determine the error vector, dO/d(wx+b)
        float[] error = null;

        //In a dense layer, the bias gradient is just the error vector
        float[] biasGradient = error.clone();
        gradient.set(1, LinearAlgebra.arrayToMatrix(biasGradient));


        //The weight gradient is just the error multiplied by the input vector component
        float[][] weightGradient = new float[weightMatrix.length][weightMatrix[0].length];

        for(int r = 0; r < weightMatrix.length; r++){
            for(int c = 0; c < weightMatrix[0].length; c++){
            
            }
        }

        gradient.set(0, weightGradient);
    }
}