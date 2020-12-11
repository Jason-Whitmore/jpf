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

    }
}