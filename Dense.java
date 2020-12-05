import java.util.ArrayList;

public class Dense extends Layer {

    private ActivationFunction activationFunction;

    private float[][] weightMatrix;

    private float[][] biasMatrix;

    public Dense(int numUnits, ActivationFunction f, Layer inputLayer){
        activationFunction = f;

        ArrayList<Layer> inputLayers = new ArrayList<Layer>(1);
        inputLayers.add(inputLayer);
        setInputLayers(inputLayers);

        int inputLayerOutputSize = inputLayer.getOutputVector().length;

        inputVector = new float[inputLayerOutputSize];

        outputVector = new float[numUnits];


        
        weightMatrix = new float[numUnits][inputLayerOutputSize];

        biasMatrix = new float[numUnits][1];

        ArrayList<float[][]> params = new ArrayList<float[][]>(2);
        params.add(weightMatrix);
        params.add(biasMatrix);
    }

    public void forwardPass(){

    }

    public void backwardPass(){

    }
}