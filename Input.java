
/**
 * Defines the Input layer class which is a concrete derived class of the Layer abstract class.
 * The Input layer is used as the first layers in a NeuralNetwork model.
 */
public class Input extends Layer{

    /**
     * Basic constructor for the Input layer object which is used as the first layers
     * in a NeuralNetwork model.
     * @param inputVectorSize The length of the input vector for this layer. Should be >= 1.
     */
    public Input(int inputVectorSize){
        super();

        this.inputVector = new float[inputVectorSize];
        this.outputVector = new float[inputVectorSize];
    }

    /**
     * Constructs an Input layer object from the string produced by the toString method of this class.
     * @param layerInfoString The layer info string produced by toString. The syntax of this string is simple:
     * The string "INPUT(5)" defines an Input layer whose input vector is of length 5.
     */
    public Input(String layerInfoString){
        super();

        String layerSizeString = layerInfoString.replace("INPUT(", "").replace(")", "");

        int layerSize = Integer.parseInt(layerSizeString);

        this.inputVector = new float[layerSize];
        this.outputVector = new float[layerSize];
    }



    public void forwardPass(){
        //copy data directly from input vector to output vector
        Utility.copyArrayContents(this.inputVector, this.outputVector);

        for(int i = 0; i < this.outputLayers.size(); i++){
            float[] nextInputVector = this.outputLayers.get(i).getInputVector();

            Utility.copyArrayContents(this.outputVector, nextInputVector);
        }
    }


    public void backwardPass(){
        //No work required since there is no parameters
        return;
    }

    @Override
    public String toString(){
        //Only need to provide the length of the input vector.
        return "INPUT(" + inputVector.length + ")";
    }
}