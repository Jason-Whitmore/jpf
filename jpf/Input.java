package jpf;

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

        //Check units
        if(inputVectorSize <= 0){
            throw new AssertionError("inputVectorSize should be >= 1");
        }

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

        int numUnits = Integer.parseInt(layerSizeString);

        if(numUnits <= 0){
            throw new AssertionError("numUnits should be >= 1");
        }

        this.inputVector = new float[numUnits];
        this.outputVector = new float[numUnits];
    }



    public void forwardPass(){
        //Since this layer only serves as a placeholder for the model's input vectors, the data simply needs to be copied over to the output.
        Utility.copyArrayContents(this.inputVector, this.outputVector);
    }


    public void backwardPass(){
        //No work required since there is no parameters or computations done.
        return;
    }

    @Override
    public String toString(){
        //Only need to provide the length of the input vector.
        return "INPUT(" + inputVector.length + ")";
    }

}