import java.util.ArrayList;

/**
 * Defines the Add layer class, which is a concrete class derived from the Layer abstract class.
 * This layer is used to combine the output vector of several layers into a single output vector.
 */
public class Add extends Layer{

    /**
     * The number of units in the layer, which should be the
     * same size as all of the input layer's output vector.
     */
    private int numUnits;

    /**
     * The main constructor for the Add layer, which allows output vectors from multiple layers to be combined
     * into a single vector by performing elementwise addition. This layer contains no parameters.
     * @param inputLayers The list of layer's whose output vectors will be added together to create this layer's output vector.
     *  All of the output vectors for these layers must be of the same size.
     */
    public Add(ArrayList<Layer> inputLayers){
        super();

        //Check parameter
        Utility.checkNotNull(inputLayers);
        if(inputLayers.size() <= 0){
            throw new AssertionError("inputLayers list is empty.");
        }

        //Confirm that the size of all input and output layers have vectors of equal length
        Utility.checkNotNull(inputLayers.get(0), inputLayers.get(0).outputVector);
        this.numUnits = inputLayers.get(0).getOutputVector().length;

        for(int i = 1; i < inputLayers.size(); i++){
            Utility.checkNotNull(inputLayers.get(i), inputLayers.get(i).outputVector);
            Utility.checkEqual(this.numUnits, inputLayers.get(i).outputVector.length);
        }

        //Initialize the fields.
        this.inputLayers = inputLayers;

        this.inputVector = new float[this.numUnits];
        this.outputVector = new float[this.numUnits];

        this.dLdX = new float[this.numUnits];
        this.dLdY = new float[this.numUnits];

        this.connectInputAndOutputLayers();
    }

    /**
     * Constructs add layer from string. String should be of form "Add(n)" where n is the layer size.
     * @param layerInfoString The layer information string that is used to construct the layer.
     */
    public Add(String layerInfoString){
        super();

        String layerSizeString = layerInfoString.replace("ADD(", "").replace(")", "");

        this.numUnits = Integer.parseInt(layerSizeString);

        //Check to see if numunits makes sense
        if(this.numUnits <= 0){
            throw new AssertionError("numUnits should be >= 1");
        }

        this.inputVector = new float[this.numUnits];
        this.outputVector = new float[this.numUnits];

        this.dLdX = new float[this.numUnits];
        this.dLdY = new float[this.numUnits];

    }


    public void forwardPass(){
        this.clearForwardPassArrays();

        for(int j = 0; j < inputLayers.size(); j++){
            for(int i = 0; i < this.numUnits; i++){
                this.inputVector[i] += inputLayers.get(j).getOutputVector()[i];
            }
        }

        for(int i = 0; i < this.outputVector.length; i++){
            this.outputVector[i] = this.inputVector[i];
        }
    }


    public void backwardPass(){
        initializedLdY();

        for(int i = 0; i < getdLdX().length; i++){
            getdLdX()[i] = getdLdY()[i];
        }
    }

    public String toString(){
        return "ADD(" + this.numUnits + ")";
    }
}