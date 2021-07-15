import java.util.ArrayList;

/**
 * Defines the abstract Layer class which provides the basic functionality
 * and interface for all concrete derived classes that are used in the
 * NeuralNetwork class.
 */
public abstract class Layer {

    /**
     * The parameters of the layer represented as an ArrayList of float matricies.
     */
    protected ArrayList<float[][]> parameters;

    /**
     * The input layers that connect with this layer. Outputs of these layers
     * are the inputs of this layer.
     */
    protected ArrayList<Layer> inputLayers;

    /**
     * The input vector for this layer, represented as an array of floating points.
     */
    protected float[] inputVector;

    /**
     * The output layers that connect with this layer. The output of this layer
     * become inputs to these layers.
     */
    protected ArrayList<Layer> outputLayers;

    /**
     * The output vector for this layer, represented as an array of floating points.
     */
    protected float[] outputVector;


    /**
     * The gradient of the loss function wrt the layer's parameters.
     */
    protected ArrayList<float[][]> gradient;

    /**
     * Gradient of the loss function wrt this layer's output vector
     */
    protected float[] dLdY;

    /**
     * Gradient of the loss function wrt this layer's input vector
     */
    protected float[] dLdX;


    /**
     * Basic constructor that initializes the lists of the method. Other fields are not initialized
     * because they need more information, such as the number of units in the layer.
     */
    public Layer(){
        this.parameters = new ArrayList<float[][]>();
        this.inputLayers = new ArrayList<Layer>();
        this.outputLayers = new ArrayList<Layer>();
    }

    //getters and setters for class fields

    public ArrayList<float[][]> getParameters(){
        return parameters;
    }

    public ArrayList<Layer> getInputLayers(){
        return inputLayers;
    }

    public float[] getInputVector(){
        return inputVector;
    }

    public ArrayList<Layer> getOutputLayers(){
        return outputLayers;
    }

    public float[] getOutputVector(){
        return outputVector;
    }

    public ArrayList<float[][]> getGradient(){
        return gradient;
    }

    public float[] getdLdY(){
        return dLdY;
    }


    public float[] getdLdX(){
        return dLdX;
    }

    /**
     * Populates the dLdY vector with the sum of dLdX vector from this layer's output layers
     */
    protected void initializedLdY(){
        for(int i = 0; i < this.getOutputLayers().size(); i++){
            for(int j = 0; j < dLdY.length; j++){
                this.dLdY[j] += getOutputLayers().get(i).getdLdX()[j];
            }
        }
    }

    /**
     * Populates the input vector by copying the output vector contents of the previous layer
     * into the input vector of this layer. Should only be used for Layers that only have 1
     * input layer.
     */
    protected void initializeInputVectorCopy(){
        
        if(this.inputLayers.size() == 1){
            Utility.copyArrayContents(this.inputLayers.get(0).outputVector, this.inputVector);
        }

    }

    /**
     * Connects this layer to the input and output layers so that the forward and backward pass
     * methods can work properly.
     */
    protected void connectInputAndOutputLayers(){
        for(int i = 0; i < this.inputLayers.size(); i++){
            this.inputLayers.get(i).getOutputLayers().add(this);
        }

        for(int i = 0; i < this.outputLayers.size(); i++){
            this.outputLayers.get(i).getInputLayers().add(this);
        }
    }

    /**
     * Clears the backpropagation arrays so that they can be used to find new gradients.
     */
    public void clearBackpropArrays(){
        if(this.dLdX != null){
            Utility.clearArray(this.dLdX);
        }
        
        if(this.dLdY != null){
            Utility.clearArray(this.dLdY);
        }

        if(this.gradient != null){
            Utility.clearArrays(this.gradient);
        }
    }

    /**
     * Clears the forward pass arrays so that they can be used for predictions and backpropagation.
     */
    public void clearForwardPassArrays(){
        if(this.inputVector != null){
            Utility.clearArray(this.inputVector);
        }

        if(this.outputVector != null){
            Utility.clearArray(this.outputVector);
        }
    }

    /**
     * Performs the forward pass, which is used to make predictions and set up arrays for a possible
     * backpropagation pass.
     * 
     * When implementing this method, the layer should "pull" the input vector from the layer's input layers,
     * perform some computation, then place the result in the layer's output vector.
     */
    public abstract void forwardPass();

    /**
     * Performs the backward pass, which is used to calculate the gradient of the loss function with
     * respect to the layer parameters for the last input calculated using forwardPass().
     * 
     * When implementing this method, the layer should first initialize the dLdY vector from the layer's
     * output layers, populate the gradient list, then populate the dLdX vector.
     */
    public abstract void backwardPass();


    /**
     * Creates a Layer solely from the layerInfoString that is produced from a layer's toString method.
     * When creating new layers, the if/else chain should be expanded to include a way to construct the
     * new layer.
     * 
     * @param layerInfoString The string produced from a layer's toString method and read from a model's
     * saved-to-disk form.
     * 
     * @return The constructed layer with the correct parameters and properties. Returns null if unsuccessful.
     */
    public static Layer createLayerFromString(String layerInfoString){
        Utility.checkNotNull(layerInfoString);
        System.out.println(layerInfoString);

        if(layerInfoString.contains("INPUT")){
            return new Input(layerInfoString);
        } else if(layerInfoString.contains("DENSE")){
            return new Dense(layerInfoString);
        } else if(layerInfoString.contains("ADD")){
            return new Add(layerInfoString);
        } else if(layerInfoString.contains("SOFTMAX")){
            return new SoftmaxLayer(layerInfoString);
        }

        return null;
    }

}