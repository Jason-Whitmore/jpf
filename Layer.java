import java.util.ArrayList;

public abstract class Layer {

    protected ArrayList<float[][]> parameters;


    protected ArrayList<Layer> inputLayers;

    protected float[] inputVector;


    protected ArrayList<Layer> outputLayers;

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

    public Layer(){
        parameters = new ArrayList<float[][]>();
        inputLayers = new ArrayList<Layer>();
        outputLayers = new ArrayList<Layer>();
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
        Utility.clearArray(this.dLdY);

        for(int i = 0; i < getOutputLayers().size(); i++){
            for(int j = 0; j < dLdY.length; j++){
                dLdY[j] += getOutputLayers().get(i).getdLdX()[j];
            }
        }
    }

    /**
     * Populates the input vector so the forward pass can be performed.
     */
    public void initializeInputVector(){
        Utility.clearArray(this.inputVector);

        for(int i = 0; i < inputLayers.size(); i++){
            for(int j = 0; j < this.inputVector.length; j++){
                this.inputVector[j] += inputLayers.get(i).outputVector[j];
            }
        }
    }

    protected void connectInputAndOutputLayers(){
        for(int i = 0; i < inputLayers.size(); i++){
            inputLayers.get(i).getOutputLayers().add(this);
        }

        for(int i = 0; i < outputLayers.size(); i++){
            outputLayers.get(i).getInputLayers().add(this);
        }
    }

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

    public void clearForwardPassArrays(){
        if(this.inputVector != null){
            Utility.clearArray(this.inputVector);
        }

        if(this.outputVector != null){
            Utility.clearArray(this.outputVector);
        }
    }

    
    public abstract void forwardPass();

    public abstract void backwardPass();


    public static Layer createLayerFromString(String layerInfoString){

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