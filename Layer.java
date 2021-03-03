import java.util.ArrayList;

public abstract class Layer {

    protected ArrayList<float[][]> parameters;


    protected ArrayList<Layer> inputLayers;

    protected float[] inputVector;




    protected ArrayList<Layer> outputLayers;

    protected float[] outputVector;

    
    //Fields for storing information for backpropagation

    protected ArrayList<float[][]> gradient;

    //Gradient of the loss function wrt this layer's output
    protected float[] dLdY;

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


    public void setParameters(ArrayList<float[][]> params){
        parameters = params;
    }


    public ArrayList<Layer> getInputLayers(){
        return inputLayers;
    }

    public void setInputLayers(ArrayList<Layer> layers){
        inputLayers = layers;
    }


    public float[] getInputVector(){
        return inputVector;
    }

    public void setInputVector(float[] newInputVector){
        inputVector = newInputVector;
    }




    public ArrayList<Layer> getOutputLayers(){
        return outputLayers;
    }

    public void setOutputLayers(ArrayList<Layer> layers){
        outputLayers = layers;
    }




    public float[] getOutputVector(){
        return outputVector;
    }

    public void setOutputVector(float[] newOutputVector){
        outputVector = newOutputVector;
    }


    
    public ArrayList<float[][]> getGradient(){
        return gradient;
    }

    public float[] getdLdY(){
        return dLdY;
    }

    public void setdLdY(float[] newDerivative){
        dLdY = newDerivative;
    }

    public float[] getdLdX(){
        return dLdX;
    }

    public void setdLdX(float[] newDerivative){
        dLdX = newDerivative;
    }

    protected void initializedLdY(){
        Utility.clearArray(dLdY);

        for(int i = 0; i < getOutputLayers().size(); i++){
            for(int j = 0; j < dLdY.length; j++){
                dLdY[j] += getOutputLayers().get(i).getdLdX()[j];
            }
        }
    }

    protected void connectInputLayers(){
        for(int i = 0; i < inputLayers.size(); i++){
            inputLayers.get(i).getOutputLayers().add(this);
        }
    }

    protected void distributeOutputToNextLayers(){
        ArrayList<Layer> outputLayers = getOutputLayers();

        for(int i = 0; i < outputLayers.size(); i++){
            float[] nextInputVector = outputLayers.get(i).getInputVector();

            Utility.copyArrayContents(outputVector, nextInputVector);
        }
    }
    

    
    public abstract void forwardPass();

    public abstract void backwardPass();

}