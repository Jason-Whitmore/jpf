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
    protected float[] dLdO;

    protected float[] dOdI;

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

    public float[] getdLdO(){
        return dLdO;
    }

    public void setdLdO(float[] newDerivative){
        dLdO = newDerivative;
    }

    public float[] getdOdI(){
        return dOdI;
    }

    public void setdOdI(float[] newDerivative){
        dOdI = newDerivative;
    }
    

    
    public abstract void forwardPass();

    public abstract void backwardPass();

}