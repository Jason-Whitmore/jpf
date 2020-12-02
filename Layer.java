import java.util.ArrayList;

public abstract class Layer {

    protected ArrayList<float[][]> parameters;


    protected ArrayList<Layer> inputLayers;




    protected ArrayList<Layer> outputLayers;



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


    public ArrayList<Layer> getOutputLayers(){
        return outputLayers;
    }

    public void setOutputLayers(ArrayList<Layer> layers){
        outputLayers = layers;
    }

    
    public abstract void forwardPass();

    public abstract void backwardPass();

}