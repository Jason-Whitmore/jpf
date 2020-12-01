import java.util.ArrayList;

public abstract class Layer {

    protected ArrayList<float[][]> parameters;


    protected ArrayList<Layer> inputLayers;

    protected ArrayList<float[]> inputVectors;



    protected ArrayList<Layer> outputLayers;

    protected ArrayList<float[]> outputVectors;


    
    public abstract void forwardPass();


    public abstract void backwardPass();

}