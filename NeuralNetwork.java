import java.util.ArrayList;


public class NeuralNetwork extends Model{

    private ArrayList<Input> inputLayers;

    private ArrayList<Layer> outputLayers;

    private ArrayList<Layer> allLayers;

    public NeuralNetwork(ArrayList<Input> inputLayers, ArrayList<Layer> outputLayers){

        this.inputLayers = inputLayers;

        this.outputLayers = outputLayers;

        allLayers = serializeLayers();
    }

    private ArrayList<Layer> serializeLayers(){
        //Use depth first traversal to grab all layers

        return null;
    }


    public void saveModel(String filePath){

    }
}