import java.util.ArrayList;
import java.util.Collections;

public class Concatenate extends Layer{

    private int size;

    private ArrayList<Layer> inputLayers;

    public Concatenate(ArrayList<Layer> inputLayers){
        this.inputLayers = inputLayers;
        updateLayerSize();
    }

    public Concatenate(Layer[] inputLayers){
        ArrayList<Layer> inputLayersList = new ArrayList<Layer>(inputLayers.length);

        Collections.addAll(inputLayersList, inputLayers);

        this.inputLayers = inputLayersList;
        updateLayerSize();
    }

    public void forwardPass(){

    }

    public void backwardPass(){
        
    }




    private void updateLayerSize(){
        int inputLayerSizeSum = 0;

        for(int i = 0; i < inputLayers.size(); i++){
            inputLayerSizeSum += inputLayers.get(i).outputVector.length;
        }

        this.size = inputLayerSizeSum;
    }

}