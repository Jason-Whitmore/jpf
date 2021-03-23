import java.util.ArrayList;
import java.util.Collections;

public class Concatenate extends Layer{

    private int size;

    public Concatenate(ArrayList<Layer> inputLayers){
        super();

        this.inputLayers = inputLayers;

        updateLayerSize();
        outputVector = new float[this.size];
    }

    public Concatenate(Layer[] inputLayers){
        super();

        ArrayList<Layer> inputLayersList = new ArrayList<Layer>(inputLayers.length);

        Collections.addAll(inputLayersList, inputLayers);
        this.inputLayers = inputLayersList;

        updateLayerSize();

        outputVector = new float[this.size];
    }

    public void forwardPass(){

        int outputIndex = 0;

        for(int i = 0; i < inputLayers.size(); i++){
            for(int j = 0; j < inputLayers.get(i).inputVector.length; j++){
                outputVector[outputIndex] = inputLayers.get(i).inputVector[j];
                outputIndex++;
            }
        }
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