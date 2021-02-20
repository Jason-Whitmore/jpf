import java.util.ArrayList;


class Add extends Layer{

    private int layerSize;

    public Add(ArrayList<Layer> inputLayers, ArrayList<Layer> outputLayers){
        super();

        //Confirm that the size of all input and output layers have vectors of equal length
        layerSize = inputLayers.get(0).getOutputVector().length;

        for(int i = 1; i < inputLayers.size(); i++){
            if(inputLayers.get(i).getOutputVector().length != layerSize){
                //TODO: Fatal error here
            }
        }

        for(int i = 0; i < outputLayers.size(); i++){
            if(outputLayers.get(i).getInputVector().length != layerSize){
                //TODO: Fatal error here
            }
        }

        this.inputLayers = inputLayers;
        this.outputLayers = outputLayers;
        this.parameters = null;

        this.inputVector = new float[layerSize];
        this.outputVector = new float[layerSize];

        this.dLdX = new float[layerSize];
        this.dLdY = new float[layerSize];
    }


    public void forwardPass(){
        Utility.clearArray(this.inputVector);
        Utility.clearArray(this.outputVector);

        for(int j = 0; j < inputLayers.size(); j++){
            for(int i = 0; i < layerSize; i++){
                this.inputVector[i] += inputLayers.get(j).getOutputVector()[i];
                this.outputVector[i] += this.inputVector[i];
            }
        }
    }


    public void backwardPass(){

    }
}