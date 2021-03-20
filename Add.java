import java.util.ArrayList;


class Add extends Layer{

    private int layerSize;


    public Add(ArrayList<Layer> inputLayers){
        super();

        //Confirm that the size of all input and output layers have vectors of equal length
        layerSize = inputLayers.get(0).getOutputVector().length;

        for(int i = 1; i < inputLayers.size(); i++){
            if(inputLayers.get(i).getOutputVector().length != layerSize){
                //TODO: Fatal error here
            }
        }


        this.inputLayers = inputLayers;

        //TODO: Add weight parameters? More research may be required.
        this.parameters = new ArrayList<float[][]>();

        this.inputVector = new float[layerSize];
        this.outputVector = new float[layerSize];

        this.dLdX = new float[layerSize];
        this.dLdY = new float[layerSize];

        connectInputLayers();
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
        initializedLdY();

        for(int i = 0; i < getdLdX().length; i++){
            getdLdX()[i] = getdLdY()[i];
        }
    }

    public String toString(){
        return "Add(" + this.layerSize + ")";
    }
}