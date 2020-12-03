


public class Input extends Layer{

    private int inputVectorSize;

    public Input(int inputVectorSize){
        setInputVectorSize(inputVectorSize);
    }


    public int getInputVectorSize(){
        return inputVectorSize;
    }

    public void setInputVectorSize(int newInputVectorSize){
        this.inputVectorSize = newInputVectorSize;
    }

    public void forwardPass(){

        //copy data directly from input vector to output vector
        for(int j = 0; j < getInputVector().length; j++){
            getOutputVector()[j] = getInputVector()[j];
        }

        for(int i = 0; i < getOutputLayers().size(); i++){

            for(int j = 0; j < getOutputLayers().get(i).getInputVector().length; j++){
                getOutputLayers().get(i).getInputVector()[j] = getOutputVector()[j];
            }
            
        }
    }


    public void backwardPass(){
        return;
    }
}