


public class Input extends Layer{

    public Input(int inputVectorSize){
        super();

        inputVector = new float[inputVectorSize];

        outputVector = new float[inputVectorSize];
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