


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
        return;
    }


    public void backwardPass(){
        return;
    }
}