import java.util.Arrays;

public class Input extends Layer{

    public Input(int inputVectorSize){
        super();

        inputVector = new float[inputVectorSize];

        outputVector = new float[inputVectorSize];
    }



    public void forwardPass(){
        //copy data directly from input vector to output vector
        Utility.copyArrayContents(getInputVector(), getOutputVector());

        for(int i = 0; i < outputLayers.size(); i++){
            float[] nextInputVector = outputLayers.get(i).getInputVector();

            Utility.copyArrayContents(outputVector, nextInputVector);
        }
    }


    public void backwardPass(){
        //No work required since there is no parameters
        return;
    }

    @Override
    public String toString(){
        //Only need to provide the length of the input vector.
        return "INPUT(" + inputVector.length + ")";
    }
}