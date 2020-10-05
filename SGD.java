import java.util.ArrayList;

public class SGD implements Optimizer{

    private float learningRate;

    public SGD(){
        learningRate = 0.0001f;
    }

    public SGD(float learningRate){
        this.learningRate = learningRate;
    }


    @Override
    public ArrayList<float[][]> processGradient(ArrayList<float[][]> rawGradient){
        Utility.scaleGradient(rawGradient, this.learningRate);

        return rawGradient;
    }
}