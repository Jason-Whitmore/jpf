import java.util.ArrayList;

public class RMSProp implements Optimizer{

    private float learningRate;

    private float rho;

    private float epsilon;

    ArrayList<float[][]> prevGradSquare;

    public RMSProp(){
        learningRate = 0.0001f;

        rho = 0.9f;

        epsilon = 0.0000001f;

        prevGradSquare = null;
    }

    public RMSProp(float learningRate, float rho, float epsilon){
        this.learningRate = learningRate;

        this.rho = rho;

        this.epsilon = epsilon;

        prevGradSquare = null;
    }


    public ArrayList<float[][]> processGradient(ArrayList<float[][]> rawGradient){
        return null;
    }

    public void resetState(){
        if(prevGradSquare == null){
            return;
        }

        ArrayList<float[][]> newState = Utility.cloneArrays(prevGradSquare);

        prevGradSquare = newState;
    }
}