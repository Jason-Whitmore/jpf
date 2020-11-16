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
        if(prevGradSquare == null){
            ArrayList<float[][]> newState = Utility.cloneArrays(rawGradient);
            Utility.clearArrays(newState);
        }

        ArrayList<float[][]> grad = Utility.cloneArrays(rawGradient);
        Utility.clearArrays(grad);

        
    }

    /**
     * Resets the optimizer state. Useful when retraining a model on new data.
     */
    public void resetState(){
        if(prevGradSquare == null){
            return;
        }

        ArrayList<float[][]> newState = Utility.cloneArrays(prevGradSquare);
        Utility.clearArrays(newState);

        prevGradSquare = newState;
    }
}