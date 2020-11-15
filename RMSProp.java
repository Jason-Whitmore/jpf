import java.util.ArrayList;

public class RMSProp implements Optimizer{

    private float learningRate;

    private float rho;

    private float epsilon;

    ArrayList<float[][]> state;

    public RMSProp(){
        learningRate = 0.0001f;

        rho = 0.9f;

        epsilon = 0.0000001f;
    }

    public RMSProp(float learningRate, float rho, float epsilon){
        this.learningRate = learningRate;

        this.rho = rho;

        this.epsilon = epsilon;
    }


    public ArrayList<float[][]> processGradient(ArrayList<float[][]> rawGradient){
        return null;
    }
}