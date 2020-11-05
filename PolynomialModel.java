import java.util.ArrayList;

public class PolynomialModel extends Model {

    int numInputs;

    int numOutputs;

    int degree;

    ArrayList<float[][]> weightMatricies;

    float[][] biasVector;

    



    public ArrayList<float[]> predict(ArrayList<float[]> inputVectors) {
        // TODO Auto-generated method stub
        return null;
    }

    
    public void fit(ArrayList<float[][]> x, ArrayList<float[][]> y, int epochs, int minibatchSize, float valueClip,
            Optimizer opt, Loss loss) {
        // TODO Auto-generated method stub

    }

    
    public float calculateLoss(ArrayList<float[][]> x, ArrayList<float[][]> y, Loss loss) {
        // TODO Auto-generated method stub
        return 0;
    }

    
    public void saveModel(String filePath) {
        // TODO Auto-generated method stub

    }
    
}