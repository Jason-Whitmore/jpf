import java.util.ArrayList;

public class PolynomialModel extends Model {

    int numInputs;

    int numOutputs;

    int degree;

    ArrayList<float[][]> weightMatricies;

    float[][] biasVector;

    
    public PolynomialModel(int numInputs, int numOutputs, int degree){
        this.numInputs = numInputs;

        this.numOutputs = numOutputs;

        this.degree = degree;

        //Create the polynomial weight matricies and initialize
        for(int i = 0; i < numOutputs; i++){
            weightMatricies.add(new float[numOutputs][degree]);
        }

        ArrayList<float[][]> params = new ArrayList<float[][]>();

        params.addAll(weightMatricies);



        this.biasVector = new float[numOutputs][1];

        params.add(this.biasVector);

        setParameters(params);


        Utility.Initializers.initializeUniform(getParameters(), -1f, 1f);
        
        
    }


    public ArrayList<float[]> predict(ArrayList<float[]> inputVectors) {
        // TODO Auto-generated method stub
        return null;
    }

    
    public void fit(ArrayList<float[][]> x, ArrayList<float[][]> y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss) {
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