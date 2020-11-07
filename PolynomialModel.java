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


    public float[] predict(float[] inputVector){
        //TODO: Check input dimensions

        float[] outputVector = new float[numOutputs];

        for(int x = 0; x < inputVector.length; x++){
            float[] powers = calculatePowers(inputVector[x], degree);


            for(int y = 0; y < numOutputs; y++){

                for(int d = 0; d < powers.length; d++){
                    outputVector[y] += weightMatricies.get(x)[y][d] * powers[d];
                }
            }
        }

        //Add bias matrix
        for(int y = 0; y < numOutputs; y++){
            outputVector[y] += biasVector[y][0];
        }

        return outputVector;
    }

    /**
     * Calculates an array of form [x, x^2, ..., x^maxPower]
     * @param x The number to raise to the power of.
     * @param maxPower The highest exponent in the sequence.
     * @return Returns the power array.
     */
    private float[] calculatePowers(float x, int maxPower){
        float[] r = new float[maxPower];
        r[0] = x;

        for(int i = 1; i < r.length; i++){
            r[i] = x * r[i - 1];
        }

        return r;
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