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

    /**
     * Makes multiple predictions.
     * @param inputVectors The set of input vectors to feed into the model.
     * @return The predictions corresponding with the input vectors.
     */
    public float[][] predict(float[][] inputVectors){
        float[][] r = new float[inputVectors.length][numOutputs];

        for(int i = 0; i < r.length; i++){
            float[] temp = predict(inputVectors[i]);

            r[i] = temp;
        }

        return r;
    }

    /**
     * Makes multiple predictions.
     * @param inputVectors The set of input vectors to feed into the model.
     * @return The predictions corresponding with the input vectors.
     */
    public ArrayList<float[]> predict(ArrayList<float[]> inputVectors) {
        ArrayList<float[]> r = new ArrayList<float[]>();

        for(int i = 0; i < inputVectors.size(); i++){
            float[] temp = predict(inputVectors.get(i));

            r.add(temp);
        }

        return r;
    }

    
    public void fit(ArrayList<float[][]> x, ArrayList<float[][]> y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss) {
        if(x.size() != 1 && y.size() != 1){
            return;
        }

        fit(x.get(0), y.get(0), epochs, minibatchSize, valueClip, opt, loss);
    }

    public ArrayList<float[][]> calculateGradient(float[] x, float[] y, Loss loss){
        ArrayList<float[][]> grad = Utility.cloneArrays(getParameters());

        float[] yPred = predict(x);
        float[] error = loss.calculateLossVectorGradient(y, yPred);

        for(int i = 0; i < x.length; i++){
            float[] powers = calculatePowers(x[i], degree);

            for(int j = 0; j < y.length; j++){

                for(int d = 0; d < degree; d++){
                    grad.get(i)[j][d] = powers[d] * error[j];
                }
            }
        }

        return grad;
    }

    public void fit(float[][] x, float[][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){

    }

    
    public float calculateLoss(ArrayList<float[][]> x, ArrayList<float[][]> y, Loss loss) {
        if(x.size() != 1 || y.size() != 1){
            return Float.NaN;
        }

        return calculateLoss(x.get(0), y.get(0), loss);
    }

    public float calculateLoss(float[][] x, float[][] y, Loss loss){
        float sum = 0;

        for(int i = 0; i < x.length; i++){
            sum += calculateLoss(x[i], y[i], loss);
        }

        return sum / x.length;
    }

    public float calculateLoss(float[] x, float[] y, Loss loss){
        float[] yPred = predict(x);

        return loss.calculateLossScalar(y, yPred);
    }

    
    public void saveModel(String filePath) {
        // TODO Auto-generated method stub

    }
    
}