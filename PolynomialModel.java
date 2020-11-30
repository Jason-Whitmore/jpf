import java.util.ArrayList;

public class PolynomialModel extends SimpleModel {

    int numInputs;

    int numOutputs;

    int degree;

    ArrayList<float[][]> weightMatricies;

    float[][] biasVector;

    
    public PolynomialModel(int numInputs, int numOutputs, int degree){
        this.numInputs = numInputs;

        this.numOutputs = numOutputs;

        this.degree = degree;

        weightMatricies = new ArrayList<float[][]>();

        ArrayList<float[][]> params = new ArrayList<float[][]>();

        //Create the polynomial weight matricies and initialize
        for(int i = 0; i < numOutputs; i++){
            weightMatricies.add(new float[numOutputs][degree]);
            params.add(weightMatricies.get(i));
        }




        this.biasVector = new float[numOutputs][1];

        params.add(this.biasVector);

        setParameters(params);


        Utility.Initializers.initializeUniform(getParameters(), -1f, 1f);
        
        
    }


    public PolynomialModel(String filePath){
        //Load file contents
        String contents = Utility.getTextFileContents(filePath);

        if(contents == null){
            return;
        }

        //Remove extranous brackets in the string

        ArrayList<float[][]> arrays = Utility.initializeMatrixListFromString(contents);

        setParameters(arrays);

        weightMatricies = new ArrayList<float[][]>();
        for(int i = 0; i < arrays.size() - 1; i++){
            weightMatricies.add(arrays.get(i));
        }

        biasVector = arrays.get(arrays.size() - 1);

        this.numInputs = arrays.size() - 1;

        this.numOutputs = LinearAlgebra.getNumRows(biasVector);

        this.degree = LinearAlgebra.getNumColumns(arrays.get(0));
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

    

    public ArrayList<float[][]> calculateGradient(float[] x, float[] y, Loss loss){
        ArrayList<float[][]> grad = Utility.cloneArrays(getParameters());
        Utility.clearArrays(grad);

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

        grad.set(grad.size() - 1, LinearAlgebra.arrayToMatrix(error));


        return grad;
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
        //TODO: Check if file path is valid

        String contents = Utility.arraysToString(getParameters());

        boolean success = Utility.writeStringToFile(filePath, contents);

        if(!success){
            
        }
    }
    
}