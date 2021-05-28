import java.util.ArrayList;

public class PolynomialModel extends SimpleModel{

    private int degree;

    private ArrayList<float[][]> weightMatricies;

    private float[][] biasVector;

    
    public PolynomialModel(int numInputs, int numOutputs, int degree){
        super(numInputs, numOutputs);

        if(degree < 1){
            System.err.println("Error: Using a polynomial model with a degree of less than 1 may cause issues.");
        }

        this.degree = degree;

        weightMatricies = new ArrayList<float[][]>();

        //Create the polynomial weight matricies and initialize
        for(int i = 0; i < this.numInputs; i++){
            weightMatricies.add(new float[this.numOutputs][this.degree]);
            this.parameters.add(this.weightMatricies.get(i));
        }

        this.biasVector = new float[this.numOutputs][1];

        this.parameters.add(this.biasVector);

        Utility.initializeUniform(getParameters(), -0.001f, 0.001f);
    }


    public PolynomialModel(String filePath){
        super();
        //Load file contents
        String contents = Utility.getTextFileContents(filePath);
        contents = contents.replace("POLYNOMIALMODEL\n", "");

        ArrayList<float[][]> arrays = Utility.stringToMatrixList(contents);

        this.parameters = arrays;

        weightMatricies = new ArrayList<float[][]>();
        for(int i = 0; i < arrays.size() - 1; i++){
            weightMatricies.add(arrays.get(i));
        }

        biasVector = arrays.get(arrays.size() - 1);

        this.numInputs = arrays.size() - 1;

        this.numOutputs = LinearAlgebra.getNumRows(biasVector);

        this.degree = LinearAlgebra.getNumColumns(arrays.get(0));

    }

    public int getDegree(){
        return this.degree;
    }


    public float[] predict(float[] inputVector){
        //TODO: Check input dimensions

        float[] outputVector = new float[this.numOutputs];

        for(int x = 0; x < inputVector.length; x++){
            float[] powers = this.calculatePowers(inputVector[x], degree);
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
     * @param maxPower The highest exponent in the sequence. Should be >= 1.
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
    

    public ArrayList<float[][]> calculateGradient(float[] x, float[] y, Loss loss){
        ArrayList<float[][]> grad = Utility.cloneArrays(getParameters());
        Utility.clearArrays(grad);

        float[] yPred = this.predict(x);

        float[] dLdY = loss.calculateLossVectorGradient(y, yPred);

        for(int i = 0; i < x.length; i++){
            float[] powers = this.calculatePowers(x[i], this.degree);

            for(int j = 0; j < y.length; j++){

                for(int d = 0; d < degree; d++){
                    grad.get(i)[j][d] = powers[d] * dLdY[j];
                }
            }
        }

        grad.set(grad.size() - 1, LinearAlgebra.arrayToMatrix(dLdY));


        return grad;
    }


    public float calculateLoss(float[] x, float[] y, Loss loss){
        float[] yPred = predict(x);

        return loss.calculateLossScalar(y, yPred);
    }

    
    public void saveModel(String filePath) {
        //create the model string
        String contents = Utility.arraysToString(this.getParameters());
        contents = "POLYNOMIALMODEL\n" + contents;

        boolean success = Utility.writeStringToFile(filePath, contents);

        if(!success){
            System.err.println("Error: Polynomial model could not be saved.");
        }
    }
    
}