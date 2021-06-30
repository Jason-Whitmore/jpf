import java.util.ArrayList;

/**
 * Defines the PolynomialModel class, which aims to learn polynomial relationships between data.
 * Unlike the LinearModel class, the PolynomialModel class allows the user to adjust the complexity/capacity
 * of the model by using the degree parameter in the constructor.
 */
public class PolynomialModel extends SimpleModel{

    /**
     * The degree of the polynomials. Also known as the maximum exponent
     * applied to a component of the input vector.
     */
    private int degree;

    /**
     * The weight matricies of the model which represent the coefficients
     * that are multiplied with x^n. Entries are indexed with weightMatricies.get(x)[y][p]
     * where x is the input vector component index, y is the output vector component index,
     * and p is the index of the coefficient multiplied by x raised to the (p - 1)th power.
     */
    private ArrayList<float[][]> weightMatricies;

    /**
     * The bias vector contains the constants added to the output vector.
     * This is like the b in y = mx + b.
     */
    private float[][] biasVector;

    
    /**
     * Creates a PolynomialModel with user defined input/output sizes and degree.
     * @param numInputs The size of the input vector.
     * @param numOutputs The size of the output vector.
     * @param degree The maximum power applied to an input vector component.
     * A larger degree increases the capacity of the model, but can lead to overfitting.
     * degree should be greater than or equal to 1
     */
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

        Utility.initializeUniform(getParameters(), -0.01f, 0.01f);
    }

    /**
     * Creates a PolynomialModel from a file saved to disk.
     * @param filePath The filepath to the file generated from the saveModel() method
     */
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

    /**
     * @return The degree of the polynomials. Also known as the maximum exponent
     * applied to an input vector component.
     */
    public int getDegree(){
        return this.degree;
    }



    public float[] predict(float[] inputVector){
        //TODO: Check input vector size

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
                    //Using the chain rule:
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