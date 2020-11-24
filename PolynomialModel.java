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


    /**
     * Fits the model's parameters to minimize the loss on the training dataset.
     * @param x The training inputs. Number of columns should match the model's input size, and rows should be the number of data points.
     * @param y The training outputs. Number of columns should match the model's output size, and rows should be the number of data points.
     * @param epochs The number of epochs (or complete training passes over the training dataset). Should be greater than 0.
     * @param minibatchSize The number of training data examples used when making a single update to the model's parameters.
     * A higher number will take longer to compute, but the updates will be less noisy. Should be greater than 0.
     * @param valueClip Clips the gradient's components to be in the range [-valueClip, valueClip]. Helps to stop exploding gradients.
     * @param opt The optimizer to use during fitting.
     * @param Loss The loss function used to make the model more accurate to the training dataset.
     */
    public void fit(float[][] x, float[][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){

        for(int e = 0; e < epochs; e++){
            //calculate minibatch indicies
            ArrayList<ArrayList<Integer>> indicies = Utility.getMinibatchIndicies(x.length, minibatchSize);
            //for each minibatch...
            for(int mb = 0; mb < indicies.size(); mb++){

                ArrayList<float[][]> minibatchGradient = Utility.cloneArrays(getParameters());
                Utility.clearArrays(minibatchGradient);

                //for each data point in the minibatch
                for(int i = 0; i < indicies.get(mb).size(); i++){
                    int index = indicies.get(mb).get(i);

                    //calculate the gradient
                    ArrayList<float[][]> rawGradient = calculateGradient(x[index], y[index], loss);

                    //clip the gradient
                    Utility.clip(rawGradient, -valueClip, valueClip);

                    //add it to the minibatch pool
                    Utility.addGradient(minibatchGradient, rawGradient, 1.0f / indicies.get(mb).size());
                }

                minibatchGradient = opt.processGradient(minibatchGradient);

                //minibatch gradient is calculated. Add to the model's parameters
                Utility.addGradient(getParameters(), minibatchGradient, -1f);
            }

        }

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