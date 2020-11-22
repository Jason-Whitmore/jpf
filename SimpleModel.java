import java.util.ArrayList;


public abstract class SimpleModel extends Model{

    private int numInputs;

    private int numOutputs;



    public abstract float[] predict(float[] inputVector);


    public float[][] predict(float[][] inputVectors){
        float[][] outputVectors = new float[inputVectors.length][];

        for(int i = 0; i < outputVectors.length; i++){
            outputVectors[i] = predict(inputVectors[i]);
        }

        return outputVectors;
    }

    /**
     * Fits the model based on training parameters
     * @param x The training inputs as a 2d array. x[i] should be the ith input vector in the
     * training dataset.
     * @param y The training outputs as a 2d array. y[i] should be the ith output vector in the
     * training dataset.
     * @param epochs The number of complete passes over the dataset during training.
     * @param minibatchSize The number of training examples used in a single update of model parameters. A higher number
     * takes longer to compute, but is more representative of the dataset and produces a "smoother" gradient.
     * @param valueClip Elementwise clip to the gradient vector to prevent exploding gradients
     * @param opt The optimizer to use during training.
     * @param loss The loss function used to calculate the loss at the output vector(s)
     */
    public abstract void fit(float[][] x, float[][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss);

    public void fit(ArrayList<float[]> x, ArrayList<float[]> y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){
        float[][] xArray = new float[x.size()][];
        
        for(int i = 0; i < x.size(); i++){
            xArray[i] = x.get(i);
        }


        float[][] yArray = new float[y.size()][];

        for(int i = 0; i < y.size(); i++){
            yArray[i] = y.get(i);
        }

        fit(xArray, yArray, epochs, minibatchSize, valueClip, opt, loss);
    }

    /**
     * Calculates scalar loss on one data sample.
     * @param x The model input.
     * @param y The model output.
     * @param loss The loss function to use to calculate the scalar loss.
     * @return The scalar loss
     */
    public abstract float calculateLoss(float[] x, float[] y, Loss loss);

    /**
     * Calculates the average scalar loss on multiple data samples.
     * @param x The model inputs.
     * @param y The model ouputs corresponding with inputs
     * @param loss The loss function to use to calculate the scalar losses.
     * @return The average scalar loss.
     */
    public float calculateLoss(ArrayList<float[]> x, ArrayList<float[]> y, Loss loss){
        float[][] xArray = new float[x.size()][];

        for(int i = 0; i < x.size(); i++){
            xArray[i] = x.get(i);
        }


        float[][] yArray = new float[y.size()][];

        for(int i = 0; i < y.size(); i++){
            yArray[i] = y.get(i);
        }

        return calculateLoss(xArray, yArray, loss);
    }
    
    /**
     * Calculates the average scalar loss on multiple data samples.
     * @param x The model inputs.
     * @param y The model ouputs corresponding with inputs
     * @param loss The loss function to use to calculate the scalar losses.
     * @return The average scalar loss.
     */
    public float calculateLoss(float[][] x, float[][] y, Loss loss){
        float sum = 0;

        for(int i = 0; i < x.length; i++){
            sum += calculateLoss(x[i], y[i], loss);
        }

        return sum / x.length;
    }
}