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


    public abstract ArrayList<float[][]> calculateGradient(float[] x, float[] y, Loss loss);

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

                    //clip the gradient if applicable
                    if(valueClip > 0){
                        Utility.clip(rawGradient, -valueClip, valueClip);
                    }

                    //add it to the minibatch pool
                    Utility.addGradient(minibatchGradient, rawGradient, 1.0f / indicies.get(mb).size());
                }

                minibatchGradient = opt.processGradient(minibatchGradient);

                //minibatch gradient is calculated. Add to the model's parameters
                Utility.addGradient(getParameters(), minibatchGradient, -1f);
            }

        }

    }

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
    public float calculateLoss(float[] x, float[] y, Loss loss){

        float[] yPred = predict(x);
        return loss.calculateLossScalar(y, yPred);
    }

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