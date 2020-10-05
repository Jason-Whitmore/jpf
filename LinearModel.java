import java.util.ArrayList;


public class LinearModel extends Model{

    private float[][] transformationMatrix;

    private float[][] biasVector;

    private int numInputs;

    private int numOutputs;

    

    public LinearModel(int numInputs, int numOutputs){
        transformationMatrix = new float[numOutputs][numInputs];

        biasVector = new float[numOutputs][1];

        ArrayList<float[][]> params = new ArrayList<float[][]>();

        params.add(transformationMatrix);
        params.add(biasVector);

        this.numInputs = numInputs;

        this.numOutputs = numOutputs;
    }

    public void setParameters(ArrayList<float[][]> newParams){
        if(newParams.size() != 2){
            //TODO: Not correct number of parameter matricies
        }

        //Validate parameter shapes

        if(LinearAlgebra.getNumRows(newParams.get(0)) != numOutputs || LinearAlgebra.getNumColumns(newParams.get(0)) != numInputs){
            //TODO: Bad transformation matrix dimensions
        }

        if(LinearAlgebra.getNumColumns(newParams.get(1)) != 1 || LinearAlgebra.getNumRows(newParams.get(1)) != numOutputs){
            //TODO: Bad bias matrix dimensions
        }

        super.setParameters(newParams);

        transformationMatrix = newParams.get(0);

        biasVector = newParams.get(1);
    }

    private ArrayList<float[][]> calculateGradient(float[] inputVector, float[] outputVector, Loss loss){
        ArrayList<float[][]> gradient = new ArrayList<float[][]>(2);

        float[] yPredicted = this.predict(inputVector);

        float[] errorArray = loss.lossVectorGradient(outputVector, yPredicted);

        //Calculate the error for the final step (adding the bias)
        float[][] biasGradient = LinearAlgebra.arrayToMatrix(errorArray);
        gradient.set(1, biasGradient);

        
        //Calculate error for the transformation matrix
        float[][] transformationGradient = new float[LinearAlgebra.getNumRows(transformationMatrix)][LinearAlgebra.getNumColumns(transformationMatrix)];

        for(int r = 0; r < LinearAlgebra.getNumRows(transformationGradient); r++){
            for(int c = 0; c < LinearAlgebra.getNumColumns(transformationGradient); c++){
                transformationGradient[r][c] = inputVector[c] * biasGradient[r][0];
            }
        }

        gradient.set(0, transformationGradient);

        return gradient;
    }


    public ArrayList<float[]> predict(ArrayList<float[]> inputVectors){
        ArrayList<float[]> r = new ArrayList<float[]>();

        for(int i = 0; i < inputVectors.size(); i++){
            r.add(predict(inputVectors.get(i)));
        }

        return r;
    }

    public float[] predict(float[] inputVector){
        float[] result = new float[numOutputs];
        float[][] resultMatrix = new float[numOutputs][1];

        float[][] matrixB = new float[inputVector.length][1];

        for(int i = 0; i < inputVector.length; i++){
            matrixB[i][0] = inputVector[i];
        }

        LinearAlgebra.matrixMultiply(transformationMatrix, matrixB, resultMatrix);

        //Add the bias
        LinearAlgebra.matrixAdd(resultMatrix, biasVector, resultMatrix);

        //Turn back into a vector
        for(int i = 0; i < result.length; i++){
            result[i] = resultMatrix[i][0];
        }

        return result;
    }

    public void fit(ArrayList<float[][]> x, ArrayList<float[][]> y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){
        if(x.size() != 1 || y.size() != 1){
            //TODO: Throw exception here
        }

        fit(x.get(0), y.get(0), epochs, minibatchSize, valueClip, opt, loss);
    }

    public void fit(float[][] x, float[][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){


        for(int e = 0; e < epochs; e++){
            //calculate minibatch indicies
            ArrayList<ArrayList<Integer>> indicies = Utility.getMinibatchIndicies(x.length, minibatchSize);



            //for each minibatch...
            for(int mb = 0; mb < indicies.size(); mb++){

                ArrayList<float[][]> minibatchGradient = Utility.cloneArrays(getParameters());

                //for each data point in the minibatch:
                for(int i = 0; i < indicies.get(mb).size(); i++){
                    int index = indicies.get(mb).get(i);

                    //calculate the gradient
                    ArrayList<float[][]> rawGradient = calculateGradient(x[index], y[index], loss);

                    //clip the gradient
                    Utility.clip(rawGradient, -valueClip, valueClip);

                    //get the new gradient from the optimizer
                    ArrayList<float[][]> gradient = opt.processGradient(rawGradient);

                    //add it to the minibatch pool

                    Utility.addGradient(minibatchGradient, gradient, 1.0f / indicies.get(mb).size());
                }

                //minibatch gradient is calculated. Add to the model's parameters
                Utility.addGradient(getParameters(), minibatchGradient, -1);
            }

        }
    }
}