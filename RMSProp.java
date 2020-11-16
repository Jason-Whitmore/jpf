import java.util.ArrayList;

public class RMSProp implements Optimizer{

    private float learningRate;

    private float rho;

    private float epsilon;

    ArrayList<float[][]> gradSquare;

    public RMSProp(){
        learningRate = 0.0001f;

        rho = 0.9f;

        epsilon = 0.000001f;

        gradSquare = null;
    }

    public RMSProp(float learningRate, float rho, float epsilon){
        this.learningRate = learningRate;

        this.rho = rho;

        this.epsilon = epsilon;

        gradSquare = null;
    }


    public ArrayList<float[][]> processGradient(ArrayList<float[][]> rawGradient){

        //Create internal state if it isn't already there.
        if(gradSquare == null){
            ArrayList<float[][]> newState = Utility.cloneArrays(rawGradient);
            Utility.clearArrays(newState);
            gradSquare = newState;
        }



        //Update the state of the optimizer (estimation for avg squared gradient)
        for(int i = 0; i < gradSquare.size(); i++){
            for(int r = 0; r < gradSquare.get(i).length; r++){
                for(int c = 0; c < gradSquare.get(i)[r].length; c++){

                    //maintain exponentially decaying average
                    gradSquare.get(i)[r][c] = (rho * gradSquare.get(i)[r][c]) + ((1 - rho) * ((float)Math.pow(rawGradient.get(i)[r][c], 2)));
                }
            }
        }


        ArrayList<float[][]> grad = Utility.cloneArrays(rawGradient);
        Utility.clearArrays(grad);
        
        //populate gradient
        for(int i = 0; i < grad.size(); i++){
            for(int r = 0; r < grad.get(i).length; r++){
                for(int c = 0; c < grad.get(i)[r].length; c++){
                    grad.get(i)[r][c] = learningRate * (1f / (float)Math.sqrt(gradSquare.get(i)[r][c] + epsilon)) * rawGradient.get(i)[r][c];
                }
            }
        }

        return grad;
    }

    /**
     * Resets the optimizer state. Useful when retraining a model on new data.
     */
    public void resetState(){
        if(gradSquare == null){
            return;
        }

        ArrayList<float[][]> newState = Utility.cloneArrays(gradSquare);
        Utility.clearArrays(newState);

        gradSquare = newState;
    }
}