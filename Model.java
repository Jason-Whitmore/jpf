import java.util.ArrayList;


public abstract class Model{
    private ArrayList<float[][]> parameters;

    public void setParameters(ArrayList<float[][]> newParameters){
        parameters = newParameters;
    }

    public ArrayList<float[][]> getParameters(){
        return parameters;
    }

    public int getParameterCount(){
        int count = 0;
        
        for(int i = 0; i < parameters.size(); i++){
            count += parameters.get(i).length * parameters.get(i)[0].length;
        }

        return count;
    }

    public abstract ArrayList<float[]> predict(ArrayList<float[]> inputVectors);

    public abstract void fit(ArrayList<float[][]> x, ArrayList<float[][]> y, int epochs, int minibatchSize, float valueClip, Optimizer opt);

    
}