import java.util.ArrayList;
import java.util.HashSet;
import java.util.Stack;


public class NeuralNetwork extends Model{

    private ArrayList<Input> inputLayers;

    private ArrayList<Layer> outputLayers;

    private ArrayList<Layer> allLayers;

    public NeuralNetwork(ArrayList<Input> inputLayers, ArrayList<Layer> outputLayers){

        this.inputLayers = inputLayers;

        this.outputLayers = outputLayers;

        allLayers = serializeLayers();
    }

    private ArrayList<Layer> serializeLayers(){
        //Use depth first traversal to grab all layers
        HashSet<Layer> visited = new HashSet<Layer>();

        Stack<Layer> stack = new Stack<Layer>();

        ArrayList<Layer> r = new ArrayList<Layer>();

        stack.push(inputLayers.get(0));

        while(!stack.empty()){

            Layer top = stack.pop();
            
            if(!visited.contains(top)){
                visited.add(top);
                
                r.add(top);

                for(int i = 0; i < top.getOutputLayers().size(); i++){
                    stack.push(top.getOutputLayers().get(i));
                }
            }
            
        }

        return r;
    }


    private void updateParameters(){
        if(allLayers == null){
            allLayers = serializeLayers();
        }

        ArrayList<float[][]> params = new ArrayList<float[][]>();

        for(int i = 0; i < allLayers.size(); i++){
            
        }
    }




    public ArrayList<float[]> predict(ArrayList<float[]> inputVectors){
        return null;
    }

    private ArrayList<float[][]> calculateGradient(ArrayList<float[]> inputVectors, ArrayList<float[]> outputVectors){
        return null;
    }


    public void saveModel(String filePath){

    }
}