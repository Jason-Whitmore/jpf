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
        updateParameters();
    }



    public NeuralNetwork(Input inputLayer, Layer outputLayer){
        ArrayList<Input> inputList = new ArrayList<Input>();
        inputList.add(inputLayer);
        this.inputLayers = inputList;

        ArrayList<Layer> outputList = new ArrayList<Layer>();
        outputList.add(outputLayer);
        this.outputLayers = outputList;

        allLayers = serializeLayers();
        updateParameters();
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

                for(int i = 0; i < top.getInputLayers().size(); i++){
                    stack.push(top.getInputLayers().get(i));
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
            params.addAll(allLayers.get(i).getParameters());
        }

        setParameters(params);
    }




    public ArrayList<float[]> predict(ArrayList<float[]> inputVectors){
        ArrayList<float[]> outputVectors = new ArrayList<float[]>(outputLayers.size());

        Stack<Layer> stack = new Stack<Layer>();

        HashSet<Layer> completedPass = new HashSet<Layer>();

        for(int i = 0; i < inputLayers.size(); i++){
            Utility.copyArrayContents(inputVectors.get(i), inputLayers.get(i).getInputVector());
        }

        for(int i = 0; i < outputLayers.size(); i++){
            stack.push(outputLayers.get(i));
        }


        while(!stack.empty()){
            Layer top = stack.peek();

            if(completedPass.contains(top)){
                stack.pop();
            } else {

                //check to see if all of the required input passes have been completed
                boolean canForwardPass = true;
                for(int i = 0; i < top.getInputLayers().size(); i++){
                    if(!completedPass.contains(top.getInputLayers().get(i))){
                        stack.push(top.getInputLayers().get(i));
                        canForwardPass = false;
                    }
                }

                if(canForwardPass){
                    top.forwardPass();
                    completedPass.add(top);
                    stack.pop();
                }
            }
        }

        //Allocate copies of the output vectors from the output layers
        for(int i = 0; i < outputLayers.size(); i++){
            float[] output = outputLayers.get(i).inputVector.clone();
            outputVectors.add(output);
        }

        return outputVectors;
    }


    public float[] predict(float[] x){
        //Check to see if model input arrays are compatible
        if(inputLayers.size() != 1 || outputLayers.size() != 1){
            return null;
        }

        ArrayList<float[]> inputVectors = new ArrayList<float[]>();

        inputVectors.add(x);

        ArrayList<float[]> outputList = predict(inputVectors);

        return outputList.get(0);
    }


    private ArrayList<float[][]> calculateGradient(float[] inputVector, float[] outputVector, Loss loss){
        ArrayList<float[]> inputList = new ArrayList<float[]>();
        inputList.add(inputVector);

        ArrayList<float[]> outputList = new ArrayList<float[]>();
        outputList.add(outputVector);

        ArrayList<Loss> lossList = new ArrayList<Loss>();
        lossList.add(loss);

        return calculateGradient(inputList, outputList, lossList);
    }

    private ArrayList<float[][]> calculateGradient(ArrayList<float[]> inputVectors, ArrayList<float[]> outputVectors, ArrayList<Loss> losses){
        ArrayList<float[][]> grad = new ArrayList<float[][]>();
        //complete the forward pass
        ArrayList<float[]> yPreds = predict(inputVectors);

        Stack<Layer> stack = new Stack<Layer>();
        HashSet<Layer> completed = new HashSet<Layer>();

        for(int i = 0; i < outputLayers.size(); i++){
            float[] error = losses.get(i).calculateLossVectorGradient(outputVectors.get(i), yPreds.get(i));
            outputLayers.get(i).setLayerError(error);
        }

        for(int i = 0; i < inputLayers.size(); i++){
            stack.push(inputLayers.get(i));
        }

        while(!stack.empty()){
            Layer top = stack.pop();

            boolean canComplete = true;

            for(int i = 0; i < top.getOutputLayers().size(); i++){
                if(!completed.contains(top.getOutputLayers().get(i))){
                    canComplete = false;
                    stack.push(top.getOutputLayers().get(i));
                }
            }

            if(canComplete){
                top.backwardPass();
                stack.pop();
                completed.add(top);
            }
        }



        //Iterate over the layers and collect the gradients into one ArrayList
        for(int i = 0; i < allLayers.size(); i++){
            grad.addAll(allLayers.get(i).getGradient());
        }

        return grad;
    }


    public void resetRecurrentStates(){

        for(int i = 0; i < allLayers.size(); i++){

            //Reset layer if LSTM
            if(true){

            }

            //Resest layer if recurrent
            if(true){

            }
        }
    }


    public void fit(ArrayList<float[][]> x, ArrayList<float[][]> y, int epochs, int minibatchSize, float valueClip, Optimizer opt, ArrayList<Loss> losses){
        
    }


    public void saveModel(String filePath){

    }
}