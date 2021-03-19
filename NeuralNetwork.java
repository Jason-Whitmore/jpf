import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Stack;


public class NeuralNetwork extends Model{

    private Input[] inputLayers;

    private Layer[] outputLayers;

    private Layer[] allLayers;

    public NeuralNetwork(ArrayList<Input> inputLayers, ArrayList<Layer> outputLayers){
        this.inputLayers = inputLayers.toArray(new Input[inputLayers.size()]);
        this.outputLayers = outputLayers.toArray(new Input[outputLayers.size()]);;

        allLayers = serializeLayers();
        updateParameters();
    }



    public NeuralNetwork(Input inputLayer, Layer outputLayer){
        Input[] inputArray = new Input[1];
        inputArray[0] = inputLayer;
        this.inputLayers = inputArray;

        Layer[] outputArray = new Layer[1];
        outputArray[0] = outputLayer;
        this.outputLayers = outputArray;

        allLayers = serializeLayers();
        updateParameters();
    }

    private Layer[] serializeLayers(){
        //Use depth first traversal to grab all layers
        HashSet<Layer> visited = new HashSet<Layer>();

        Stack<Layer> stack = new Stack<Layer>();

        ArrayList<Layer> r = new ArrayList<Layer>();

        stack.push(inputLayers[0]);

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

        return r.toArray(new Layer[r.size()]);
    }


    private void updateParameters(){
        if(allLayers == null){
            allLayers = serializeLayers();
        }

        ArrayList<float[][]> params = new ArrayList<float[][]>();

        for(int i = 0; i < allLayers.length; i++){
            params.addAll(allLayers[i].getParameters());
        }

        setParameters(params);
    }




    public ArrayList<float[]> predict(ArrayList<float[]> inputVectors){
        ArrayList<float[]> outputVectors = new ArrayList<float[]>(outputLayers.length);

        Stack<Layer> stack = new Stack<Layer>();

        HashSet<Layer> completedPass = new HashSet<Layer>();

        for(int i = 0; i < inputLayers.length; i++){
            Utility.copyArrayContents(inputVectors.get(i), inputLayers[i].getInputVector());
        }

        for(int i = 0; i < outputLayers.length; i++){
            stack.push(outputLayers[i]);
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
        for(int i = 0; i < outputLayers.length; i++){
            float[] output = outputLayers[i].outputVector.clone();
            outputVectors.add(output);
        }

        return outputVectors;
    }

    public float calculateLoss(float[] inputVector, float[] outputVector, Loss loss){
        if(inputLayers.length != 1 || outputLayers.length != 1){
            //TODO: Crash program if wrong loss function is used?
            return Float.NaN;
        }

        float[] yPred = predict(inputVector);

        return loss.calculateLossScalar(outputVector, yPred);
    }

    public float calculateLoss(float[][] inputVectors, float[][] outputVectors, Loss loss){
        float sum = 0;

        for(int i = 0; i < inputVectors.length; i++){
            sum += calculateLoss(inputVectors[i], outputVectors[i], loss);
        }

        return sum / inputVectors.length;
    }



    public float[] calculateLoss(ArrayList<float[]> inputs, ArrayList<float[]> outputs, ArrayList<Loss> losses){
        float[] scalarLosses = new float[outputs.size()];

        ArrayList<float[]> yPreds = predict(inputs);

        for(int i = 0; i < scalarLosses.length; i++){
            scalarLosses[i] = losses.get(i).calculateLossScalar(outputs.get(i), yPreds.get(i));
        }

        return scalarLosses;
    }


    public float calculateAverageLoss(ArrayList<float[]> inputs, ArrayList<float[]> outputs, ArrayList<Loss> losses){
        float[] scalarLosses = calculateLoss(inputs, outputs, losses);

        float sum = 0;

        for(int i = 0; i < scalarLosses.length; i++){
            sum += scalarLosses[i];
        }

        return sum / scalarLosses.length;
    }


    public float[] predict(float[] x){
        //Check to see if model input arrays are compatible
        if(inputLayers.length != 1 || outputLayers.length != 1){
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
        //Clear the dLdX and dLdY vectors from all layers
        for(int i = 0; i < allLayers.length; i++){
            if(allLayers[i].getdLdX() != null){
                Utility.clearArray(allLayers[i].getdLdX());
            }
            if(allLayers[i].getdLdY() != null){
                Utility.clearArray(allLayers[i].getdLdY());
            }
        }


        ArrayList<float[][]> grad = new ArrayList<float[][]>();
        //complete the forward pass
        ArrayList<float[]> yPreds = predict(inputVectors);

        Stack<Layer> stack = new Stack<Layer>();
        HashSet<Layer> completed = new HashSet<Layer>();

        for(int i = 0; i < outputLayers.length; i++){
            float[] error = losses.get(i).calculateLossVectorGradient(outputVectors.get(i), yPreds.get(i));
            outputLayers[i].setdLdY(error);
            outputLayers[i].backwardPass();
            completed.add(outputLayers[i]);
        }

        for(int i = 0; i < inputLayers.length; i++){
            stack.push(inputLayers[i]);
        }

        while(!stack.empty()){
            Layer top = stack.peek();

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



        //Iterate over the layers and collect the gradients into one ArrayList. Also, reset the dLdY and dLdX vectors
        for(int i = 0; i < allLayers.length; i++){
            if(allLayers[i].getGradient() != null){
                grad.addAll(allLayers[i].getGradient());
            }

        }

        return grad;
    }


    public void resetRecurrentStates(){

        for(int i = 0; i < allLayers.length; i++){

            //Reset layer if LSTM
            if(true){

            }

            //Reset layer if recurrent
            if(true){

            }
        }
    }


    public void fit(ArrayList<float[][]> x, ArrayList<float[][]> y, int epochs, int minibatchSize, float valueClip, Optimizer opt, ArrayList<Loss> losses){
        
        for(int e = 0; e < epochs; e++){
            //calculate minibatch indicies
            ArrayList<ArrayList<Integer>> indicies = Utility.getMinibatchIndicies(x.get(0).length, minibatchSize);
            

            for(int mb = 0; mb < indicies.size(); mb++){
                //create space to store the averaged collection of gradients
                ArrayList<float[][]> minibatchGradient = Utility.cloneArrays(getParameters());
                Utility.clearArrays(minibatchGradient);

                //calculate gradients based on each data sample in the minibatch
                for(int i = 0; i < indicies.get(mb).size(); i++){

                    ArrayList<float[]> trainX = isolateRow(x, indicies.get(mb).get(i));
                    ArrayList<float[]> trainY = isolateRow(y, indicies.get(mb).get(i));

                    ArrayList<float[][]> rawGradient = calculateGradient(trainX, trainY, losses);

                    //System.out.println(Utility.arraysToString(rawGradient));
                    //System.exit(0);


                    //add gradient to minibatch pool
                    Utility.addGradient(minibatchGradient, rawGradient, 1.0f / indicies.get(mb).size());
                }

                //clip the gradient if applicable
                if(valueClip > 0){
                    Utility.clip(minibatchGradient, -valueClip, valueClip);
                }

                minibatchGradient = opt.processGradient(minibatchGradient);

                //minibatch processed. Add to Network's parameters
                Utility.addGradient(getParameters(), minibatchGradient, -1f);

            }
        }
    }



    private ArrayList<float[]> isolateRow(ArrayList<float[][]> data, int row){
        ArrayList<float[]> ret = new ArrayList<float[]>(data.size());

        for(int i = 0; i < data.size(); i++){
            ret.add(data.get(i)[row]);
        }

        return ret;
    }


    public void fit(float[][] x, float[][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){
        ArrayList<float[][]> inputList = new ArrayList<float[][]>();
        inputList.add(x);

        ArrayList<float[][]> outputList = new ArrayList<float[][]>();
        outputList.add(y);

        ArrayList<Loss> lossList = new ArrayList<Loss>();
        lossList.add(loss);

        fit(inputList, outputList, epochs, minibatchSize, valueClip, opt, lossList);
    }


    private HashMap<Layer, Integer> getLayerIndexMap(){
        HashMap<Layer, Integer> map = new HashMap<>();

        for(int i = 0; i < allLayers.length; i++){
            map.put(allLayers[i], i);
        }

        return map;
    }

    private String connectionInfoToString(){
        HashMap<Layer, Integer> indexMap = getLayerIndexMap();

        //Determine the connection information and write to string
        //From-to format for connections: fromIndex -> toIndex, toIndex, ...

        StringBuilder connectionSB = new StringBuilder();

        //Write the starting string for this section
        connectionSB.append("START LAYER CONNECTIONS:\n");

        for(int i = 0; i < allLayers.length; i++){
            StringBuilder lineSB = new StringBuilder();

            lineSB.append(i + " -> ");

            for(int j = 0; j < allLayers[i].getOutputLayers().size() - 1; i++){
                lineSB.append(indexMap + ", ");
            }

            lineSB.append(allLayers[i].getOutputLayers().get(allLayers[i].getOutputLayers().size() - 1));
            lineSB.append("\n");
        }


        //Write the end of section string
        connectionSB.append("END LAYER CONNECTIONS");

        return connectionSB.toString();
    }


    private String layerInfoToString(){
        StringBuilder sb = new StringBuilder();

        sb.append("START LAYER INFO");

        for(int i = 0; i < allLayers.length; i++){
            sb.append(allLayers[i].toString());
            sb.append("\n");
        }

        sb.append("END LAYER INFO");

        return sb.toString();
    }

    public void saveModel(String filePath){

        String connectionInfo = connectionInfoToString();
        String layerInfo = layerInfoToString();
        String modelInfo = connectionInfo + "\n" + layerInfo;

        Utility.writeStringToFile(filePath, modelInfo);

    }
}