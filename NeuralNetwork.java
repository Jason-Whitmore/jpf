import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Stack;

/**
 * Defines the NeuralNetwork class, which allows users to construct Neural Networks of various
 * complexity. Includes functionality to create, fit, predict, and evaluate loss.
 */
public class NeuralNetwork extends Model{

    private ArrayList<Input> inputLayers;

    private ArrayList<Layer> outputLayers;

    private ArrayList<Layer> allLayers;

    public NeuralNetwork(ArrayList<Input> inputLayers, ArrayList<Layer> outputLayers){
        super();

        this.inputLayers = inputLayers;
        this.outputLayers = outputLayers;

        allLayers = this.serializeLayers();
        this.updateParameters();
    }



    public NeuralNetwork(Input inputLayer, Layer outputLayer){
        super();
        
        this.inputLayers = new ArrayList<Input>(1);
        this.inputLayers.add(inputLayer);

        this.outputLayers = new ArrayList<Layer>(1);
        this.outputLayers.add(outputLayer);

        this.allLayers = serializeLayers();
        this.updateParameters();
    }


    public NeuralNetwork(String filePath){
        super();
        String neuralNetworkInfo = Utility.getTextFileContents(filePath);

        //Isolate the layer connection information
        String layerConnectionInfo = neuralNetworkInfo.substring(0, neuralNetworkInfo.indexOf("END LAYER CONNECTIONS"));
        layerConnectionInfo = layerConnectionInfo.replace("START LAYER CONNECTIONS", "");

        //Isolate the layer information

        String layerInfo = neuralNetworkInfo.substring(neuralNetworkInfo.indexOf("START ALL LAYER INFO"));

        //Remove the header and footer
        layerInfo.replace("START ALL LAYER INFO\n", "");
        layerInfo.replace("END ALL LAYER INFO\n", "");

        //Split the strings based on layer
        String[] layerStrings = layerInfo.split("LAYER START\n");

        //Construct the layers
        for(int i = 0; i < layerStrings.length; i++){
            //Remove the footer string
            layerStrings[i] = layerStrings[i].replace("LAYER END\n", "");

            Layer l = Layer.createLayerFromString(layerStrings[i]);
            this.allLayers.add(l);
        }


        //Parse the layer info into an adjacency list representation.
        ArrayList<ArrayList<Integer>> adjList = connectionInfoToAdjList(layerConnectionInfo);

        for(int i = 0; i < adjList.size(); i++){
            Layer from = this.allLayers.get(i);

            for(int j = 0; j < adjList.get(i).size(); j++){
                Layer to = this.allLayers.get(j);

                from.getOutputLayers().add(to);
            }
        }

        //Connect all the layers input and output layers
        for(int i = 0; i < this.allLayers.size(); i++){
            this.allLayers.get(i).connectInputAndOutputLayers();
        }

    }

    private ArrayList<ArrayList<Integer>> connectionInfoToAdjList(String layerConnectionInfo){
        ArrayList<ArrayList<Integer>> adjList = new ArrayList<ArrayList<Integer>>(this.allLayers.size());

        String[] connectionSplit = layerConnectionInfo.split("\n");

        for(int i = 0; i < connectionSplit.length; i++){
            //Get the "from" index before the arrow
            int startIndex = Integer.parseInt(connectionSplit[i].substring(0, connectionSplit[i].indexOf(" -> ")));

            //Isolate the "to" indicies and split
            String toIndicies = connectionSplit[i].substring(connectionSplit[i].indexOf(" -> ") + 4);
            String[] numStrings = toIndicies.split(", ");

            ArrayList<Integer> row = new ArrayList<Integer>();

            for(int j = 0; j < numStrings.length; j++){
                row.add(Integer.parseInt(numStrings[j]));
            }

            adjList.add(startIndex, row);
        }

        return adjList;
    }

    /**
     * Performs a depth first search of all the layers in order to serialize them so that the entire model
     * can be saved more easily.
     * @return The serialized list of layers.
     */
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

    /**
     * Populates the parameters of this model with all of the parameters from the layers
     * in the same order as they were serialized.
     */
    private void updateParameters(){

        //Serialize the layers before updating the parameters, if needed
        if(allLayers == null){
            allLayers = this.serializeLayers();
        }

        //Clear the current parameter list
        this.parameters.clear();

        //For each layer, add layer parameters to model parameter list
        for(int i = 0; i < allLayers.size(); i++){
            this.parameters.addAll(allLayers.get(i).getParameters());
        }

    }

    /**
     * Calculates the scalar loss on one sample for a simple neural network.
     * @param x The input vector
     * @param yTrue The ground truth output vector
     * @param loss The loss function to use.
     * @return The scalar loss.
     */
    public float calculateScalarLoss(float[] x, float[] yTrue, Loss loss){
        if(!this.isSimple()){
            throw new AssertionError("Simple version of method called on non simple model. Check for method with different 2d inputs.");
        }

        //Check parameters
        Utility.checkNotNull((Object)x, (Object)yTrue, loss);
        Utility.checkArrayLengthsEqual(x, yTrue);
        Utility.checkEqual(x.length, this.inputLayers.get(0).inputVector.length);
        Utility.checkEqual(yTrue.length, this.outputLayers.get(0).outputVector.length);

        float[] yPred = this.predict(x);

        float[] lossVector = loss.calculateLossVector(yTrue, yPred);

        return Utility.mean(lossVector);
    }

    /**
     * Calculates the scalar loss for each input sample
     * @param x The batch of input vectors.
     * @param yTrue The batch of ground truth output vectors.
     * @param loss The loss function to use.
     * @return An array of scalar losses, each corresponding to a sample.
     */
    public float[] calculateScalarLossBatch(float[][] x, float[][] yTrue, Loss loss){
        if(!this.isSimple()){
            throw new AssertionError("Simple version of method called on complex model. Check for method with different 2d inputs.");
        }

        //Check parameters
        Utility.checkNotNull((Object)x, (Object)yTrue, loss);
        Utility.checkMatrixRectange(x, yTrue);
        Utility.checkArrayLengthsEqual(x, yTrue);
        Utility.checkEqual(x[0].length, this.inputLayers.get(0).inputVector.length);
        Utility.checkEqual(yTrue[0].length, this.outputLayers.get(0).outputVector.length);
        
        float[] result = new float[x.length];

        for(int i = 0; i < result.length; i++){
            result[i] = this.calculateScalarLoss(x[i], yTrue[i], loss);
        }

        return result;
    }

    /**
     * Calculates the scalar loss for each of the output vectors in a complex model.
     * @param x The input vectors for this sample
     * @param yTrue The ground truth output vectors for this sample.
     * @param losses The loss functions, each corresponding with each output vector
     * @return The losses, each entry corresponding with each output vector
     */
    public float[] calculateScalarLoss(float[][] x, float[][] yTrue, Loss[] losses){
        if(this.isSimple()){
            throw new AssertionError("Complex version of method called on simple model. Check for method with different 1d inputs.");
        }

        //Check parameters
        Utility.checkNotNull((Object)x, (Object)yTrue, (Object)losses);
        Utility.checkEqual(x.length, this.inputLayers.size());
        Utility.checkEqual(yTrue.length, this.outputLayers.size());
        Utility.checkEqual(losses.length, this.outputLayers.size());

        float[][] yPred = this.predict(x);

        float[] r = new float[this.outputLayers.size()];
        for(int i = 0; i < r.length; i++){
            Utility.checkNotNull(losses[i]);
            r[i] = losses[i].calculateLossScalar(yTrue[i], yPred[i]);
        }

        return r;
    }

    /**
     * Calculates the scalar loss on multiple data samples for a complex model.
     * @param x The input data samples.
     * @param yTrue The output data samples.
     * @param losses The loss function, one for each output vector.
     * @return The scalar losses, for each sample (first index) and each output vector (second index)
     */
    public float[][] calculateScalarLossBatch(float[][][] x, float[][][] yTrue, Loss[] losses){
        if(this.isSimple()){
            throw new AssertionError("Complex version of method called on simple model. Check for method with 2d inputs.");
        }

        //Check parameters
        Utility.checkNotNull((Object)x, (Object)yTrue, (Object)losses);
        Utility.checkEqual(x.length, yTrue.length);
        Utility.checkEqual(losses.length, this.outputLayers.size());


        float[][] r = new float[x.length][];

        for(int i = 0; i < x.length; i++){
            //Method call to single sample loss should catch any issues with the inputs
            r[i] = this.calculateScalarLoss(x[i], yTrue[i], losses);
        }

        return r;
    }


    /**
     * Predict function for Neural Networks that have only one input vector and one output vector
     * @param x The input vector.
     * @return The output vector. If this model has more than one input vector or more than one output vector, then null is returned.
     */
    public float[] predict(float[] x){
        if(!this.isSimple()){
            throw new AssertionError("Called simple method when the complex version should have been called. Check other predictBatch with 3d input");
        }

        Utility.checkNotNull((Object)x);
        Utility.checkEqual(x.length, this.outputLayers.get(0).outputVector.length);

        float[][] xVectors = new float[1][x.length];
        Utility.copyArrayContents(x, xVectors[0]);
        return predict(xVectors)[0];
    }

    /**
     * Makes predictions on a batch of inputs for simple neural networks.
     * @param x The input vectors.
     * @return The predicted output vectors.
     */
    public float[][] predictBatch(float[][] x){
        if(!this.isSimple()){
            throw new AssertionError("Called simple method when the complex version should have been called. Check other predictBatch with 3d input");
        }

        Utility.checkNotNull((Object)x);
        Utility.checkMatrixRectangle(x);
        Utility.checkArrayNotEmpty(x);
        Utility.checkEqual(x[0].length, this.outputLayers.size());

        float[][] y = new float[x.length][];

        for(int i = 0; i < y.length; i++){
            //predict() will check if x[i] is valid
            y[i] = this.predict(x[i]);
        }

        return y;
    }

    /**
     * Makes a prediction on one data sample. Can be used for both simple and complex models.
     * @param x The input vectors, one for each output layer.
     * @return The output vectors, one for each output layer.
     */
    public float[][] predict(float[][] x){
        //Check parameter
        Utility.checkNotNull((Object)x);
        Utility.checkEqual(x.length, this.inputLayers.size());
        for(int i = 0; i < x.length; i++){
            Utility.checkNotNull((Object)x[i]);
            Utility.checkEqual(x.length, this.inputLayers.get(i).inputVector.length);
        }
        

        //Create the output vectors
        float[][] y = new float[this.outputLayers.size()][];

        for(int i = 0; i < this.outputLayers.size(); i++){
            y[i] = new float[this.outputLayers.get(i).getOutputVector().length];
        }

        Stack<Layer> stack = new Stack<Layer>();

        HashSet<Layer> completedPass = new HashSet<Layer>();

        for(int i = 0; i < inputLayers.size(); i++){
            Utility.copyArrayContents(x[i], inputLayers.get(i).getInputVector());
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
            Utility.copyArrayContents(outputLayers.get(i).getOutputVector(), y[i]);
        }

        return y;
    }

    /**
     * Makes predictions on multiple samples. Designed for complex models but can be used for simple models.
     * @param x The input samples.
     * @return The prediction outputs, one for each input sample.
     */
    public float[][][] predictBatch(float[][][] x){
        //Check parameters
        Utility.checkNotNull((Object)x);
        
        float[][][] y = new float[x.length][][];

        for(int i = 0; i < x.length; i++){
            //predictBatch() should check x[i]
            y[i] = this.predictBatch(x[i]);
        }

        return y;
    }


    private ArrayList<float[][]> calculateGradient(float[][] inputVectors, float[][] outputVectors, Loss[] losses){
        //Clear the dLdX and dLdY vectors from all layers
        for(int i = 0; i < allLayers.size(); i++){
            if(allLayers.get(i).getdLdX() != null){
                Utility.clearArray(allLayers.get(i).getdLdX());
            }
            if(allLayers.get(i).getdLdY() != null){
                Utility.clearArray(allLayers.get(i).getdLdY());
            }
        }


        ArrayList<float[][]> grad = new ArrayList<float[][]>();
        //complete the forward pass
        float[][] yPreds = predict(inputVectors);

        Stack<Layer> stack = new Stack<Layer>();
        HashSet<Layer> completed = new HashSet<Layer>();

        for(int i = 0; i < outputLayers.size(); i++){
            float[] error = losses[i].calculateLossVectorGradient(outputVectors[i], yPreds[i]);
            Utility.copyArrayContents(error, outputLayers.get(i).getdLdY());
            outputLayers.get(i).backwardPass();
            completed.add(outputLayers.get(i));
        }

        for(int i = 0; i < inputLayers.size(); i++){
            stack.push(inputLayers.get(i));
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



        //Iterate over the layers and collect the gradients into one ArrayList.
        for(int i = 0; i < allLayers.size(); i++){
            if(allLayers.get(i).getGradient() != null){
                grad.addAll(allLayers.get(i).getGradient());
            }

        }

        return grad;
    }


    public void fit(float[][][] x, float[][][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss[] losses){
        
        for(int e = 0; e < epochs; e++){
            //calculate minibatch indicies
            ArrayList<ArrayList<Integer>> indicies = Utility.getMinibatchIndicies(x.length, minibatchSize);
            

            for(int mb = 0; mb < indicies.size(); mb++){
                //create space to store the averaged collection of gradients
                ArrayList<float[][]> minibatchGradient = Utility.cloneArrays(getParameters());
                Utility.clearArrays(minibatchGradient);

                //calculate gradients based on each data sample in the minibatch
                for(int i = 0; i < indicies.get(mb).size(); i++){

                    float[][] trainX = x[indicies.get(mb).get(i)];
                    float[][] trainY = y[indicies.get(mb).get(i)];

                    ArrayList<float[][]> rawGradient = calculateGradient(trainX, trainY, losses);

                    //add gradient to minibatch pool
                    Utility.addList(minibatchGradient, rawGradient, 1.0f / indicies.get(mb).size());
                }

                //clip the gradient if applicable
                if(valueClip > 0){
                    Utility.clip(minibatchGradient, -valueClip, valueClip);
                }

                minibatchGradient = opt.processGradient(minibatchGradient);

                //minibatch processed. Add to Network's parameters
                Utility.addList(getParameters(), minibatchGradient, -1f);

            }
        }
    }


    public void fit(float[][][] x, float[][][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, ArrayList<Loss> losses){
        Loss[] lossArray = new Loss[losses.size()];
        lossArray = losses.toArray(lossArray);

        fit(x, y, epochs, minibatchSize, valueClip, opt, lossArray);
    }


    /**
     * Fits a single input, single output vector dataset to this model.
     * @param x
     * @param y
     * @param epochs
     * @param minibatchSize
     * @param valueClip
     * @param opt
     * @param loss
     */
    public void fit(float[][] x, float[][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){
        //Check if model has one input and one output vector, if not, throw an exception?
        float[][][] trainX = new float[x.length][1][];
        float[][][] trainY = new float[y.length][1][];

        for(int i = 0; i < x.length; i++){
            trainX[i][0] = x[i];
            trainY[i][0] = y[i];
        }


        Loss[] losses = new Loss[1];
        losses[0] = loss;

        fit(trainX, trainY, epochs, minibatchSize, valueClip, opt, losses);
    }


    private HashMap<Layer, Integer> getLayerIndexMap(){
        HashMap<Layer, Integer> map = new HashMap<>();

        for(int i = 0; i < allLayers.size(); i++){
            map.put(allLayers.get(i), i);
        }

        return map;
    }

    private String connectionInfoToString(){
        HashMap<Layer, Integer> indexMap = getLayerIndexMap();

        //Determine the connection information and write to string
        //From-to format for connections: fromIndex -> toIndex, toIndex, ...

        StringBuilder connectionSB = new StringBuilder();

        //Write the starting string for this section
        connectionSB.append("START LAYER CONNECTIONS INFO\n");

        for(int i = 0; i < allLayers.size(); i++){
            StringBuilder lineSB = new StringBuilder();

            lineSB.append(i + " -> ");
            
            if(allLayers.get(i).getOutputLayers().size() >= 1){
                for(int j = 0; j < allLayers.get(i).getOutputLayers().size() - 1; j++){
                    lineSB.append(indexMap.get(allLayers.get(i).getOutputLayers().get(j)) + ", ");
                }
    
                lineSB.append(indexMap.get(allLayers.get(i).getOutputLayers().get(allLayers.get(i).getOutputLayers().size() - 1)));
            }

            lineSB.append("\n");
            connectionSB.append(lineSB);
        }


        //Write the end of section string
        connectionSB.append("END LAYER CONNECTIONS INFO");

        return connectionSB.toString();
    }


    private String layerInfoToString(){
        StringBuilder sb = new StringBuilder();

        sb.append("START ALL LAYER INFO\n");

        for(int i = 0; i < allLayers.size(); i++){
            sb.append("LAYER START\n");
            sb.append(allLayers.get(i).toString());
            sb.append("\n");
            sb.append("LAYER END\n");
        }

        sb.append("END ALL LAYER INFO");

        return sb.toString();
    }

    /**
     * Determines whether or not the neural network is simple (exactly 1 input and output layer)
     * or complex (other combinations of input and output layers)
     * @return True if the model is simple.
     */
    public boolean isSimple(){
        return this.inputLayers.size() == 1 && this.outputLayers.size() == 1;
    }

    public void saveModel(String filePath){

        String connectionInfo = connectionInfoToString();
        String layerInfo = layerInfoToString();
        String modelInfo = connectionInfo + "\n" + layerInfo;

        Utility.writeStringToFile(filePath, modelInfo);

    }
}