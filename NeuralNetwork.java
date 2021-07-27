import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Stack;

/**
 * Defines the NeuralNetwork class, which allows users to construct Neural Networks of various
 * complexity. Includes functionality to create, fit, predict, and evaluate loss.
 */
public class NeuralNetwork extends Model{

    /**
     * The list of input layers which will recieve the input vectors during computations.
     * These have an in-degree of 0.
     */
    private ArrayList<Input> inputLayers;

    /**
     * The list of output layers which output the predictions.
     * These have an out-degree of 0.
     */
    private ArrayList<Layer> outputLayers;

    /**
     * The list of all layers in the neural network, ordering done via algorithm in serializeLayers().
     */
    private ArrayList<Layer> allLayers;

    /**
     * Constructor for creating Neural Networks using lists of input and output vectors.
     * This is the most common way of constructing complex (multi input and output) neural networks.
     * @param inputLayers The input layers to the neural network.
     * @param outputLayers The output layers to the neural network.
     */
    public NeuralNetwork(ArrayList<Input> inputLayers, ArrayList<Layer> outputLayers){
        super();

        //Check parameters
        Utility.checkNotNull(inputLayers);
        Utility.checkNotNull(outputLayers);
        
        for(int i = 0; i < inputLayers.size(); i++){
            Utility.checkNotNull(inputLayers.get(i));
        }

        for(int i = 0; i < outputLayers.size(); i++){
            Utility.checkNotNull(outputLayers.get(i));
        }



        this.inputLayers = inputLayers;
        this.outputLayers = outputLayers;

        allLayers = this.serializeLayers();
        this.updateParameters();

        if(this.hasCycle()){
            throw new AssertionError("Neural network contains a cycle, which is forbidden.");
        }
    }


    /**
     * Constructor for a simple (one input layer, one output layer) neural network.
     * @param inputLayer The input layer where the input vector will be fed to.
     * @param outputLayer The output layer, which will produce the neural network's output vector.
     */
    public NeuralNetwork(Input inputLayer, Layer outputLayer){
        super();

        //Check params
        Utility.checkNotNull(inputLayer);
        Utility.checkNotNull(outputLayer);

        
        this.inputLayers = new ArrayList<Input>(1);
        this.inputLayers.add(inputLayer);

        this.outputLayers = new ArrayList<Layer>(1);
        this.outputLayers.add(outputLayer);

        this.allLayers = this.serializeLayers();
        this.updateParameters();

        if(this.hasCycle()){
            throw new AssertionError("Neural network contains a cycle, which is forbidden.");
        }
    }

    /**
     * Constructs a neural network from a file that was created using the saveModel() method.
     * Useful when loading a pretrained neural network rather than training a fresh neural network.
     * @param filePath The filepath of the file created by saveModel()
     */
    public NeuralNetwork(String filePath){
        super();

        //Check param
        Utility.checkNotNull(filePath);


        String neuralNetworkInfo = Utility.getTextFileContents(filePath);

        this.allLayers = this.layersFromString(neuralNetworkInfo);

        this.connectLayers(neuralNetworkInfo);

        //Connect all the layers input and output layers, also, create the input and output layers lists
        this.inputLayers = this.createInputLayerList();
        this.outputLayers = this.createOutputLayerList();

        if(this.hasCycle()){
            throw new AssertionError("Neural network contains a cycle, which is forbidden.");
        }
    }

    /**
     * Detects if the neural network has a cycle in it (possible to form a path that visits a layer twice).
     * Cycles cannot work in this implementation.
     * @return True if there is a cycle in the neural network, else false.
     */
    private boolean hasCycle(){
        
        for(int i = 0; i < this.allLayers.size(); i++){

            if(this.hasCycle(this.allLayers.get(i))){
                return true;
            }

        }

        return false;
    }

    /**
     * Detects if there is a path (cycle) that starts at source and eventually returns to the source
     * @param source The source layer to detect a cycle from.
     * @return True if there is a cycle that starts and ends at source, else false
     */
    private boolean hasCycle(Layer source){

        HashSet<Layer> discovered = new HashSet<Layer>();

        Stack<Layer> stack = new Stack<Layer>();
        stack.push(source);

        while(!stack.empty()){
            Layer l = stack.pop();

            if(!discovered.contains(l)){
                discovered.add(l);

                for(int i = 0; i < l.getOutputLayers().size(); i++){
                    //Check adjacent for source
                    if(l.getOutputLayers().get(i) == source){
                        return true;
                    }

                    stack.push(l.getOutputLayers().get(i));
                }
            }
        }

        return false;

    }

    /**
     * Creates the output layer list based on layers inside of the allLayers field.
     * @return The populated list of output layers.
     */
    private ArrayList<Layer> createOutputLayerList(){
        ArrayList<Layer> outputs = new ArrayList<Layer>();

        for(int i = 0; i < this.allLayers.size(); i++){
            //Check if the layer is an output layer (layers with no outgoing layers)
            if(this.allLayers.get(i).getOutputLayers().size() == 0){
                outputs.add(this.allLayers.get(i));
            }
        }

        return outputs;
    }

    /**
     * Creates the input layer list based on the layers inside of the allLayers field.
     * @return The populated list of input layers.
     */
    private ArrayList<Input> createInputLayerList(){
        ArrayList<Input> inputs = new ArrayList<Input>();

        for(int i = 0; i < this.allLayers.size(); i++){
            //Check to see if the layer is an input layer
            if(this.allLayers.get(i) instanceof Input){
                inputs.add((Input)(this.allLayers.get(i)));
            }
        }

        return inputs;
    }

    /**
     * Reads the adjacency list information stored inside a neural network info string and connects
     * the layers together. The allLayers field should be already created and populated.
     * @param neuralNetworkInfo The string read from the file created from the saveModel() method.
     */
    private void connectLayers(String neuralNetworkInfo){
        //Isolate the layer connection information
        String layerConnectionInfo = neuralNetworkInfo.substring(0, neuralNetworkInfo.indexOf("END LAYER CONNECTIONS INFO"));
        layerConnectionInfo = layerConnectionInfo.replace("START LAYER CONNECTIONS INFO\n", "");

        //Parse the layer info into an adjacency list representation.
        ArrayList<ArrayList<Integer>> adjList = this.connectionInfoToAdjList(layerConnectionInfo);

        for(int i = 0; i < adjList.size(); i++){
            Layer from = this.allLayers.get(i);

            for(int j = 0; j < adjList.get(i).size(); j++){
                int index = adjList.get(i).get(j);
                Layer to = this.allLayers.get(index);

                from.getOutputLayers().add(to);
            }
        }

        //Connect the "backwards" connections (connect from output to input layers)
        for(int i = 0; i < this.allLayers.size(); i++){
            this.allLayers.get(i).connectInputAndOutputLayers();
        }
    }

    /**
     * Initializes the layers of the neural network from a string. Throws an Assertion error on failure to create a layer.
     * @param neuralNetworkInfo The string that is output from the contents written to disk from the saveModel() method
     * @return An ArrayList of initialized layers.
     */
    private ArrayList<Layer> layersFromString(String neuralNetworkInfo){
        //Isolate the layer information
        String layerInfo = neuralNetworkInfo.substring(neuralNetworkInfo.indexOf("START ALL LAYER INFO"));

        //Remove the header and footer
        layerInfo = layerInfo.replace("START ALL LAYER INFO\n", "");
        layerInfo = layerInfo.replace("\nEND ALL LAYER INFO", "");

        //Split the strings based on layer
        String[] layerStrings = layerInfo.split("LAYER START\n");

        ArrayList<Layer> layers = new ArrayList<Layer>();
        //Construct the layers. Start at index 1 since index 0 is an empty string.
        for(int i = 0; i < layerStrings.length; i++){
            //Remove the footer string
            layerStrings[i] = layerStrings[i].replace("\nLAYER END\n", "");

            //Due to the way split works, there may be empty strings in layerStrings array, which should not be read from.
            if(!layerStrings[i].equals("")){
                Layer l = Layer.createLayerFromString(layerStrings[i]);
                if(l == null){
                    throw new AssertionError("Layer creation from string failed on index " + i);
                }
                
                layers.add(l);
            }
        }

        return layers;
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

            //Only add to row if there is anything to add to the row
            if(toIndicies.length() > 0){
                for(int j = 0; j < numStrings.length; j++){
                    row.add(Integer.parseInt(numStrings[j]));
                }
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

        //First, add the input layers in order
        for(int i = 0; i < this.inputLayers.size(); i++){
            r.add(this.inputLayers.get(i));
            visited.add(this.inputLayers.get(i));

            for(int j = 0; j < this.inputLayers.get(i).getOutputLayers().size(); j++){
                stack.push(this.inputLayers.get(i).getOutputLayers().get(j));
            }
        }

        //Add the output layers in order
        for(int i = 0; i < this.outputLayers.size(); i++){
            r.add(this.outputLayers.get(i));
            visited.add(this.outputLayers.get(i));

            for(int j = 0; j < this.outputLayers.get(i).getInputLayers().size(); j++){
                stack.push(this.outputLayers.get(i).getInputLayers().get(j));
            }
        }

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
        Utility.checkMatrixRectangle(x, yTrue);
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
     * @param losses The loss functions, one for each output vector.
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
     * Calculates the scalar loss on multiple data samples for a complex model.
     * @param x The input data samples.
     * @param yTrue The output data samples.
     * @param losses The loss functions, one for each output vector.
     * @return The scalar losses, for each sample (first index) and each output vector (second index)
     */
    public float[][] calculateScalarLossBatch(float[][][] x, float[][][] yTrue, ArrayList<Loss> losses){
        //Check only the loss parameters, since the other parameters will be checked in the next function call.
        Utility.checkNotNull(losses);

        Loss[] lossArray = new Loss[losses.size()];
        lossArray = losses.toArray(lossArray);

        return this.calculateScalarLossBatch(x, yTrue, lossArray);
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
        Utility.checkEqual(x.length, this.inputLayers.get(0).inputVector.length);

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
     * @param x The input vectors, one for each input layer.
     * @return The output vectors, one for each output layer.
     */
    public float[][] predict(float[][] x){
        //Check parameter
        Utility.checkNotNull((Object)x);
        Utility.checkEqual(x.length, this.inputLayers.size());
        for(int i = 0; i < x.length; i++){
            Utility.checkNotNull((Object)x[i]);
            Utility.checkEqual(x[i].length, this.inputLayers.get(i).inputVector.length);
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

    /**
     * Calculates the gradient of the loss functions wrt a complex model's parameters
     * @param inputVectors The input vectors, one vector per input layer.
     * @param outputVectors The output vectors, one vector per output layer.
     * @param losses The loss functions, one per output layer.
     * @return The gradient list, which matches the same shape as the parameters list.
     */
    private ArrayList<float[][]> calculateGradient(float[][] inputVectors, float[][] outputVectors, Loss[] losses){
        //Check params
        this.checkCalculateGradientParams(inputVectors, outputVectors, losses);

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

    /**
     * Checks the parameters for the calculateGradient method.
     * @param inputVectors The input vectors to check compatibility.
     * @param outputVectors The output vectors to check compatibility.
     * @param losses The loss functions to check compatibility.
     */
    private void checkCalculateGradientParams(float[][] inputVectors, float[][] outputVectors, Loss[] losses){
        Utility.checkNotNull(inputVectors, outputVectors, losses);
        Utility.checkEqual(inputVectors.length, this.inputLayers.size());
        Utility.checkEqual(outputVectors.length, this.outputLayers.size());
        Utility.checkEqual(outputVectors.length, losses.length);

        for(int i = 0; i < inputVectors.length; i++){
            Utility.checkNotNull(inputVectors[i]);
            Utility.checkEqual(inputVectors[i].length, this.inputLayers.get(i).outputVector.length);
        }

        for(int i = 0; i < outputVectors.length; i++){
            Utility.checkNotNull(outputVectors[i]);
            Utility.checkEqual(outputVectors[i].length, this.outputLayers.get(i).outputVector.length);
        }
    }

    /**
     * Fits the neural network to the training data and under the specified parameters.
     * @param x The training inputs. Should be consistent with number of input layers and lengths of input vectors.
     * @param y The training output. Should be consistent with number of output layers and the lengths of output vectors.
     * @param epochs The number of epochs, or full passes over the dataset, to complete. Should be >= 0.
     * @param minibatchSize The number of training data samples used in a single parameter update. Larger number increases stability. Should be >= 1.
     * @param valueClip The maximum absoulte value a component of the gradient can be. This helps prevent unstable updates. Values below 0 turn off clipping.
     * @param opt The optimizer to use, which processes the raw gradients in hopes of increasing the rate of learning.
     * @param losses The loss functions to use. Each loss function corresponds to an output layer.
     */
    public void fit(float[][][] x, float[][][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss[] losses){
        //Check parameters
        this.checkFitParams(x, y, epochs, minibatchSize, valueClip, opt, losses);
        
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

                    ArrayList<float[][]> rawGradient = this.calculateGradient(trainX, trainY, losses);

                    //add gradient to minibatch pool
                    Utility.addList(minibatchGradient, rawGradient, 1.0f / indicies.get(mb).size());
                }

                //clip the gradient if applicable
                if(valueClip > 0){
                    Utility.clip(minibatchGradient, -valueClip, valueClip);
                }

                //System.out.println(Utility.arraysToString(minibatchGradient));

                minibatchGradient = opt.processGradient(minibatchGradient);

                //minibatch processed. Add to Network's parameters
                Utility.addList(getParameters(), minibatchGradient, -1f);

            }
        }
    }

    /**
     * Checks the fit parameters for consistency and validity.
     * @param x The training input data.
     * @param y The training output data.
     * @param epochs The number of epochs to train for.
     * @param minibatchSize The number of data samples used in a single parameter update.
     * @param valueClip The maximum absolute value that a gradient component can be.
     * @param opt The training optimizer.
     * @param losses The loss functions.
     */
    private void checkFitParams(float[][][] x, float[][][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss[] losses){
        Utility.checkNotNull((Object)x, (Object)y, opt, losses);

        if(epochs <= 0){
            throw new AssertionError("Number of epochs should be > 0");
        }

        if(minibatchSize <= 0){
            throw new AssertionError("Minibatch size should be > 0");
        }

        if(valueClip <= 0 && valueClip != -1){
            throw new AssertionError("Valueclip should either be > 0 or -1 if no clipping is desired.");
        }

        for(int i = 0; i < losses.length; i++){
            Utility.checkNotNull(losses[i]);
        }
    }

    /**
     * Fits the neural network to the training data and under the specified parameters. Wrapper function.
     * @param x The training inputs. Should be consistent with number of input layers and lengths of input vectors.
     * @param y The training output. Should be consistent with number of output layers and the lengths of output vectors.
     * @param epochs The number of epochs, or full passes over the dataset, to complete. Should be >= 0.
     * @param minibatchSize The number of training data samples used in a single parameter update. Larger number increases stability. Should be >= 1.
     * @param valueClip The maximum absoulte value a component of the gradient can be. This helps prevent unstable updates. Values below 0 turn off clipping.
     * @param opt The optimizer to use, which processes the raw gradients in hopes of increasing the rate of learning.
     * @param losses The loss functions to use. Each loss function corresponds to an output layer.
     */
    public void fit(float[][][] x, float[][][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, ArrayList<Loss> losses){
        Loss[] lossArray = new Loss[losses.size()];
        lossArray = losses.toArray(lossArray);

        this.fit(x, y, epochs, minibatchSize, valueClip, opt, lossArray);
    }


    /**
     * Fits a single input, single output vector dataset to this model. Used for simple models.
     * @param x The training input data.
     * @param y The training output data.
     * @param epochs The number of epochs or number of passes, to train the model for.
     * @param minibatchSize The number of samples used per parameter update.
     * @param valueClip The maximum absolute value a gradient component is limited to being.
     * @param opt The optimizer used to process the raw gradients.
     * @param loss The loss function to minimize during training.
     */
    public void fit(float[][] x, float[][] y, int epochs, int minibatchSize, float valueClip, Optimizer opt, Loss loss){
        //Do some basic parameter checking.
        if(!this.isSimple()){
            throw new AssertionError("Simple version of fit called when the model is complex. Check the fit method with 3d inputs.");
        }

        Utility.checkNotNull((Object)x, (Object)y, opt, loss);
        Utility.checkArrayNotEmpty(x);
        Utility.checkArrayNotEmpty(y);

        float[][][] trainX = new float[x.length][1][];
        float[][][] trainY = new float[y.length][1][];

        for(int i = 0; i < x.length; i++){
            trainX[i][0] = x[i];
            trainY[i][0] = y[i];
        }


        Loss[] losses = new Loss[1];
        losses[0] = loss;

        //Much of the parameter checking will be done in this call
        this.fit(trainX, trainY, epochs, minibatchSize, valueClip, opt, losses);
    }

    /**
     * Creates a map that contains (layer, index) where index is the location of the layer in the allLayers list.
     * Useful when generating the adjacency list.
     * @return The map containing (layer, index) pairs.
     */
    private HashMap<Layer, Integer> getLayerIndexMap(){
        HashMap<Layer, Integer> map = new HashMap<>();

        for(int i = 0; i < this.allLayers.size(); i++){
            map.put(this.allLayers.get(i), i);
        }

        return map;
    }

    /**
     * Creates the connection info string that is used when saving the model to disk.
     * Format of each row: fromIndex -> toIndex, toIndex, ...
     * @return The string containing the connection/adjacency information
     */
    private String connectionInfoToString(){
        HashMap<Layer, Integer> indexMap = this.getLayerIndexMap();

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

    /**
     * Converts all of the layers in the neural network into a String representation that can be saved.
     * @return The string representation of the layers.
     */
    private String layerInfoToString(){
        StringBuilder sb = new StringBuilder();

        sb.append("START ALL LAYER INFO\n");

        for(int i = 0; i < this.allLayers.size(); i++){
            sb.append("LAYER START\n");
            sb.append(this.allLayers.get(i).toString());
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

    /**
     * Saves the neural network to disk where it can be reloaded at a later time fully constructed.
     * @param filePath The filepath to save the neural network to.
     */
    public void saveModel(String filePath){
        //Check param
        Utility.checkNotNull(filePath);

        String connectionInfo = connectionInfoToString();
        String layerInfo = layerInfoToString();
        String modelInfo = connectionInfo + "\n" + layerInfo;

        Utility.writeStringToFile(filePath, modelInfo);

    }
}