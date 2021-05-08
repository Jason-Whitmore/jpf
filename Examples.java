public class Examples{

    private static final String OPTION_STRING = "";

    
    private static void SimpleLinear(){
        System.out.println("Starting Simple Linear Model example: Fitting model to f(x) = 2x - 1");

        //Creating dataset for f(x) = 2x - 1
        System.out.println("Creating dataset...");

        int numSamples = 1000;
        float[][] trainingInputs = new float[numSamples][];
        float[][] trainingOutputs = new float[numSamples][];

        for(int i = 0; i < numSamples; i++){
            float[] x = new float[1];
            x[0] = Utility.getRandomUniform(-10f, 10f);

            float[] y = new float[1];
            y[0] = (2 * x[0]) - 1;

            trainingInputs[i] = x;
            trainingOutputs[i] = y;
        }

        //Create the model
        System.out.println("Creating the linear model...");

        LinearModel model = new LinearModel(1, 1);

        //Train the model
        System.out.println("Training the linear model...");

        float untrainedLoss = model.calculateLoss(trainingInputs, trainingOutputs, new MSE());

        model.fit(trainingInputs, trainingOutputs, 1000, 32, 0.01f, new SGD(0.01f), new MSE());

        //Calculate the loss

        float trainedLoss = model.calculateLoss(trainingInputs, trainingOutputs, new MSE());

        //Print out training loss progress
        System.out.println("Training complete. Loss after training should be greater than loss after training:");
        System.out.println("Loss before training: " + untrainedLoss);
        System.out.println("Loss after training: " + trainedLoss);
    }

    public static void complexLinear(){

        System.out.println("Starting complex LinearModel example.");
        System.out.println("Will generate data from a randomized transformation and bias matrix, then train a linear model on it.");
        System.out.println("After training, the learned parameters will then be compared to the original data generation matricies to determine if training worked.");
        System.out.println("To see if model saving/loading works, the LinearModel will be saved to disk, reloaded, and then compared outputs.");

        int numInputs = 3;
        int numOutputs = 3;

        float[][] transformationMatrix = new float[numOutputs][numInputs];
        Utility.initializeUniform(transformationMatrix, -1f, 1f);

        float[][] biasMatrix = new float[numOutputs][1];
        Utility.initializeUniform(biasMatrix, -1f, 1f);

        //Generate data
        System.out.println("Generating data...");
        int numSamples = 1000;

        float[][] trainingInputs = new float[numSamples][];
        float[][] trainingOutputs = new float[numSamples][];

        for(int i = 0; i < numSamples; i++){
            float[] x = new float[numInputs];
            Utility.initializeUniform(x, -10f, 10f);
            
            float[][] postTransformMatrix = LinearAlgebra.matrixMultiply(transformationMatrix, LinearAlgebra.arrayToMatrix(x));

            float[][] postAddMatrix = LinearAlgebra.matrixAdd(postTransformMatrix, biasMatrix);

            float[] y = LinearAlgebra.matrixToArray(postAddMatrix);

            trainingInputs[i] = x;
            trainingOutputs[i] = y;
        }

        //Create LinearModel and train
        System.out.println("Creating LinearModel and training...");
        LinearModel model = new LinearModel(numInputs, numOutputs);

        float startLoss = model.calculateLoss(trainingInputs, trainingOutputs, new MSE());

        model.fit(trainingInputs, trainingOutputs, 1000, 32, 0.1f, new SGD(0.001f), new MSE());

        float endLoss = model.calculateLoss(trainingInputs, trainingOutputs, new MSE());

        System.out.println("Model trained. End loss should be smaller than start loss:");
        System.out.println("Start loss: " + startLoss);
        System.out.println("End loss: " + endLoss);

        //Save model and reload
        String filePath = "ComplexLinearModel";

        System.out.println("Saving trained model to disk...");
        model.saveModel(filePath);

        model = null;

        System.out.println("Loading model from disk...");
        model = new LinearModel(filePath);

        System.out.println("Model loaded. Check loss. Loss from model should mostly match to loaded model:");
        System.out.println("Loss from trained model: " + endLoss);

        float loadedModelLoss = model.calculateLoss(trainingInputs, trainingOutputs, new MSE());
        
        System.out.println("Loss from loaded model: " + loadedModelLoss);

    }

    public static void main(String[] args){

        if(args.length != 1){
            System.out.println(OPTION_STRING);
        }

        runExample(args[0]);
    }

    public static void runExample(String exampleName){
        exampleName = exampleName.toLowerCase();

        switch (exampleName) {
            case "simplelinear":
                SimpleLinear();
                break;
        
            default:
                break;
        }
    }
}