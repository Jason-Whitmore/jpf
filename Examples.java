public class Examples{

    private static final String OPTION_STRING = "Arg options: simplelinear, complexlinear, polynomialsin";

    
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
        System.out.println("Training complete. Loss before training should be greater than loss after training:");
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


    public static void polynomialSin(){
        System.out.println("In this example, a polynomial model will be fit to data produced from the sin(x) function in order to find a fast approximation.");

        //Generate the data
        System.out.println("Creating the dataset...");

        int numSamples = 1000;

        float[][] trainingInputs = new float[numSamples][];
        float[][] trainingOutputs = new float[numSamples][];

        for(int i = 0; i < numSamples; i++){
            float[] x = new float[1];

            x[0] = (float)(Math.random() * 3f);

            float[] y = new float[1];

            y[0] = (float)(Math.sin((double) x[0]));

            trainingInputs[i] = x;
            trainingOutputs[i] = y;
        }

        //Create the polynomial model
        int degree = 1;
        System.out.println("Creating a polynomial model of degree " + degree);

        PolynomialModel model = new PolynomialModel(1, 1, degree);

        //Before training, evaluate loss
        float beforeLoss = model.calculateLoss(trainingInputs, trainingOutputs, new MSE());

        //Train the polynomial model
        System.out.println("Training polynomial model...");
        model.fit(trainingInputs, trainingOutputs, 200000, 32, 0.01f, new RMSProp(), new MSE());

        //Calculate loss after training
        float afterLoss = model.calculateLoss(trainingInputs, trainingOutputs, new MSE());

        System.out.println("Training complete. Loss from before training should be larger than loss after training:");
        System.out.println("Loss before training: " + beforeLoss);
        System.out.println("Loss after training: " + afterLoss);

        //Print out the learned equation for approximating sin(x)
        float[] coefficients = new float[degree + 1];
        for(int i = 0; i < degree; i++){
            coefficients[i] = model.getParameters().get(0)[0][i];
        }

        coefficients[coefficients.length - 1] = model.getParameters().get(1)[0][0];

        //Print function with highest order terms first
        System.out.print("Learned polynomial is: f(x) = ");
        for(int i = coefficients.length - 2; i >= 1; i--){
            System.out.print(coefficients[i] + "x^" + (i + 1) + " + ");
        }

        System.out.print(coefficients[0] + "x + ");
        System.out.print(coefficients[coefficients.length - 1] + "\n");

        //Test the saving/loading methods of the polynomial model
        String filePath = "polynomial_model_saved";
        System.out.println("Now saving the model to file: " + filePath);
        model.saveModel(filePath);

        System.out.println("Model saved. Now loading the model from file...");
        model = null;
        model = new PolynomialModel(filePath);

        //Get the loss on the dataset. Should match old model's loss
        float loadedModelLoss = model.calculateLoss(trainingInputs, trainingOutputs, new MSE());

        System.out.println("Model loaded. Loss on dataset on loaded model should match the previous model's loss:");
        System.out.println("Old model loss: " + afterLoss);
        System.out.println("Loaded model loss: " + loadedModelLoss);
    }


    public void polynomialOverfit(){
        System.out.println("In this example, polynomial models will be fit on randomly generated data.");
        System.out.println("As higher degree polynomial models are trained, test loss should be much higher than training loss as a result of overfitting.");
        System.out.println("Test and train loss can be expressed as the fraction (test/train).");
        System.out.println("A test/train ratio should be close to 1 with lower degree models, and should increase as overfitting becomes apparent.\n");

        System.out.println("Generating data...");
        
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
        
            case "complexlinear":
                complexLinear();
                break;
                
            case "polynomialsin":
                polynomialSin();
                break;

            default:
                break;
        }
    }
}