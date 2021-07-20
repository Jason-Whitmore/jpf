public class Examples{

    private static final String OPTION_STRING = "Arg options:\n" + 
                                                "LinearModel: simplelinear, complexlinear\n" + 
                                                "PolynomialModel: polynomialsin, polynomialoverfit\n" + 
                                                "NeuralNetwork: nnquadratic, nnoverfit, nnbinaryclassification";

    
    private static void simpleLinear(){
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

        int numSamples = 1024;

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
        int degree = 2;
        System.out.println("Creating a polynomial model of degree " + degree);

        PolynomialModel model = new PolynomialModel(1, 1, degree);

        //Before training, evaluate loss
        float beforeLoss = model.calculateLoss(trainingInputs, trainingOutputs, new MSE());

        //Train the polynomial model
        System.out.println("Training polynomial model...");
        model.fit(trainingInputs, trainingOutputs, 10000, 1, 0.1f, new RMSProp(0.001f, 0.9f, 0.0001f), new MSE());

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


    public static void polynomialOverfit(){
        System.out.println("In this example, polynomial models will be fit on randomly generated data.");
        System.out.println("As higher degree polynomial models are trained, test loss should be much higher than training loss as a result of overfitting.");
        System.out.println("Test and train loss can be expressed as the ratio (test/train loss).");
        System.out.println("A test/train ratio should be close to 1 with lower degree models, and should increase as overfitting becomes apparent.\n");

        System.out.println("Generating data...");
        int trainingDataSize = 10;
        float[][] trainingInputs = new float[trainingDataSize][1];
        float[][] trainingOutputs = new float[trainingDataSize][1];

        int testDataSize = 100;
        float[][] testingInputs = new float[testDataSize][1];
        float[][] testingOutputs = new float[testDataSize][1];

        for(int i = 0; i < trainingDataSize; i++){
            float x = Utility.getRandomUniform(-10f, 10f);
            float y = x * x;

            trainingInputs[i][0] = x;
            trainingOutputs[i][0] = y;
        }

        for(int i = 0; i < testDataSize; i++){
            float x = Utility.getRandomUniform(-10f, 10f);
            float y = x * x;

            testingInputs[i][0] = x;
            testingOutputs[i][0] = y;
        }

        System.out.println("Data generated. Now training polynomial models.");

        for(int degree = 1; degree <= 15; degree += 3){
            PolynomialModel model = new PolynomialModel(1,1, degree);

            model.fit(trainingInputs, trainingOutputs, 1000000, 1, 0.1f, new RMSProp(0.00001f, 0.9f, 0.001f), new MSE());

            float trainingLoss = model.calculateLoss(trainingInputs, trainingOutputs, new MSE());
            float testingLoss = model.calculateLoss(testingInputs, testingOutputs, new MSE());

            System.out.println("Degree: " + degree + ", test/train loss ratio: " + (testingLoss / trainingLoss));
        }
    }


    public static void nnQuadratic(){
        System.out.println("In this example, a neural network will be created to fit f(x) = x^2 for x in [-10, 10].");
        System.out.println("Over the course of training, the training loss will be recorded as well as the neural network output after each epoch");
        System.out.println("After training, the training loss will be recorded before the model is saved to disk, deallocated, and recreated from disk.");
        
        //Create dataset of f(x) = x^2 for x in [-10, 10]
        int n = 1000;

        System.out.println("Creating training dataset of size " + n);
        float[][] trainX = new float[n][];
        float[][] trainY = new float[n][];

        for(int i = 0; i < trainX.length; i++){
            float[] x = new float[1];
            float[] y = new float[1];

            x[0] = Utility.getRandomUniform(-10f, 10f);
            y[0] = x[0] * x[0];

            trainX[i] = x;
            trainY[i] = y;
        }

        System.out.println("Creating neural network");
        int hiddenUnits = 16;

        Input in = new Input(1);
        Dense h1 = new Dense(hiddenUnits, new Tanh(), in);
        Dense h2 = new Dense(hiddenUnits, new Tanh(), h1);
        Dense out = new Dense(1, new Linear(), h2);

        NeuralNetwork nn = new NeuralNetwork(in, out);

        //Create the arrays to record function output and loss per epoch
        int numEpochs = 20;
        float[] trainingLosses = new float[numEpochs];
        float[][] functionOutput = new float[numEpochs][100];

        System.out.println("Training started...");

        float delta = 20f / functionOutput[0].length;

        for(int e = 0; e < numEpochs; e++){
            //Calculate train loss
            float trainLoss = Utility.mean(nn.calculateScalarLossBatch(trainX, trainY, new MSE()));
            System.out.println("Train loss before epoch " + e + ": " + trainLoss);
            trainingLosses[e] = trainLoss;

            //Calculate function output
            for(int i = 0; i < functionOutput[0].length; i++){
                float[] x = new float[1];
                x[0] = (i * delta) - 10f;
                functionOutput[e][i] = nn.predict(x)[0];
            }


            nn.fit(trainX, trainY, 1, 32, 0.1f, new RMSProp(0.01f, 0.9f, 0.000001f), new MSE());
        }

        System.out.println("Training complete. Writing results to disk...");

        String[] lossHeader = {"Epoch", "Training loss"};
        CSVWriter lossWriter = new CSVWriter("nn_quadratic_loss.csv", lossHeader);
        for(int epoch = 0; epoch < trainingLosses.length; epoch++){
            String[] row = {"" + epoch, "" + trainingLosses[epoch]};
            lossWriter.addRow(row);
        }
        lossWriter.writeToFile();
        System.out.println("Training loss function data written to disk. Check for nn_quadratic_loss.csv where Examples.java is.");

        String[] outputHeader = new String[numEpochs + 1];
        outputHeader[0] = "x";
        for(int i = 1; i < outputHeader.length; i++){
            outputHeader[i] = "f(x) before epoch " + (i - 1);
        }
        CSVWriter outputWriter = new CSVWriter("nn_quadratic_output.csv", outputHeader);
        for(int i = 0; i < functionOutput[0].length; i++ ){
            float x = (i * delta) - 10f;
            String[] row = new String[outputHeader.length];
            row[0] = "" + x;
            for(int epoch = 0; epoch < numEpochs; epoch++){
                row[epoch + 1] = functionOutput[epoch][i] + "";
            }

            outputWriter.addRow(row);
        }

        outputWriter.writeToFile();
        System.out.println("Function output data written to disk. Check for nn_quadratic_output.csv where Examples.java is.\n");
        

        System.out.println("Now checking to see if saving and loading the neural network works. Will record train loss, save model, reload it, and check train loss again");

        float lossBefore = Utility.mean(nn.calculateScalarLossBatch(trainX, trainY, new MSE()));
        String fileName = "nn_quadratic_model";
        nn.saveModel(fileName);
        nn = null;
        nn = new NeuralNetwork(fileName);
        float lossAfter = Utility.mean(nn.calculateScalarLossBatch(trainX, trainY, new MSE()));

        System.out.println("Test loss before saving model: " + lossBefore);
        System.out.println("Test loss after saving model: " + lossAfter);
        System.out.println("Both losses should be very similar or the same.");
    }

    public static void nnOverfit(){
        System.out.println("In this example, a demonstration of overfitting will be conducted using a neural network fitting to f(x)=x^2");
        System.out.println("Both a test and training dataset will be created with outputs from f(x) = x^2 for x in (-10, 10)");
        System.out.println("The standard 2 hidden neural network will be created with varying numbers of units in the hidden layers.");
        System.out.println("After a neural network is trained, the loss will be collected for both the training and testing datasets.");
        System.out.println("At the end of the example, the results will be saved to nn_overfit_results.csv so that they can be graphed.\n");

        System.out.println("Creating the datasets...");
        int n = 30;

        float[][] trainX = new float[n][];
        float[][] trainY = new float[n][];

        float[][] testX = new float[n][];
        float[][] testY = new float[n][];

        for(int i = 0; i < trainX.length; i++){
            float[] x = new float[1];
            x[0] = Utility.getRandomUniform(-10f, 10f);

            float[] y = new float[1];
            y[0] = x[0] * x[0];

            trainX[i] = x;
            trainY[i] = y;

            x = new float[1];
            y = new float[1];
            x[0] = Utility.getRandomUniform(-10f, 10f);
            y[0] = x[0] * x[0];

            testX[i] = x;
            testY[i] = y;
        }

        System.out.println("Creating and training the neural networks...");

        String[] header = {"Number of parameters", "Training loss", "Testing loss"};
        CSVWriter results = new CSVWriter("nn_overfit_results.csv", header);

        for(int h = 20; h <= 120; h += 20){
            //Create the neural network

            //Perform multiple runs to average out results
            int numRuns = 10;

            float trainLoss = 0;
            float testLoss = 0;
            int numParameters = 0;
            for(int run = 0; run < numRuns; run++){
                Input in = new Input(1);
                Dense hidden1 = new Dense(h, new Tanh(), in);
                Dense hidden2 = new Dense(h, new Tanh(), hidden1);
                Dense out = new Dense(1, new Linear(), hidden2);

                NeuralNetwork nn = new NeuralNetwork(in, out);

                nn.fit(trainX, trainY, 10000, 8, 10f, new SGD(), new MSE());

                trainLoss += Utility.mean(nn.calculateScalarLossBatch(trainX, trainY, new MSE()));
                testLoss += Utility.mean(nn.calculateScalarLossBatch(testX, testY, new MSE()));
                numParameters = nn.getParameterCount();
            }

            trainLoss /= numRuns;
            testLoss /= numRuns;
            
            System.out.println("Num parameters: " + numParameters);
            System.out.println("Training loss: " + trainLoss);
            System.out.println("Testing loss: " + testLoss);
            System.out.println();

            String[] newRow = {"" + numParameters, "" + trainLoss, "" + testLoss};

            results.addRow(newRow);
        }

        results.writeToFile();

    }


    public static void nnBinaryClassification(){
        System.out.println("In this example a demonstration of a binary classification task will be performed.");
        System.out.println("A neural network will be constructed with an input vector of size two and an output vector of size 1 with a sigmoid activation function");
        System.out.println("The training input data will be (x1,x2) coordinates with x1, x2 in range (0,1). Each (x1,x2) pair will be considered to be a member of 2 classes.");
        System.out.println("(x1,x2) coordinates that lie inside of a circle that has a center at (0.5,0.5) and a radius of 0.5 will belong to class 0.");
        System.out.println("All other (x1,x2) coordinates will belong to class 1.\n");

        System.out.println("Creating the dataset...");
        int n = 1000;

        float[][] trainX = new float[n][];
        float[][] trainY = new float[n][];

        for(int i = 0; i < trainX.length; i++){
            float[] x = new float[2];
            float[] y = new float[1];

            x[0] = Utility.getRandomUniform(0, 1f);
            x[1] = Utility.getRandomUniform(0, 1f);

            float distance = (float)Math.sqrt(Math.pow(x[0] - 0.5, 2) + Math.pow(x[0] - 0.5, 2));
            
            if(distance < 0.5){
                y[0] = 0;
            } else {
                y[0] = 1f;
            }

            trainX[i] = x;
            trainY[i] = y;
        }

        System.out.println("Creating the neural network classifier...");

        int hiddenSize = 8;

        Input in = new Input(2);
        Dense h1 = new Dense(hiddenSize, new LeakyReLU(0.1f), in);
        Dense h2 = new Dense(hiddenSize, new LeakyReLU(0.1f), h1);
        Dense out = new Dense(1, new Sigmoid(), h2);

        NeuralNetwork nn = new NeuralNetwork(in, out);

        System.out.println("Fitting the neural network.");
        nn.fit(trainX, trainY, 500, 32, 0.1f, new RMSProp(), new CrossEntropy());

        System.out.println("Evaluating the neural network.");
        float trainLoss = Utility.mean(nn.calculateScalarLossBatch(trainX, trainY, new CrossEntropy()));

        float correctPredictions = 0;
        for(int i = 0; i < trainX.length; i++){
            float yPred = nn.predict(trainX[i])[0];

            if(trainY[i][0] == 0 && yPred < 0.5f){
                correctPredictions += 1;
            } else if(trainY[i][0] == 1 && yPred > 0.5f){
                correctPredictions += 1;
            }
        }

        float accuracy = correctPredictions / trainX.length;

        System.out.println("Training loss: " + trainLoss);
        System.out.println("Accuracy (correct predictions / number of samples): " + accuracy);
    }

    public static void main(String[] args){

        if(args.length != 1){
            System.out.println(OPTION_STRING);
            System.exit(0);
        }

        runExample(args[0]);
    }

    public static void runExample(String exampleName){

        exampleName = exampleName.toLowerCase();

        switch (exampleName) {
            case "simplelinear":
                simpleLinear();
                break;
        
            case "complexlinear":
                complexLinear();
                break;
                
            case "polynomialsin":
                polynomialSin();
                break;

            case "polynomialoverfit":
                polynomialOverfit();
                break;

            case "nnquadratic":
                nnQuadratic();
                break;

            case "nnoverfit":
                nnOverfit();
                break;

            case "nnbinaryclassification":
                nnBinaryClassification();
                break;

            default:
                System.out.println("Example string not recognized.");
                System.out.println(Examples.OPTION_STRING);
                break;
        }
    }
}