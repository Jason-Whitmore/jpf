import java.util.ArrayList;
import jpf.*;

public class Examples{

    private static final String OPTION_STRING = "Arg options:\n" + 
                                                "LinearModel: simplelinear, complexlinear\n" + 
                                                "PolynomialModel: polynomialsin, polynomialoverfit\n" + 
                                                "NeuralNetwork: nnquadratic, nnoverfit, nnbinaryclassification, nnresnet, nnmulticlass, nncomplex";

    
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
        System.out.println("To see if model saving/loading works, the LinearModel will be saved to disk, reloaded, and then the losses will be compared.");

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

        System.out.println("Model loaded. Check loss. Loss from saved model should mostly match to loaded model:");
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

        System.out.println("Model loaded. Loss on dataset on loaded model should match the saved model's loss:");
        System.out.println("Saved model loss: " + afterLoss);
        System.out.println("Loaded model loss: " + loadedModelLoss);
    }


    public static void polynomialOverfit(){
        System.out.println("In this example, polynomial models will be fit on randomly generated data.");
        System.out.println("As higher degree polynomial models are trained, test loss should be much higher than training loss as a result of overfitting.");
        System.out.println("Test and train loss can be expressed as the ratio (test/train loss).");
        System.out.println("A test/train loss ratio should be close to 1 with lower degree models, and should increase as overfitting becomes apparent.\n");

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
        System.out.println("After training, the training loss will be recorded before the model is saved to disk, and compared to the loss from the model loaded from disk.");
        
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
            lossWriter.addRow("" + epoch, "" + trainingLosses[epoch]);
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
        System.out.println("In this example, a demonstration of overfitting will be conducted using a neural network fitting to f(x)=x^2.");
        System.out.println("Both a test and training dataset will be created with outputs from f(x) = x^2 for x in (-10, 10).");
        System.out.println("A standard 2 hidden layer neural network will be created with varying numbers of units in the hidden layers.");
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

        String[] header = {"Number of parameters", "Test / train loss ratio"};
        CSVWriter results = new CSVWriter("nn_overfit_results.csv", header);

        for(int h = 5; h <= 30; h += 5){
            //Create the neural network

            //Perform multiple runs to average out results
            int numRuns = 10;

            float ratio = 0;
            int numParameters = 0;
            for(int run = 0; run < numRuns; run++){
                Input in = new Input(1);
                Dense hidden1 = new Dense(h, new Tanh(), in);
                Dense hidden2 = new Dense(h, new Tanh(), hidden1);
                Dense out = new Dense(1, new Linear(), hidden2);

                NeuralNetwork nn = new NeuralNetwork(in, out);

                nn.fit(trainX, trainY, 2000, 8, 0.01f, new RMSProp(0.001f, 0.9f, 0.0001f), new MSE());

                float trainLoss = Utility.mean(nn.calculateScalarLossBatch(trainX, trainY, new MSE()));
                float testLoss = Utility.mean(nn.calculateScalarLossBatch(testX, testY, new MSE()));

                ratio += testLoss / trainLoss;

                numParameters = nn.getParameterCount();
            }

            ratio /= numRuns;
            
            System.out.println("Num parameters: " + numParameters);
            System.out.println("Test loss / train loss ratio: " + ratio);
            System.out.println();

            results.addRow("" + numParameters, "" + ratio);
        }

        results.writeToFile();

    }


    public static void nnBinaryClassification(){
        System.out.println("In this example a demonstration of a binary classification task will be performed.");
        System.out.println("A neural network will be constructed with an input vector of size 2 and an output vector of size 1 with a sigmoid activation function.");
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


    public static void nnResNet(){
        System.out.println("In this example, two very deep neural networks will be created with a small number of units in the hidden layers.");
        System.out.println("Both neural networks will use the Tanh activation functions for the hidden layers. One of the neural networks will be");
        System.out.println("implemented as a ResNet, which contains \"skip connections\" to a layer closer to the output layer.");
        System.out.println("This is done via an Add layer, which contains no parameters and simply adds the output vectors from other layers together.");
        System.out.println("Both neural networks will be trained on a dataset from the function f(x1,x2) = (x1)^2 - (x2)^2.");
        System.out.println("Training loss will be measured at every epoch and the results will be written to disk.");
        System.out.println("After training, the ResNet neural network will be saved to disk and loaded to test saving and loading functionality.\n");

        System.out.println("Creating the dataset...");
        int n = 1000;
        float[][] trainX = new float[n][];
        float[][] trainY = new float[n][];

        for(int i = 0; i < n; i++){
            float[] x = new float[2];
            x[0] = Utility.getRandomUniform(-5f, 5f);
            x[1] = Utility.getRandomUniform(-5f, 5f);

            float[] y = new float[1];
            y[0] = (x[0] * x[0]) - (x[1] * x[1]);

            trainX[i] = x;
            trainY[i] = y;
        }

        System.out.println("Creating the neural networks...");
        int hiddenSize = 8;

        Input normalIn = new Input(2);
        Dense normalh1 = new Dense(hiddenSize, new Tanh(), normalIn);
        Dense normalh2 = new Dense(hiddenSize, new Tanh(), normalh1);
        Dense normalh3 = new Dense(hiddenSize, new Tanh(), normalh2);
        Dense normalh4 = new Dense(hiddenSize, new Tanh(), normalh3);
        Dense normalOut = new Dense(1, new Linear(), normalh4);

        NeuralNetwork normal = new NeuralNetwork(normalIn, normalOut);

        //Creating the resnet
        Input resnetIn = new Input(2);
        Dense resneth1 = new Dense(hiddenSize, new Tanh(), resnetIn);
        Dense resneth2 = new Dense(hiddenSize, new Tanh(), resneth1);
        Dense resneth3 = new Dense(hiddenSize, new Tanh(), resneth2);
        Dense resneth4 = new Dense(hiddenSize, new Tanh(), resneth3);
        ArrayList<Layer> layerList = new ArrayList<Layer>();

        layerList.add(resneth1);
        layerList.add(resneth2);
        layerList.add(resneth3);
        layerList.add(resneth4);
        Add addLayer = new Add(layerList);

        Dense resnetOut = new Dense(1, new Linear(), addLayer); 

        NeuralNetwork resnet = new NeuralNetwork(resnetIn, resnetOut);

        //Create the writer object
        String[] headers = {"Epoch", "Normal training loss", "Resnet training loss"};
        CSVWriter writer = new CSVWriter("resnet_results.csv", headers);

        System.out.println("Training both neural networks...");

        int epochs = 200;
        for(int e = 0; e < epochs; e++){
            normal.fit(trainX, trainY, 1, 32, 10, new SGD(0.0001f), new MSE());
            resnet.fit(trainX, trainY, 1, 32, 10, new SGD(0.0001f), new MSE());

            float normalLoss = Utility.mean(normal.calculateScalarLossBatch(trainX, trainY, new MSE()));
            float resnetLoss = Utility.mean(resnet.calculateScalarLossBatch(trainX, trainY, new MSE()));

            writer.addRow("" + e, "" + normalLoss, "" + resnetLoss);
        }

        writer.writeToFile();

        System.out.println("Epoch loss data saved to disk. Will now save and load the resnet to test functionality.");

        float beforeSave = Utility.mean(resnet.calculateScalarLossBatch(trainX, trainY, new MSE()));

        resnet.saveModel("resnet_model");

        resnet = null;

        resnet = new NeuralNetwork("resnet_model");

        float afterLoad = Utility.mean(resnet.calculateScalarLossBatch(trainX, trainY, new MSE()));

        System.out.println("Loss before saving: " + beforeSave);
        System.out.println("Loss after loading: " + afterLoad);

        System.out.println("Both losses should be very close or equal to each other.");

    }


    public static void nnMultiClass(){
        System.out.println("In this example, a neural network will be constructed and trained to classify data points into 4 separate classes.");
        System.out.println("Unlike the binary classification task, a softmax output layer will be used to predict the class the input belongs to.");
        System.out.println("The input data will be (x1,x2) coordinates where x1, x2 are in (0,1). The classes where these points belong to are defined");
        System.out.println("as the 4 regions created when 2 lines connect opposite corners of a unit square. Class 0 is the top region,");
        System.out.println("class 1 is the right region, class 2 is the bottom region, class 3 is the left region.");
        System.out.println("Like in the other examples, the neural network will also be saved to disk and loaded to test functionality.");

        System.out.println("Creating the dataset...");
        int n = 1000;
        float[][] trainX = new float[n][];
        float[][] trainY = new float[n][];

        for(int i = 0; i < trainX.length; i++){
            float[] x = new float[2];
            x[0] = Utility.getRandomUniform(0f, 1f);
            x[1] = Utility.getRandomUniform(0f, 1f);

            float[] y = new float[4];

            //Determine which class it belongs to be checking if it's above or below the two lines y = x and y = 1 - x
            boolean aboveX = x[1] > x[0];
            boolean aboveOther = x[1] > 1 - x[0];

            int label = 0;
            if(aboveX && aboveOther){
                label = 0;
            } else if(!aboveX && aboveOther){
                label = 1;
            } else if(!aboveX && !aboveOther){
                label = 2;
            } else {
                label = 3;
            }

            y[label] = 1f;

            trainX[i] = x;
            trainY[i] = y;
        }

        System.out.println("Creating neural network classifier...");
        int numHiddenUnits = 8;

        Input in = new Input(2);
        Dense hidden1 = new Dense(numHiddenUnits, new Tanh(), in);
        Dense hidden2 = new Dense(numHiddenUnits, new Tanh(), hidden1);
        Dense outDense = new Dense(4, new Linear(), hidden2);
        SoftmaxLayer out = new SoftmaxLayer(outDense);

        NeuralNetwork classifier = new NeuralNetwork(in, out);

        System.out.println("Fitting the neural network classifier...");


        classifier.fit(trainX, trainY, 200, 128, 0.1f, new RMSProp(0.001f, 0.9f, 0.00001f), new CrossEntropy());
        

        float trainLoss = Utility.mean(classifier.calculateScalarLossBatch(trainX, trainY, new CrossEntropy()));

        //Calculate the accuracy
        int numCorrect = 0;
        for(int i = 0; i < trainX.length; i++){
            
            float[] output = classifier.predict(trainX[i]);
            int predClass = Utility.argMax(output);
            int trueClass = Utility.argMax(trainY[i]);

            if(predClass == trueClass){
                numCorrect++;
            }
        }

        System.out.println("Training loss: " + trainLoss);
        System.out.println("Accuracy: " + (((float)numCorrect) / trainX.length) + "\n");

        //Save the model to disk and load it
        String filename = "nn_multiclass_model";
        classifier.saveModel(filename);
        classifier = null;
        classifier = new NeuralNetwork(filename);

        float loadLoss = Utility.mean(classifier.calculateScalarLossBatch(trainX, trainY, new CrossEntropy()));

        System.out.println("Loss before saving model: " + trainLoss);
        System.out.println("Loss after loading model from disk: " + loadLoss);
        System.out.println("Both losses should be equal.");

    }


    public static void nnComplex(){
        System.out.println("In this example, an overly complex neural network will be constructed to demonstrate the versatility of the computation graph model");
        System.out.println("which is used to implement neural networks in this package. A neural network will be constructed using 3 separate Input layers and ");
        System.out.println("3 output layers, 2 of which will be Dense and 1 a SoftmaxLayer. The neural network will be trained on dummy data and different loss");
        System.out.println("functions for each layer. The saving and loading from disk functionality will also be tested.");

        System.out.println("Creating the neural network...");

        Input in1 = new Input(3);
        Input in2 = new Input(4);
        Input in3 = new Input(5);

        Dense h1 = new Dense(10, new LeakyReLU(0.1f), in1);
        Dense h2 = new Dense(10, new LeakyReLU(0.1f), in2);
        Dense h3 = new Dense(10, new LeakyReLU(0.1f), in3);

        ArrayList<Layer> layerList = new ArrayList<Layer>();

        layerList.add(h1);
        layerList.add(h2);
        layerList.add(h3);

        Add addLayer = new Add(layerList);

        Dense h4 = new Dense(12, new Tanh(), addLayer);
        Dense h5 = new Dense(10, new LeakyReLU(0.1f), h4);
        Dense h6 = new Dense(10, new LeakyReLU(0.1f), h5);
        Dense outDense1 = new Dense(5, new Sigmoid(), h6);
        SoftmaxLayer outSoftmax = new SoftmaxLayer(h6);
        
        Dense h7 = new Dense(10, new LeakyReLU(0.1f), h4);
        Dense h8 = new Dense(10, new LeakyReLU(0.1f), h7);
        Dense outDense2 = new Dense(5, new Sigmoid(), h8);

        ArrayList<Input> inputLayers = new ArrayList<Input>();
        inputLayers.add(in1);
        inputLayers.add(in2);
        inputLayers.add(in3);

        ArrayList<Layer> outputLayers = new ArrayList<Layer>();
        outputLayers.add(outDense1);
        outputLayers.add(outSoftmax);
        outputLayers.add(outDense2);


        NeuralNetwork nn = new NeuralNetwork(inputLayers, outputLayers);

        System.out.println("Creating the dummy training data.");

        int n = 1000;
        float[][][] trainX = new float[n][][];
        float[][][] trainY = new float[n][][];

        for(int i = 0; i < trainX.length; i++){
            float[][] x = new float[3][];
            float[][] y = new float[3][];

            x[0] = Utility.getRandomUniform(0f, 1f, 3);
            x[1] = Utility.getRandomUniform(0f, 1f, 4);
            x[2] = Utility.getRandomUniform(0f, 1f, 5);

            y[0] = Utility.getRandomUniform(0f, 5f, 5);
            
            y[1] = new float[10];
            y[1][(int)Utility.getRandomUniform(0, 10)] = 1f;

            y[2] = Utility.getRandomUniform(0f, 10f, 5);


            trainX[i] = x;
            trainY[i] = y;
        }


        //Create separate losses for each output layer
        ArrayList<Loss> losses = new ArrayList<Loss>();
        losses.add(new MSE());
        losses.add(new MSE());
        losses.add(new CrossEntropy());

        float preFitLoss = Utility.mean(nn.calculateScalarLossBatch(trainX, trainY, losses));
        System.out.println("Loss before fitting: " + preFitLoss);

        System.out.println("Fitting the neural network...");
        nn.fit(trainX, trainY, 100, 32, 0.1f, new RMSProp(), losses);

        float trainLoss = Utility.mean(nn.calculateScalarLossBatch(trainX, trainY, losses));

        System.out.println("Neural network fitting complete.\nTraining loss before saving: " + trainLoss);
        String path = "nn_complex_model";

        nn.saveModel(path);

        nn = null;

        nn = new NeuralNetwork(path);

        float loadLoss = Utility.mean(nn.calculateScalarLossBatch(trainX, trainY, losses));

        System.out.println("Training loss after loading: " + loadLoss);
        System.out.println("The before saving and after loading losses should be very similar or equal.");
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
            
            case "nnresnet":
                nnResNet();
                break;

            case "nnmulticlass":
                nnMultiClass();
                break;

            case "nncomplex":
                nnComplex();
                break;

            default:
                System.out.println("Example string not recognized.");
                System.out.println(Examples.OPTION_STRING);
                break;
        }
    }
}