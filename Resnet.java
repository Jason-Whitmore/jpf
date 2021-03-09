import java.util.ArrayList;

public class Resnet {
    public static void main(String[] args){

        int numEpochs = 400;

        int hiddenLayerSize = 8;

        Input in = new Input(2);

        Dense hidden1 = new Dense(hiddenLayerSize, new Tanh(), in);

        Dense hidden2 = new Dense(hiddenLayerSize, new Tanh(), hidden1);

        Dense hidden3 = new Dense(hiddenLayerSize, new Tanh(), hidden2);

        ArrayList<Layer> addLayerInputs = new ArrayList<Layer>();
        addLayerInputs.add(hidden1);
        addLayerInputs.add(hidden2);
        addLayerInputs.add(hidden3);

        Add addLayer = new Add(addLayerInputs);

        Dense output = new Dense(1, new Linear(), addLayer);

        NeuralNetwork resnet = new NeuralNetwork(in, output);

        //create a traditional neural network with 3 hidden layers.

        Input nn_in = new Input(2);

        Dense nn_hidden1 = new Dense(hiddenLayerSize, new Tanh(), nn_in);

        Dense nn_hidden2 = new Dense(hiddenLayerSize, new Tanh(), nn_hidden1);

        Dense nn_hidden3 = new Dense(hiddenLayerSize, new Tanh(), nn_hidden2);

        Dense nn_output = new Dense(1, new Linear(), nn_hidden3);

        NeuralNetwork nn = new NeuralNetwork(nn_in, nn_output);


        int datasetSize = 1000;
        float[][] trainX = new float[datasetSize][2];
        float[][] trainY = new float[datasetSize][1];

        for(int i = 0; i < datasetSize; i++){
            //Generate a random tuple of data in range (0,1)
            trainX[i][0] = (float)(Math.random());
            trainX[i][1] = (float)(Math.random());

            //Map the input vector to the saddle function
            trainY[i][0] = (float)(Math.pow((trainX[i][0]), 2.0)) - (float)(Math.pow((trainX[i][1]), 2.0));
        }


        float[] resnet_losses = new float[numEpochs];
        float[] nn_losses = new float[numEpochs];

        //Train the resnet and normal neural network using the input data
        for(int e = 0; e < numEpochs; e++){
            resnet.fit(trainX, trainY, 1, 32, 0.1f, new SGD(0.01f), new MSE());
            nn.fit(trainX, trainY, 1, 32, 0.1f, new SGD(0.01f), new MSE());

            resnet_losses[e] = resnet.calculateLoss(trainX, trainY, new MSE());
            nn_losses[e] = nn.calculateLoss(trainX, trainY, new MSE());
        }

        //Write training results to file

        String[] headers = new String[3];
        headers[0] = "Epoch";
        headers[1] = "NN loss";
        headers[2] = "Resnet loss";
        CSVWriter writer = new CSVWriter("resnet_nn_losses.csv", headers);

        for(int e = 0; e < numEpochs; e++){
            String[] row = new String[3];

            row[0] = "" + e;
            row[1] = "" + nn_losses[e];
            row[2] = "" + resnet_losses[e];

            writer.addRow(row);
        }

        writer.writeToFile();

        System.out.println("Both models trained. Check csv file for output data");
        
    }

}