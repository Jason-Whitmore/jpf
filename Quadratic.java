import java.util.ArrayList;

public class Quadratic {
    public static void main(String[] args){

        //Neural network variables
        int numEpochs = 1000;
        int hiddenLayerSize = 32;

        //Dataset variables
        int numSamples = 1000;
        float dataMin = -3f;
        float dataMax = 3f;

        //Output variables
        float outputStepSize = 0.01f;


        float[][] trainX = new float[numSamples][1];
        float[][] trainY = new float[numSamples][1];

        int index = 0;
        for(float i = 0; i < numSamples; i++){

            float x = Utility.getRandomUniform(-3f, 3f);
            trainX[index][0] = x;
            trainY[index][0] = x * x;
            index++;
        }




        Input in = new Input(1);

        Dense hidden1 = new Dense(hiddenLayerSize, new Tanh(), in);
        Dense hidden2 = new Dense(hiddenLayerSize, new Tanh(), hidden1);

        Dense out = new Dense(1, new Linear(), hidden2);

        NeuralNetwork nn = new NeuralNetwork(in, out);

        ArrayList<float[]> outputs = new ArrayList<float[]>();
        ArrayList<Integer> epochNumber = new ArrayList<Integer>();

        for(int e = 0; e < numEpochs; e++){
            //train for some epochs

            nn.fit(trainX, trainY, 1, 32, 10f, new RMSProp(), new MSE());

            //Make predictions
            //Record the current epoch
            epochNumber.add(e);

            //Make predictions on datapoints and record the results
            float[] epochOutputs = new float[(int)((dataMax - dataMin) / outputStepSize)];
            index = 0;
            for(float x = dataMin; x <= dataMax; x += outputStepSize){
                float[] inputVector = new float[1];
                inputVector[0] = x;
                float output = nn.predict(inputVector)[0];
                epochOutputs[index] = output;
                index++;
            }

            outputs.add(epochOutputs);

            

            float loss = nn.calculateLoss(trainX, trainY, new MSE());

            System.out.println("Epoch " + e + " complete. Training loss: " + loss);
        }

        //Write results into a csv file
        String[] columnHeaders = new String[11];

        columnHeaders[0] = "x";
        for(int i = 1; i < columnHeaders.length; i++){
            columnHeaders[i] = "" + (i - 1) * 100;
        }

        CSVWriter writer = new CSVWriter("Quadratic.csv", columnHeaders);
        
    }
}