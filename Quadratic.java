import java.util.ArrayList;

public class Quadratic {
    public static void main(String[] args){

        int numSamples = 1000;
        int numEpochs = 1000;
        int hiddenLayerSize = 16;
        int epochsPerOutput = 100;

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
        ArrayList<Integer> epochOutput = new ArrayList<Integer>();

        for(int e = 0; e < numEpochs; e++){
            //train for some epochs

            nn.fit(trainX, trainY, 1, 32, 10f, new RMSProp(), new MSE());

            //Make predictions
            if(e % epochsPerOutput == 0){
                //Record the current epoch
                epochOutput.add(e);

                //Make predictions on datapoints and record the results
                
            }

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

        for(int r = 0; r < trainX.length; r++){
            ArrayList<String> rowString = new ArrayList<String>();
            rowString.add("" + trainX[r][0]);

            for(int c = 0; c < yPred[0].length; c++){
                rowString.add("" + yPred[r][c]);
            }

            writer.addRow(rowString.toArray(new String[rowString.size()]));
        }

        writer.writeToFile();
        
    }
}