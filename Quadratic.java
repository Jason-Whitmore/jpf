import java.util.ArrayList;

public class Quadratic {
    public static void main(String[] args){

        int numSamples = 1000;
        float[][] trainX = new float[numSamples][1];
        float[][] trainY = new float[numSamples][1];

        int index = 0;
        for(float x = 0; x < 10; x += 0.01){
            trainX[index][0] = x;
            trainY[index][0] = x * x;
            index++;
        }




        Input in = new Input(1);

        Dense hidden1 = new Dense(64, new Tanh(), in);
        Dense hidden2 = new Dense(64, new Tanh(), hidden1);

        Dense out = new Dense(1, new Linear(), hidden2);

        NeuralNetwork nn = new NeuralNetwork(in, out);

        float[][] yPred = new float[trainX.length][10];

        for(int e = 0; e < 1000; e += 100){
            //train for some epochs

            nn.fit(trainX, trainY, 100, 32, 0.1f, new RMSProp(), new MSE());

            //Make predictions
            for(int i = 0; i < trainX.length; i++){
                yPred[i][e / 100] = nn.predict(trainX[i])[0];
            }
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