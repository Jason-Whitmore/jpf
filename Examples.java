

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