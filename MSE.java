public class MSE implements Loss{

    public float[] lossVector(float[] yTrue, float[] yPredicted){
        float[] r = new float[yTrue.length];

        for(int i = 0; i < r.length; i++){
            float difference = yTrue[i] - yPredicted[i];

            r[i] = difference * difference;
        }

        return r;
    }

    public float[] lossVectorGradient(float[] yTrue, float[] yPredicted){
        float[] r = new float[yTrue.length];

        for(int i = 0; i < r.length; i++){
            float difference = yPredicted[i] - yTrue[i];

            r[i] = 2 * difference;
        }

        return r;
    }

    public float calculateLossScalar(float[] yTrue, float[] yPredicted){
        //TODO: Enforce array sizes

        float[] lossVector = lossVector(yTrue, yPredicted);

        float sum = 0;

        for(int i = 0; i < yTrue.length; i++){
            sum += lossVector[i];
        }

        return sum / yTrue.length;
    }

}