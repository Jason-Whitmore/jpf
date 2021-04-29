


public class CrossEntropy implements Loss{

    public CrossEntropy(){

    }

    private float CELoss(float yTrue, float yPredicted){
        if(yTrue == 1){
            return (float)(-Math.log((yPredicted)));
        } else {
            return (float)(-Math.log((1f - yPredicted)));
        }
    }

    private float CELossPrime(float yTrue, float yPredicted){
        return 0;
    }

    public float[] calculateLossVector(float[] yTrue, float[] yPredicted){
        float[] r = new float[yTrue.length];

        for(int i = 0; i < r.length; i++){
            r[i] = this.CELoss(yTrue[i], yPredicted[i]);
        }

        return r;
    }

    public float[] calculateLossVectorGradient(float[] yTrue, float[] yPredicted){
        float[] r = new float[yTrue.length];

        for(int i = 0; i < r.length; i++){
            r[i] = this.CELossPrime(yTrue[i], yPredicted[i]);
        }

        return r;
    }

    public float calculateLossScalar(float[] yTrue, float[] yPredicted){
        float[] lossVector = this.calculateLossVector(yTrue, yPredicted);

        return Utility.mean(lossVector);
    }
}