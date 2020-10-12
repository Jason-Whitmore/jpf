public interface Loss{
    public float[] lossVector(float[] yTrue, float[] yPredicted);

    public float[] lossVectorGradient(float[] yTrue, float[] yPredicted);

    public float calculateLossScalar(float[] yTrue, float[] yPredicted);
}