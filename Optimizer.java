import java.util.ArrayList;

public interface Optimizer {
    public ArrayList<float[][]> processGradient(ArrayList<float[][]> rawGradient);
}