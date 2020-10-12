import java.util.ArrayList;
import java.util.Collections;

public class Utility{

    public static ArrayList<ArrayList<Integer>> getMinibatchIndicies(int n, int minibatchSize){

        //get the indicies
        ArrayList<Integer> indicies = new ArrayList<Integer>(n);

        for(int i = 0; i < n; i++){
            indicies.add(i);
        }

        //randomly shuffle the indicies
        Collections.shuffle(indicies);

        //Separate the indicies into minibatches

        ArrayList<ArrayList<Integer>> r = new ArrayList<ArrayList<Integer>>();

        ArrayList<Integer> minibatchIndicies = new ArrayList<Integer>();
        for(int i = 0; i < indicies.size(); i++){

            if(minibatchIndicies.size() >= minibatchSize){
                r.add(new ArrayList<Integer>(minibatchIndicies));
                minibatchIndicies = new ArrayList<Integer>();
            } else {
                minibatchIndicies.add(indicies.get(i));
            }
        }


        if(minibatchIndicies.size() > 0){
            r.add(new ArrayList<Integer>(minibatchIndicies));
        }

        return r;
    }

    public static void clip(float[] array, float min, float max){
        for(int i = 0; i < array.length; i++){
            if(array[i] < min){
                array[i] = min;
            } else if(array[i] > max){
                array[i] = max;
            }
        }
    }

    public static void clip(float[][] array, float min, float max){
        for(int i = 0; i < array.length; i++){
            clip(array[i], min, max);
        }
    }

    public static void clip(ArrayList<float[][]> arrays, float min, float max){
        for(int i = 0; i < arrays.size(); i++){
            clip(arrays.get(i), min, max);
        }
    }

    public static void clearArray(float[] a){
        for(int i = 0; i < a.length; i++){
            a[i] = 0;
        }
    }

    public static void clearArray(float[][] a){
        for(int i = 0; i < a.length; i++){
            clearArray(a[i]);
        }
    }

    public static void scaleArray(float[] a, float scalar){
        for(int i = 0; i < a.length; i++){
            a[i] *= scalar;
        }
    }

    public static void scaleArray(float[][] a, float scalar){
        for(int i = 0; i < a.length; i++){
            scaleArray(a[i], scalar);
        }
    }

    public static ArrayList<float[][]> cloneArrays(ArrayList<float[][]> arrays){
        if(arrays == null){
            return null;
        }

        ArrayList<float[][]> r = new ArrayList<float[][]>(arrays.size());

        for(int i = 0; i < arrays.size(); i++){
            int numRows = arrays.get(i).length;
            int numCols = arrays.get(i)[0].length;

            r.add(new float[numRows][numCols]);
        }


        return r;
    }

    public static void addArray(float[][] dest, float[][] b, float scalar){
        for(int r = 0; r < dest.length; r++){
            for(int c = 0; c < dest[r].length; c++){
                dest[r][c] = dest[r][c] + (b[r][c] * scalar);
            }
        }
    }

    public static void addGradient(ArrayList<float[][]> gradient, ArrayList<float[][]> newGradient, float scalar){
        for(int i = 0; i < gradient.size(); i++){
            addArray(gradient.get(i), newGradient.get(i), scalar);
        }
    }



    public static void scaleGradient(ArrayList<float[][]> gradient, float scalar){
        for(int i = 0; i < gradient.size(); i++){
            scaleArray(gradient.get(i), scalar);
        }
    }

    public static String arrayToString(float[] array){
        if(array.length == 0){
            return "[]";
        }

        StringBuilder sb = new StringBuilder();

        sb.append("[");

        for(int i = 0; i < array.length - 1; i++){
            sb.append(array[i]);
            sb.append(", ");
        }

        sb.append(array[array.length - 1]);
        sb.append("]");

        return new String(sb);
    }

    public static String arrayToString(float[][] array){
        StringBuilder sb = new StringBuilder();

        sb.append("[");
        sb.append(arrayToString(array[0]));

        for(int i = 1; i < array.length; i++){
            sb.append(arrayToString(array[i]));
            sb.append("\n");
        }

        sb.append("]");

        return new String(sb);
    }

    public static String arraysToString(ArrayList<float[][]> arrays){

        if(arrays.size() == 0){
            return "[]";
        }
        StringBuilder sb = new StringBuilder();

        sb.append("[");
        sb.append(arrayToString(arrays.get(0)));

        for(int i = 1; i < arrays.size() - 1; i++){
            sb.append(arrayToString(arrays.get(i)));
            sb.append(",\n");
        }

        sb.append(arrayToString(arrays.get(arrays.size() - 1)));
        sb.append("]");

        return new String(sb);
    }
}