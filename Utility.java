import java.util.ArrayList;
import java.util.Collections;

public class Utility{

    public static ArrayList<ArrayList<Integer>> getMinibatchIndicies(int n, int minibatchSize){

        //get the indicies
        ArrayList<Integer> indicies = new ArrayList<Integer>(n);

        for(int i = 0; i < indicies.size(); i++){
            indicies.set(i, i);
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
}