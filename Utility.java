import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

/**
 * Class for assorted static functions that may be useful elsewhere in the project or for the user.
 */
public class Utility{

    /**
     * Calculates randomized indicies and groups them into minibatches
     * @param n The total number of indicies forming the set {0,1,2, ..., n - 1}
     * @param minibatchSize The maximum size of minibatches, or groups, that these indicies are formed into
     * @return The randomized indicies grouped into minibatches
     */
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

    /**
     * Clips the values in the array to be in range (min, max)
     * @param array The array to modify.
     * @param min The minimum value an element can be.
     * @param max The maximum value an element can be.
     */
    public static void clip(float[] array, float min, float max){
        for(int i = 0; i < array.length; i++){
            if(array[i] < min){
                array[i] = min;
            } else if(array[i] > max){
                array[i] = max;
            }
        }
    }

    /**
     * Clips the values in the 2d array to be in range (min, max)
     * @param array The array to modify.
     * @param min The minimum value an element can be.
     * @param max The maximum value an element can be.
     */
    public static void clip(float[][] array, float min, float max){
        for(int i = 0; i < array.length; i++){
            clip(array[i], min, max);
        }
    }

    /**
     * Clips the values in the collection of 2d arrays to be in range (min, max)
     * @param arrays The collection of 2d arrays to modify
     * @param min The minimum value an element can be.
     * @param max The maximum value an element can be.
     */
    public static void clip(ArrayList<float[][]> arrays, float min, float max){
        for(int i = 0; i < arrays.size(); i++){
            clip(arrays.get(i), min, max);
        }
    }

    /**
     * Sets all the elements of the array to zero.
     * @param a The array to modify.
     */
    public static void clearArray(float[] a){
        for(int i = 0; i < a.length; i++){
            a[i] = 0;
        }
    }

    /**
     * Sets all the elements of the 2d array to zero.
     * @param a The 2d array to modify.
     */
    public static void clearArray(float[][] a){
        for(int i = 0; i < a.length; i++){
            clearArray(a[i]);
        }
    }

    /**
     * Scales all elements in the array.
     * @param a The array to modify
     * @param scalar The scaling factor
     */
    public static void scaleArray(float[] a, float scalar){
        for(int i = 0; i < a.length; i++){
            a[i] *= scalar;
        }
    }

    /**
     * Scales all the elements in the 2d array.
     * @param a The 2d array to modify.
     * @param scalar The scaling factor
     */
    public static void scaleArray(float[][] a, float scalar){
        for(int i = 0; i < a.length; i++){
            scaleArray(a[i], scalar);
        }
    }

    /**
     * Clones the input collection of arrays
     * @param arrays The collection of arrays to clone
     * @return A new ArrayList of 2d arrays identical in contents to the input.
     */
    public static ArrayList<float[][]> cloneArrays(ArrayList<float[][]> arrays){
        if(arrays == null){
            return null;
        }

        ArrayList<float[][]> r = new ArrayList<float[][]>(arrays.size());

        for(int i = 0; i < arrays.size(); i++){
            int numRows = arrays.get(i).length;
            int numCols = arrays.get(i)[0].length;

            r.add(new float[numRows][numCols]);

            //TODO: Clone the elements in the arrays?
        }


        return r;
    }

    /**
     * Adds one scaled 2d array to another. Used to apply a gradient to a model's parameters
     * @param dest The first 2d array, which is also where the result is stored.
     * @param b The second 2d array, which is scaled.
     * @param scalar The scalar to multiply the b array by.
     */
    public static void addArray(float[][] dest, float[][] b, float scalar){
        for(int r = 0; r < dest.length; r++){
            for(int c = 0; c < dest[r].length; c++){
                dest[r][c] = dest[r][c] + (b[r][c] * scalar);
            }
        }
    }

    /**
     * Adds the one scaled gradient to another gradient.
     * @param gradient The gradient to add to. This is the destination gradient.
     * @param newGradient The scaled gradient to add to the first parameter.
     * @param scalar The scalar to multiply newGradient by.
     */
    public static void addGradient(ArrayList<float[][]> gradient, ArrayList<float[][]> newGradient, float scalar){
        for(int i = 0; i < gradient.size(); i++){
            addArray(gradient.get(i), newGradient.get(i), scalar);
        }
    }

    /**
     * Scales the gradient by a given factor.
     * @param gradient The gradient to scale.
     * @param scalar The factor to scale the gradient by.
     */
    public static void scaleGradient(ArrayList<float[][]> gradient, float scalar){
        for(int i = 0; i < gradient.size(); i++){
            scaleArray(gradient.get(i), scalar);
        }
    }

    /**
     * Creates a formatted string of the array contents.
     * @param array The array to turn into a string.
     * @return The formatted string
     */
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

    /**
     * Creates a formatted string of the 2d array contents.
     * @param array The 2d array to turn into a string.
     * @return The formatted string.
     */
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

    /**
     * Creates a formatted string of the collection of 2d arrays.
     * @param arrays The collection of 2d arrays.
     * @return The formatted string.
     */
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


    public static float getRandomUniform(float lowerBound, float upperBound){
        float upper;
        float lower;

        if(upperBound > lowerBound){
            upper = upperBound;
            lower = lowerBound;
        } else {
            upper = lowerBound;
            lower = upperBound;
        }

        float delta = upper - lower;

        return lower + delta * ((float)Math.random());
    }


    static class Initializers{

        /**
         * Initializes the parameters to a uniform distribution of (min, max)
         * @param parameters The parameters to initialize
         * @param min The minimum value an entry can be.
         * @param max The maximum value an entry can be.
         */
        public static void initializeUniform(ArrayList<float[][]> parameters, float min, float max){

            for(int i = 0; i < parameters.size(); i++){
                initializeUniform(parameters.get(i), min, max);
            }

        }

        /**
         * Initializes the entries in the matrix to a unfirom distribution of (min, max).
         * @param matrix The matrix to initialize.
         * @param min The minimum value an entry can be.
         * @param max The maximum value an entry can be.
         */
        public static void initializeUniform(float[][] matrix, float min, float max){
            for(int r = 0; r < matrix.length; r++){
                for(int c = 0; c < matrix[r].length; c++){
                    matrix[r][c] = Utility.getRandomUniform(min, max);
                }
            }
        }

        /**
         * Initializes the parameters to a samples from a normal distribution
         * @param parameters The parameters to initialize.
         * @param mean The mean of the normal distribution.
         * @param variance The variance of the normal distribution.
         */
        public static void initializeNormal(ArrayList<float[][]> parameters, float mean, float variance){
            Random r = new Random();

            for(int i = 0; i < parameters.size(); i++){
                initializeNormal(parameters.get(i), mean, variance, r);
            }
        }

        /**
         * Initializes the matrix elements to be sample from a normal distribution.
         * @param matrix The matrix to initialize.
         * @param mean The mean of the normal distribution.
         * @param variance The variance of the normal distribution.
         * @param randObject The Random object to get the random samples from.
         */
        public static void initializeNormal(float[][] matrix, float mean, float variance, Random randObject){
            for(int r = 0; r < matrix.length; r++){
                for(int c = 0; c < matrix[r].length; c++){
                    float randomFloat = (float)randObject.nextGaussian();

                    matrix[r][c] = (randomFloat * variance) - mean;
                }
            }
        }

    }

    public static void passedTest(boolean worked){
        if(worked){
            System.out.println("Passed.");
        } else {
            System.out.println("Failed.");
        }
    }

    public static void unitTests(){
        //Testing initializers
        unitTestsInitializers();
    }

    public static void unitTestsInitializers(){

        boolean worked = false;
        System.out.println("Testing initializeNormal(ArrayList<float[][]>, mean, variance)");

        try{
            Initializers.initializeNormal(null, 0, 0);
        } catch(NullPointerException e){
            worked = true;
        }

        passedTest(worked);
        worked = false;


        
        
    }
    

}