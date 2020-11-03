import java.util.ArrayList;

/**
 * Class that contains static functions for linear algebra and matrix operations
 */
public class LinearAlgebra{

    //Matrix initialization methods

    /**
     * Initializes a matrix with values from a random uniform distribution
     * @param numRows The number of rows in the matrix.
     * @param numColumns The number of columns in the matrix.
     * @param lowerBound The minimum value that an element can be.
     * @param upperBound The maximum value that an element can be.
     * @return The randomized matrix.
     */
    public static float[][] initializeRandomUniformMatrix(int numRows, int numColumns, float lowerBound, float upperBound){
        float[][] ret = new float[numRows][numColumns];

        for(int r = 0; r < numRows; r++){
            for(int c = 0; c < numColumns; c++){
                ret[r][c] = Utility.getRandomUniform(lowerBound, upperBound);
            }
        }

        return ret;
    }

    /**
     * Determines if the input arraylist of arraylists is ragged, where atleast
     * one of the arraylists is a different size than the others.
     * @param <T> Can be of any type.
     * @param data The ArrayList of ArrayLists to determine if ragged
     * @return True if the array is ragged, else false 
     */
    public static <T> boolean isRagged(ArrayList<ArrayList<T>> data){
        if(data == null || data.size() == 0){
            return false;
        }

        int initialSize = data.get(0).size();

        for(int r = 0; r < data.size(); r++){
            if(initialSize != data.get(r).size()){
                return true;
            }
        }

        return false;
    }

    /**
     * Initializes a matrix from the input data
     * @param data The matrix data as an arraylist of arraylists
     * @return The allocated and initialized matrix.
     */
    public static float[][] initializeFromArrayList(ArrayList<ArrayList<Float>> data){
        //TODO: Check for bad ArrayList

        int numRows = data.size();
        int numCols = data.get(0).size();

        float[][] ret = new float[numRows][numCols];

        for(int r = 0; r < numRows; r++){
            for(int c = 0; c < numCols; c++){
                ret[r][c] = data.get(r).get(c);
            }
        }

        return ret;
    }


    /**
     * Allocates and initializes an array from the string
     * 
     * Example for syntax: [0,1,2,3]
     * @param s The string to create the array from.
     * @return The allocated and initialized array.
     */
    public static float[] initializeArrayFromString(String s){

        s = s.replace("[", "");
        s = s.replace("]", "");

        String[] sSplit = s.split(",");

        float[] r = new float[sSplit.length];

        for(int i = 0; i < r.length; i++){
            r[i] = Float.parseFloat(sSplit[i]);
        }

        return r;
    }

    /**
     * Initializes a matrix from a string.
     * 
     * Example format:
     * [[1,0,0]
     *  [0,1,0]
     *  [0,0,1]]
     * @param s The string containing the data for a matrix.
     * @return The newly allocated matrix from string data.
     */
    public static float[][] initializeFromString(String s){
        String[] sSplit = s.split("\n");

        sSplit[0] = sSplit[0].replace("[[", "[");
        sSplit[sSplit.length - 1] = sSplit[sSplit.length - 1].replace("]]", "]");

        float[][] r = new float[sSplit.length][];

        for(int i = 0; i < r.length; i++){
            r[i] = initializeArrayFromString(sSplit[i]);
        }


        return r;
    }

    
    /**
     * Simple wrapper function for getting the number of columns in a matrix.
     * @param matrix The matrix to retrieve the number of columns.
     * @return The number of columns.
     */
    public static int getNumColumns(float[][] matrix){
        return matrix[0].length;
    }

    /**
     * Simple wrapper function for getting the number of rows in a matrix.
     * @param matrix The matrix to retrieve the number of rows.
     * @return The number of rows.
     */
    public static int getNumRows(float[][] matrix){
        return matrix.length;
    }

    /**
     * Transposes one matrix into another matrix
     * @param a The matrix to transpose.
     * @param t The matrix to place the result in.
     */
    public static void transpose(float[][] a, float[][] t){
        //TODO: Enforce dimensions

        for(int r = 0; r < getNumRows(a); r++){
            for(int c = 0; c < getNumColumns(a); c++){
                t[c][r] = a[r][c];
            }
        }
    }

    /**
     * Transposes the input matrix.
     * @param a The matrix to transpose.
     * @return The newly allocated transposed matrix.
     */
    public static float[][] transpose(float[][] a){
        float[][] t = new float[getNumColumns(a)][getNumRows(a)];

        transpose(a, t);
        return t;
    }

    /**
     * Performs matrix multiplication. Matrix dimensions should be valid.
     * @param a The first matrix.
     * @param b The second matrix.
     * @param result The resulting matrix where a * b will be placed.
     */
    public static void matrixMultiply(float[][] a, float[][] b, float[][] result){
        //TODO: First, check the parameters for dimension issues


        for(int r = 0; r < getNumRows(result); r++){
            for(int c = 0; c < getNumColumns(result); c++){
                float sum = 0;

                for(int i = 0; i < getNumColumns(a); i++){
                    sum += a[r][i] * b[i][c];
                }

                result[r][c] = sum;
            }
        }
    }

    /**
     * Performs matrix multiplication. Matrix dimensions should be valid.
     * @param a The first matrix.
     * @param b The second matrix.
     * @return The newly allocated result matrix a * b
     */
    public static float[][] matrixMultiply(float[][] a, float[][] b){
        float[][] r = new float[getNumColumns(a)][getNumRows(b)];

        matrixMultiply(a, b, r);

        return r;
    }

    /**
     * Performs matrix addition. Matrix dimensions should be the same on both
     * input matricies.
     * @param a The first matrix.
     * @param b The second matrix.
     * @param result The matrix to place a + b into.
     */
    public static void matrixAdd(float[][] a, float[][] b, float[][] result){
        //TODO: check for parameter dimension issues


        for(int r = 0; r < a.length; r++){
            for(int c = 0; c < a[0].length; c++){
                result[r][c] = a[r][c] + b[r][c];
            }
        }

    }

    /**
     * Performs matrix addition. Matrix dimensions should be the same on both
     * input matricies.
     * @param a The first matrix.
     * @param b The second matrix.
     * @return The newly allocated result of a + b
     */
    public static float[][] matrixAdd(float[][] a, float[][] b){
        float[][] r = new float[getNumRows(a)][getNumColumns(a)];

        matrixAdd(a, b, r);

        return r;
    }


    /**
     * Performs elementwise matrix multiplication. Matrix dimensions should be
     * the same on both input matricies.
     * @param a The first matrix.
     * @param b The second matrix.
     * @param result The matrix where the elementwise multiplication result will be placed.
     */
    public static void elementwiseMultiply(float[][] a, float[][] b, float[][] result){
        //TODO: check for dimension issues

        for(int r = 0; r < a.length; r++){
            for(int c = 0; c < a.length; c++){
                result[r][c] = a[r][c] * b[r][c];
            }
        }

    }

    /**
     * Performs elementwise matrix multiplication. Matrix dimensions should be
     * the same on both input matricies.
     * @param a The first matrix.
     * @param b The second matrix.
     * @return The newly allocated result of the elementwise matrix multiplication.
     */
    public static float[][] elementwiseMultiply(float[][] a, float[][] b){
        //TODO: check for dimension issues

        float[][] r = new float[getNumRows(a)][getNumColumns(a)];

        elementwiseMultiply(a, b, r);

        return r;
    }

    /**
     * Performs elementwise vector multiplication. Array dimensions should be
     * the same on all inputs
     * @param a The first vector/array.
     * @param b The second vector/array.
     * @param result The array where the result is placed.
     */
    public static void elementwiseMultiply(float[] a, float[] b, float[] result){
        //TODO: check for dimension issues


        for(int i = 0; i < a.length; i++){
            result[i] = a[i] * b[i];
        }
    }

    /**
     * Performs elementwise vector multiplication. Array dimensions should be
     * the same on all inputs
     * @param a The first vector/array.
     * @param b The second vector/array.
     * @return The newly allocated result array.
     */
    public static float[] elementwiseMultiply(float[] a, float[] b){
        //TODO: check for dimension issues

        
        float[] r = new float[a.length];

        elementwiseMultiply(a, b, r);

        return r;
    }


    /**
     * Converts a 1d array to a 2d matrix with 1 column
     * @param array The array/vector to convert.
     * @return The newly allocated matrix conversion of the array.
     */
    public static float[][] arrayToMatrix(float[] array){
        float[][] r = new float[array.length][1];

        for(int i = 0; i < array.length; i++){
            r[i][0] = array[i];
        }

        return r;
    }
}