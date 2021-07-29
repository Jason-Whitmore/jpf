import java.util.ArrayList;


/**
 * Class that contains static functions for linear algebra and matrix operations
 */
public class LinearAlgebra{

    //Initialization methods

    /**
     * Initializes a matrix with values from a random uniform distribution
     * @param numRows The number of rows in the matrix.
     * @param numColumns The number of columns in the matrix.
     * @param lowerBound The minimum value that an element can be.
     * @param upperBound The maximum value that an element can be.
     * @return The randomized matrix.
     */
    public static float[][] initializeRandomUniformMatrix(int numRows, int numColumns, float lowerBound, float upperBound){
        //Check params
        Utility.checkGreaterThanZero(numRows, numColumns);
        Utility.checkReal(lowerBound, upperBound);

        float[][] ret = new float[numRows][numColumns];

        for(int r = 0; r < numRows; r++){
            for(int c = 0; c < numColumns; c++){
                ret[r][c] = Utility.getRandomUniform(lowerBound, upperBound);
            }
        }

        return ret;
    }

    /**
     * Initializes a float array/vector with the specified constant.
     * @param length The length of the float array. Should be >= 1
     * @param value The constant to populate the array with
     * @return The populated array.
     */
    public static float[] initializeConstant(int length, float value){
        //Check params
        Utility.checkGreaterThanZero(length);

        float[] r = new float[length];

        for(int i = 0; i < r.length; i++){
            r[i] = value;
        }

        return r;
    }


    /**
     * Initializes a 2d array/matrix with the specified constant
     * @param numRows The number of rows in the matrix. Should be >= 1
     * @param numCols The number of cols in the matrix. Should be >= 1
     * @param value The value to populate the 2d array with.
     * @return The populated 2d array.
     */
    public static float[][] initializeConstant(int numRows, int numCols, float value){
        //Check params
        Utility.checkGreaterThanZero(numRows, numCols);

        float[][] r = new float[numRows][];

        for(int i = 0; i < r.length; i++){
            r[i] = LinearAlgebra.initializeConstant(numCols, value);
        }

        return r;
    }


    /**
     * Initializes a matrix from the input data
     * @param data The matrix data as an arraylist of arraylists.
     * @return The allocated and initialized matrix.
     */
    public static float[][] initializeFromArrayList(ArrayList<ArrayList<Float>> data){
        Utility.checkNotNull(data);


        int numRows = data.size();

        float[][] ret = new float[numRows][];

        for(int r = 0; r < numRows; r++){

            ret[r] = new float[data.get(r).size()];

            for(int c = 0; c < data.get(r).size(); c++){
                ret[r][c] = data.get(r).get(c);
            }
        }

        return ret;
    }

    /**
     * Initializes a square identity matrix which is all zero entrys except for entries
     * containing 1 along the diagonal
     * @param size The size of the matrix.
     * @return An identity matrix of the given size
     */
    public static float[][] initializeIdentityMatrix(int size){
        Utility.checkGreaterThanZero(size);

        float[][] result = new float[size][size];

        for(int i = 0; i < size; i++){
            result[i][i] = 1f;
        }

        return result;
    }

    
    /**
     * Simple wrapper function for getting the number of columns in a matrix.
     * @param matrix The matrix to retrieve the number of columns.
     * @return The number of columns.
     */
    public static int getNumColumns(float[][] matrix){
        Utility.checkNotNull((Object)matrix);
        Utility.checkNotNull((Object)matrix[0]);

        return matrix[0].length;
    }

    /**
     * Simple wrapper function for getting the number of rows in a matrix.
     * @param matrix The matrix to retrieve the number of rows.
     * @return The number of rows.
     */
    public static int getNumRows(float[][] matrix){
        Utility.checkNotNull((Object)matrix);

        return matrix.length;
    }

    /**
     * Transposes one matrix into another matrix
     * @param a The matrix to transpose.
     * @param t The matrix to place the result in.
     */
    public static void transpose(float[][] a, float[][] t){
        Utility.checkNotNull((Object)a, (Object)t);
        Utility.checkEqual(LinearAlgebra.getNumColumns(a), LinearAlgebra.getNumRows(t));
        Utility.checkEqual(LinearAlgebra.getNumColumns(t), LinearAlgebra.getNumRows(a));

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
        float[][] t = new float[LinearAlgebra.getNumColumns(a)][LinearAlgebra.getNumRows(a)];

        LinearAlgebra.transpose(a, t);
        return t;
    }

    /**
     * Performs matrix multiplication. Matrix dimensions should be valid.
     * @param a The first matrix.
     * @param b The second matrix.
     * @param result The resulting matrix where a * b will be placed.
     */
    public static void matrixMultiply(float[][] a, float[][] b, float[][] result){
        LinearAlgebra.matrixMultiplyParamCheck(a, b, result);
        
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
     * Checks the parameters for the static void matrix multiplication for correctness.
     * @param a The first matrix in a * b
     * @param b The second matrix in a * b
     * @param result The result matrix to place results into
     */
    private static void matrixMultiplyParamCheck(float[][] a, float[][] b, float[][] result){
        Utility.checkNotNull((Object)a, (Object)b, (Object)result);
        Utility.checkMatrixRectangle(a, b, result);

        int aRows = LinearAlgebra.getNumRows(a);
        int aCols = LinearAlgebra.getNumColumns(a);

        int bRows = LinearAlgebra.getNumRows(b);
        int bCols = LinearAlgebra.getNumColumns(b);

        int resultRows = LinearAlgebra.getNumRows(result);
        int resultCols = LinearAlgebra.getNumColumns(result);

        //Check if a * b is possible
        Utility.checkEqual(aCols, bRows);

        //Check that result dimensions are correct
        Utility.checkEqual(aRows, resultRows);
        Utility.checkEqual(bCols, resultCols);
    }

    /**
     * Performs matrix multiplication. Matrix dimensions should be valid.
     * @param a The first matrix.
     * @param b The second matrix.
     * @return The newly allocated result matrix a * b
     */
    public static float[][] matrixMultiply(float[][] a, float[][] b){
        //Check parameters
        Utility.checkNotNull((Object)a, (Object)b);
        Utility.checkMatrixRectangle(a, b);

        int aCols = LinearAlgebra.getNumColumns(a);
        int bRows = LinearAlgebra.getNumRows(b);
        Utility.checkEqual(aCols, bRows);

        float[][] r = new float[getNumRows(a)][getNumColumns(b)];

        LinearAlgebra.matrixMultiply(a, b, r);

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
        //Check parameters
        Utility.checkNotNull((Object)a, (Object)b, (Object)result);
        Utility.checkMatrixRectangle(a, b, result);
        Utility.checkMatrixDimensionsEqual(a, b, result);

        for(int r = 0; r < LinearAlgebra.getNumRows(a); r++){
            for(int c = 0; c < LinearAlgebra.getNumColumns(a); c++){
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
        //Check parameters
        Utility.checkNotNull((Object)a, (Object)b);
        Utility.checkMatrixRectangle(a, b);
        Utility.checkMatrixDimensionsEqual(a, b);

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
        //Check parameters
        Utility.checkNotNull((Object)a, (Object)b, (Object) result);
        Utility.checkMatrixRectangle(a, b, result);
        Utility.checkMatrixDimensionsEqual(a, b, result);

        for(int r = 0; r < a.length; r++){
            for(int c = 0; c < a[r].length; c++){
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
        //Check parameters
        Utility.checkNotNull((Object)a, (Object)b);
        Utility.checkMatrixRectangle(a, b);
        Utility.checkMatrixDimensionsEqual(a, b);

        float[][] r = new float[LinearAlgebra.getNumRows(a)][LinearAlgebra.getNumColumns(a)];

        LinearAlgebra.elementwiseMultiply(a, b, r);

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
        //Check parameters
        Utility.checkNotNull((Object)a, (Object)b, (Object) result);
        Utility.checkArrayLengthsEqual(a, b);
        Utility.checkArrayLengthsEqual(a, result);

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
        //Check parameters
        Utility.checkNotNull((Object)a, (Object)b);
        Utility.checkArrayLengthsEqual(a, b);
        
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
        //Check parameter
        Utility.checkNotNull((Object)array);

        float[][] r = new float[array.length][1];

        for(int i = 0; i < array.length; i++){
            r[i][0] = array[i];
        }

        return r;
    }


    /**
     * Converts the only column of a matrix into an array
     * @param matrix The matrix to convert.
     * @return The newly allocated array conversion.
     */
    public static float[] matrixToArray(float[][] matrix){
        //Check parameter
        Utility.checkNotNull((Object)matrix);
        Utility.checkMatrixRectangle(matrix);
        Utility.checkEqual(LinearAlgebra.getNumColumns(matrix), 1);

        float[] r = new float[matrix.length];

        for(int i = 0; i < r.length; i++){
            r[i] = matrix[i][0];
        }

        return r;
    }
}