

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


    public static void elementwiseMultiply(float[][] a, float[][] b, float[][] result){
        //TODO: check for dimension issues

        for(int r = 0; r < a.length; r++){
            for(int c = 0; c < a.length; c++){
                result[r][c] = a[r][c] * b[r][c];
            }
        }

    }

    public static float[][] elementwiseMultiply(float[][] a, float[][] b){
        //TODO: check for dimension issues

        float[][] r = new float[getNumRows(a)][getNumColumns(a)];

        elementwiseMultiply(a, b, r);

        return r;
    }

    public static void elementwiseMultiply(float[] a, float[] b, float[] result){
        //TODO: check for dimension issues


        for(int i = 0; i < a.length; i++){
            result[i] = a[i] * b[i];
        }
    }

    public static float[] elementwiseMultiply(float[] a, float[] b){
        //TODO: check for dimension issues

        
        float[] r = new float[a.length];

        elementwiseMultiply(a, b, r);

        return r;
    }


    public static float[][] arrayToMatrix(float[] array){
        float[][] r = new float[array.length][1];

        for(int i = 0; i < array.length; i++){
            r[i][0] = array[i];
        }

        return r;
    }
}