



class LinearAlgebra{

    public static int getNumColumns(float[][] matrix){
        return matrix[0].length;
    }

    public static int getNumRows(float[][] matrix){
        return matrix.length;
    }

    public static void matrixMultiply(float[][] a, float[][] b, float[][] result){
        //TODO: First, check the parameters for dimension issues

    }

    public static float[][] matrixMultiply(float[][] a, float[][] b){
        float[][] r = new float[getNumColumns(a)][getNumRows(b)];

        matrixMultiply(a, b, r);

        return r;
    }

    public static void matrixAdd(float[][] a, float[][] b, float[][] result){
        //TODO: check for parameter dimension issues


        for(int r = 0; r < a.length; r++){
            for(int c = 0; c < a[0].length; c++){
                result[r][c] = a[r][c] + b[r][c];
            }
        }

    }

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