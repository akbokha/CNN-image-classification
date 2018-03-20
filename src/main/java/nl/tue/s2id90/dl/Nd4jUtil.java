package nl.tue.s2id90.dl;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;
import static org.nd4j.linalg.factory.Nd4j.reverse;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.NDArrayUtil;

/**
 * Some auxiliary functions on Nd4j arrays.
 * @author huub
 */
public class Nd4jUtil {
   
    /** returns argmax(a).
     * @param array
     * @return index of array element with highest value
     */
    static public int argMax(INDArray array) {
        return Nd4j.getExecutioner().execAndReturn(new IAMax(array)).getFinalResult();
    }
   
    /**
     * Reverses array a, row by row on the lowest level. For instance,
     * <html><pre>[[[1.00,  2.00],  
  [3.00,  4.00],  
  [5.00,  6.00]],  

 [[7.00,  8.00],  
  [9.00,  10.00],  
  [11.00,  12.00]]]
     * </pre>
     * 
     * is converted in:
     * <pre>[[[2.00,  1.00],  
  [4.00,  3.00],  
  [6.00,  5.00]],  

 [[8.00,  7.00],  
  [10.00,  9.00],  
  [12.00,  11.00]]]

     * </pre>
     * </html>
     * @param a 
     */
    static public void flipRows(INDArray a) {
        int[] shape = a.shape();
        if (shape.length==1) {
            reverse(a);
        } else if (shape.length==2) {
            for(int i=0;i<a.shape()[0];i++) {
                reverse(a.getRow(i));
            }
        } else {
            for(int i=0; i<a.shape()[0]; i++) {
                flipRows(a.getRow(i));
            }
        }
    }
    
    /** translates array a, on lowest level, by dx, dy. */
    static public void translate(INDArray a, int dx, int dy) {   
        int[] shape = a.shape();
        if (shape.length==1) {
            // do nothing
        } else if (shape.length==2) {
            int d0 = shape[0], d1 = shape[1];
            INDArray sub = a.get(
                    NDArrayIndex.interval(Math.max(0,dx), Math.min(d0,d0+dx)),
                    NDArrayIndex.interval(Math.max(0,dy), Math.min(d1,d1+dy))
            );
        } else {
            for(int i=0; i<a.shape()[0]; i++) {
                translate(a.getRow(i),dx,dy);
            }
        }
    }

    /** rotates array 90 degrees at lowest level.
     * @param a */
    public static void rotate(INDArray a) {
        int[] shape = a.shape();
        if (shape.length==1) {
            // do nothing
        } else if (shape.length==2) {
            Nd4j.rot90(a);
        } else {
            for(int i=0; i<a.shape()[0]; i++) {
                rotate(a.getRow(i));
            }
        }
    }
}
