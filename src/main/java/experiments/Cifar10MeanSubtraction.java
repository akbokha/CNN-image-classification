/*
 * Implementation of the mean-subtraction pre-processing method
 * Mean subtraction is the most common form of preprocessing. 
 * It involves subtracting the mean across every individual feature in the data, and has the geometric interpretation of centering the cloud of data around the origin along every dimension.
 * In numpy, this operation would be implemented as: X -= np.mean(X, axis = 0)
 * With images specifically, for convenience it can be common to subtract a single value from all pixels (e.g. X -= np.mean(X)), or to do so separately across the three color channels.
 * For more information: https://cs231n.github.io/neural-networks-2/
 */
package experiments;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Implementation of the mean-subtraction pre-processing method
 * @author Abdel K. Bokharouss
 * @author Adriaan Knapen
 */
public class Cifar10MeanSubtraction implements DataTransform {
    float meanRed;
    float meanGreen;
    float meanBlue;
    
    static final int RED = 0; // according to order in the shape 
    static final int BLUE = 1;
    static final int GREEN = 2;

    @Override
    public void fit(List<TensorPair> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Empty dataset");
        }
        int shape = data.size(); // to reduce future function calls
        
        float sumOverMeansRed = 0;
        float sumOverMeansGreen = 0;
        float sumOverMeansBlue = 0;
        for (TensorPair pair : data) {
            sumOverMeansRed += pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(RED), NDArrayIndex.all(),
                    NDArrayIndex.all()).meanNumber().floatValue();
            sumOverMeansGreen += pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(GREEN), NDArrayIndex.all(),
                    NDArrayIndex.all()).meanNumber().floatValue();
            sumOverMeansBlue += pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(BLUE), NDArrayIndex.all(),
                    NDArrayIndex.all()).meanNumber().floatValue();
        }
        this.meanRed = sumOverMeansRed / shape; // this should be subtracted from every feature
        this.meanGreen = sumOverMeansGreen / shape;
        this.meanBlue = sumOverMeansBlue / shape;
    }
    
    @Override 
    public void transform(List<TensorPair> data) {
        for (TensorPair pair : data) {
            // (in-place) fit first (this.mean will otherwise be 0)
            pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(RED), NDArrayIndex.all(), NDArrayIndex.all()).subi(this.meanRed); 
            pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(GREEN), NDArrayIndex.all(), NDArrayIndex.all()).subi(this.meanGreen);
            pair.model_input.getValues().get(NDArrayIndex.point(0), NDArrayIndex.point(BLUE), NDArrayIndex.all(), NDArrayIndex.all()).subi(this.meanBlue);
        }
    }
}
