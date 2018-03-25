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

/**
 * Implementation of the mean-subtraction pre-processing method
 * @author Abdel K. Bokharouss
 * @author Adriaan Knapen
 */
public class MeanSubtraction implements DataTransform {
    float mean;
    
    @Override
    public void fit(List<TensorPair> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Empty dataset");
        }
        float sumOverMeans = 0; // to calculate the mean over the means 
        for (TensorPair pair : data) {
            sumOverMeans += pair.model_input.getValues().meanNumber().floatValue();
        }
        this.mean = sumOverMeans / data.size(); // this should be subtracted from every feature
    }
    
    @Override 
    public void transform(List<TensorPair> data) {
        for (TensorPair pair : data) {
            pair.model_input.getValues().subi(this.mean); // (in-place) fit first (this.mean will otherwise be 0)
        }
    }
}
