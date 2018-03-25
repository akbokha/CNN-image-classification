/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;

/**
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
    }
    
    @Override 
    public void transform(List<TensorPair> data) {
        // to-do
    }
}
