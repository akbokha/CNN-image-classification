package nl.tue.s2id90.dl.NN.validate;

import nl.tue.s2id90.dl.NN.tensor.Tensor;

/**
 * dummy validator, typically used if no other validator is provided.
 * @author huub
 */
public class ZeroValidator implements Validator {

    @Override
    public float validate(Tensor label, Tensor prediction) {
        return 0f;
    }
    
}
