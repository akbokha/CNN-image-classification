package nl.tue.s2id90.dl.NN.tensor;

import static java.lang.String.format;

/**
 * Tensor_Pair
 * convenience class holding a Tensor pair, model input and model output
 * 
 * @author Roel van Engelen
 */
public class TensorPair {
    
    public final Tensor model_input;
    public final Tensor model_output;
    
    /**
     * Simple data class holding two tensors, model input and model output
     * 
     * @param model_input  model input  Tensor
     * @param model_output model output Tensor ( target output )
     */
    public TensorPair( Tensor model_input, Tensor model_output ){
        
        this.model_input  = model_input;
        this.model_output = model_output;
    }
    
    @Override
    public String toString() {
        return format("output=%s: input=%s",model_output,model_input,model_output);
    }
}
