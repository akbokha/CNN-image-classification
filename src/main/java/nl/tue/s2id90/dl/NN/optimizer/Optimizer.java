package nl.tue.s2id90.dl.NN.optimizer;

import nl.tue.s2id90.dl.experiment.BatchResult;
import nl.tue.s2id90.dl.NN.error.IllegalInput;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import java.util.List;
import lombok.Getter;
import lombok.Setter;

/**
 * Optimizer
 * Abstract Optimizer class this ensures new optimizer implementation will
 * have all functions required to train a model
 * 
 * also contains all Gui functions, passing data to gui to be shown
 * 
 * @author Roel van Engelen
 */
public abstract class Optimizer {
    
    @Getter @Setter protected float learningRate;
    
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////// abstract
    
    /**
     * Train model with single batch
     * 
     * @param batch List with Tensor_Pair training batch
     * @return 
     * @throws IllegalInput
     */
    public abstract BatchResult trainOnBatch( TensorPair batch ) throws IllegalInput;
    
    /**
     * validate model accuracy
     * 
     * @param batch List with Tensor_Pair validation batch
     * @return 
     * @throws IllegalInput
     */
    public abstract BatchResult validate( List<TensorPair> batch ) throws IllegalInput;
}
