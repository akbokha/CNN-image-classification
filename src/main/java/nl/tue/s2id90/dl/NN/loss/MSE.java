package nl.tue.s2id90.dl.NN.loss;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author Roel van Engelen
 */
public class MSE implements Loss{
    
    // Nd4j loss function
    private final ILossFunction loss;
    
    public MSE(){
        
        // initialize Nd4j loss function
        loss = LossFunctions.LossFunction.MSE.getILossFunction();
    }
    
    /**
     * Calculate loss over label and prediction
     * 
     * @param labels      tensor with correct output
     * @param preoutput   tensor with pre activation output
     * @return            mse loss value
     */
    @Override
    public float calculate_loss( INDArray labels, INDArray preoutput, IActivation activation ){
        
        return (float)loss.computeScore( labels, preoutput, activation, null, true );
    }
    
    /**
     * calculate final layer MSE backpropagation gradient
     * 
     * @param labels      tensor with correct output
     * @param preoutput   tensor with pre activation output
     * @param activation  ND4J Iactivation type
     * @return            INDArray backpropagation gradient
     */
    @Override
    public INDArray computeGradient( INDArray labels, INDArray preoutput, IActivation activation ){
        
        return loss.computeGradient( labels , preoutput, activation, null );
    }
}
