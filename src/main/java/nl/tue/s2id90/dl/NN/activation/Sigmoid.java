package nl.tue.s2id90.dl.NN.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Roel van Engelen
 */
public class Sigmoid implements Activation{

    private final INDArray          epsilon;
    private final ActivationSigmoid sigmoid;
    
    public Sigmoid(){
        
        epsilon = Nd4j.create( new float[] { 1.0f }, new int[] { 1 }, 'c' );
        sigmoid = new ActivationSigmoid();
    }
    
    /**
     * Calculate neuron activation
     * Sigmoid activation function defined as
     *        1
     *      --------
     *            -x
     *       1 + e
     * @param activation input value
     * @return activation value
     */
    @Override
    public float calculateActivation( float activation ){
        
        return (float)( 1 / ( 1 + Math.exp( -activation ) ) );    
    }
        
    /**
     * Apply activation to INDArray tensor
     * Sigmoid activation function defined as
     *        1
     *      --------
     *            -x
     *       1 + e
     * 
     * @param tensor INDArray activation has to be applied to
     */
    @Override
    public void activation( INDArray tensor ){
        
        sigmoid.getActivation(tensor, true);
    }
    
    /**
     * Calculate derivative of sigmoid function defined as
     * x * ( 1.0 - x )
     * 
     * @param value back-propagation value
     * @return float sigmoid activation
     */
    @Override
    public float calculateDerivative( float value ){
        
        // calculate sigmoid activation
        float sig = calculateActivation( value );
        
        // calculate sigmoid derivative
        return sig * ( 1.0f - sig );
    }
    
    /**
     * Calculate derivative of sigmoid function defined as
     * x * ( 1.0 - x )
     * 
     * @param tensor INDArray derivate has to be calculated of
     */
    @Override
    public INDArray derivative( INDArray tensor ){
        
        return sigmoid.backprop( tensor, epsilon ).getKey();
    }
    
    /**
     * Calculate activation backprop
     * 
     * @param preoutput INDArray preoutput
     * @param epsilon
     */
    @Override
    public INDArray backpropagation( INDArray preoutput, INDArray epsilon ){
                
        return sigmoid.backprop( preoutput, epsilon ).getKey();
    }
    
    /**
     * 
     * @return 
     */
    public IActivation get_IActivation(){
        
        return sigmoid;
    }
}
