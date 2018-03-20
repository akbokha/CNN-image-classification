package nl.tue.s2id90.dl.NN.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Roel van Engelen
 */
public class Linear implements Activation{

    private final INDArray           epsilon;
    private final ActivationIdentity identity;
    
    public Linear(){
        
        epsilon = Nd4j.create( new float[] { 1.0f }, new int[] { 1 }, 'c' );
        identity = new ActivationIdentity(); 
    }
    
    /**
     * Calculate neuron activation
     * RELU activation function defined as
     * max( 0, activation )
     * 
     * @param activation input value
     * @return activation value
     */
    @Override
    public float calculateActivation( float activation ){
        
        // TODO
        return 0;    
    }
    
    /**
     * Apply activation to INDArray tensor
     * RELU activation function defined as
     * max( 0, activation )
     * 
     * @param tensor INDArray activation has to be applied to
     */
    @Override
    public void activation( INDArray tensor ){
                
        identity.getActivation( tensor, true );
    }
    
    /**
     * Calculate derivative 
     * 
     * @param value
     * @return 
     */
    @Override
    public float calculateDerivative( float value ){
        
        
        // TODO
        
        return 0;
    }
    
    /**
     * Calculate derivative and apply to tensor
     * 
     * @param tensor INDArray derivate has to be calculated of
     */
    @Override
    public INDArray derivative( INDArray tensor ){
                
        return identity.backprop( tensor, epsilon ).getKey();
    }
    
    /**
     * Calculate activation backprop
     * 
     * @param preoutput INDArray preoutput
     * @param epsilon
     */
    @Override
    public INDArray backpropagation( INDArray preoutput, INDArray epsilon ){
                
        return identity.backprop( preoutput, epsilon ).getKey();
    }
    
    /**
     * 
     * @return 
     */
    public IActivation get_IActivation(){
        
        return identity;
    }
}