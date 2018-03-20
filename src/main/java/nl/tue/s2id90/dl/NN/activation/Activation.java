package nl.tue.s2id90.dl.NN.activation;

import nl.tue.s2id90.dl.json.JSONable;
import org.json.simple.JSONObject;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Roel van Engelen
 */
public interface Activation extends JSONable {
    
    /**
     * Calculate neuron activation
     * 
     * @param value float input value
     * @return float activation value
     */
    public float calculateActivation( float value );
    
    /**
     * Apply activation to INDArray tensor
     * 
     * @param tensor INDArray activation has to be applied to
     */
    public void activation( INDArray tensor );
    
    /**
     * Calculate derivative 
     * 
     * @param value
     * @return 
     */
    public float calculateDerivative( float value );
    
    /**
     * Calculate derivative and apply to tensor
     * 
     * @param tensor INDArray derivate has to be calculated of
     * @return 
     */
    public INDArray derivative( INDArray tensor );
    
    /**
     * Calculate activation backprop
     * 
     * @param preoutput INDArray preoutput
     * @param epsilon
     * @return 
     */
    public INDArray backpropagation( INDArray preoutput, INDArray epsilon );
    
    /**
     * 
     * @return 
     */
    public IActivation get_IActivation();
    
    public static Activation fromJson(JSONObject jo) {
        String type = (String)jo.get("type");
        switch(type) {
            case "RELU": return new RELU();
            case "Linear": return new Linear();
            case "Sigmoid": return new Sigmoid();
            case "Softmax": return new Softmax();
            default: throw new IllegalStateException("Unknownn activation type: "+type);
        }
    }
}