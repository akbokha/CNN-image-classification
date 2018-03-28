/*
 * Momentum update is another approach that almost always enjoys better converge rates on deep networks. 
 * This update can be motivated from a physical perspective of the optimization problem. 
 * for more information: https://cs231n.github.io/neural-networks-3/
*/

package experiments;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Implementation of the Gradient Descent Momentum variation
 * @author Abdel K. Bokharouss
 * @author Adriaan Knapen
 */
public class GradientDescentMomentum implements UpdateFunction {
    INDArray update;
    float mu;
    static float [] mu_values = new float []{0.5f, 0.9f, 0.95f, 0.99f};
    /**
     * 0 gives mu = 0.5
     * 1 gives mu = 0.9
     * 2 gives mu = 0.95
     * 3 gives mu = 0.99
     */
    
    public GradientDescentMomentum() {
        this.mu = mu_values[1];
    }
    
    public GradientDescentMomentum(int muChoice) {
        this.mu = mu_values[muChoice];
    }
    
    /**
     * Does a gradient descent step with factor minus learningRate and corrected for batchSize.
     * @param value
     * @param isBias
     * @param gradient
     */
    @Override
    public void update(INDArray value, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        if (update == null) update = gradient.dup('f').assign(0); // initiaized as zero
        
        /*
        * Implements momentum as given in https://cs231n.github.io/neural-networks-3/#sgd
        *   v = mu * v - learning_rate * dx # integrate velocity
        *   x += v # integrate position
        * with value as v, learningRate/batchSize as learning_rate, value as x, 
        * and gradient as dx.
        */
        update = update.mul(mu).sub(gradient.mul(learningRate/batchSize)); // update <-- update * mu -  (learningRate/batchSize) * gradient
        value.addi(update); // value <-- value + update
    }
}
