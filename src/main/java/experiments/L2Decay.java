package experiments;

import java.util.function.Supplier;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

public class L2Decay implements UpdateFunction {
    float decay;
    UpdateFunction f;
    
    public L2Decay(Supplier<UpdateFunction> supplier, float decay) {
        this.decay = decay;
        this.f = supplier.get();
    }
    
     /** 
     * A typical implementation of this interface does the following
     *      array <-- array - (learningRate/batchSize) * gradient
     * However, other implementations may decide to ignore e.g. the
     * learningRate.
     * @param array         array that is to be updated 
     * @param isBias        true is array represents bias values, as opposed to weights.
     * @param learningRate  learning rate for gradient descent
     * @param batchSize     the number of samples whose resulting gradients are accumulated in gradient
     * @param gradient      accumulated gradient
     */
    @Override
    public void update(INDArray array, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        f.update(array, isBias, learningRate, batchSize, gradient);
        // Only apply L2Decay when we are not working with a bias
        if (!isBias) {
            array.mul(array).mul(.5f*decay);
        }
    }
}
