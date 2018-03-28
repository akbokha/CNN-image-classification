package experiments;

import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Adadelta implements UpdateFunction {
    INDArray update;
    
    @Override
    public void update(INDArray array, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        if (update == null) update = gradient.dup('f').assign(0); // initiaized as zero
        
        
    }
    
}
