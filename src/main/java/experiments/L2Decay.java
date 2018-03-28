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
    
    @Override
    public void update(INDArray array, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        
    }
}
