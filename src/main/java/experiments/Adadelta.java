package experiments;

import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import static org.nd4j.linalg.ops.transforms.Transforms.neg;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;
import static org.nd4j.linalg.ops.transforms.Transforms.sqrt;

/*
* Implements ADELTA from https://arxiv.org/pdf/1212.5701.pdf.
*/
public class Adadelta implements UpdateFunction {
    INDArray Eg2;
    INDArray Ex2;
    float e = 0.000001f;
    
    float mu;
    static float [] mu_values = new float []{0.5f, 0.9f, 0.95f, 0.99f};
    /**
     * 0 gives mu = 0.5
     * 1 gives mu = 0.9
     * 2 gives mu = 0.95
     * 3 gives mu = 0.99
     */
    
    public Adadelta() {
        this.mu = mu_values[1];
    }
    
    public Adadelta(int muChoice) {
        this.mu = mu_values[muChoice];
    }
    
    @Override
    public void update(INDArray value, boolean isBias, float learningRate, int batchSize, INDArray gradient) {
        if (Eg2 == null) Eg2 = gradient.dup('f').assign(0); // initiaized as zero
        if (Ex2 == null) Ex2 = gradient.dup('f').assign(0); // initiaized as zero
        
        /*
        * With mu as ρ, gradient as g, e as e, array as x_t
        */
        Eg2 = Eg2.muli(mu).addi(pow(gradient, 2)); // E[g^2]_t <-- ρE[g^2]_{t−1} + (1−ρ)g^2
        INDArray deltaX = neg(sqrt(Ex2.add(e)).muli(gradient).divi(sqrt(Eg2.add(e)))); // ∆x_t = −(g_t * RMS[∆x]_{t-1}) / RMS[g]t = -(g_t * sqrt(E[x^2]_{t-1} + e) / sqrt(E[g^2]_t + e)
        Ex2 = Ex2.muli(mu).addi(pow(deltaX, 2).muli(1f - mu)); // E[∆x^2]_t = ρE[∆x^2]_{t−1} + (1−ρ)∆x^2_t
        
        value.addi(deltaX); // x_{t+1} = x_t + ∆x_t
        
        gradient.assign(0);
    }
    
}
