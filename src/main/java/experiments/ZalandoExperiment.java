/*
 * Classification: fashion-MNIST
 * Building a neural network for the classification of images.
 * The network is built for classifying gray-scale images of 28x28 pixels.
 * Each image is associated with a label from 10 classes.
 * 60k images are used to train the model and 10k images are used for validation
 * The fashion-MNIST dataset is made available by Zalando Research.
 * For more information: https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/
 */
package experiments;

import java.io.IOException;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.MNISTReader;

/**
 * @author Abdel K. Bokharouss
 * @author Adriaan Knapen
 */

public class ZalandoExperiment extends SgdExperimentTemplate {
    
    public ZalandoExperiment(float learningRate, int batchSize, int epochs) {
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.epochs = epochs;
    }
    
    @Override
    protected String[] getLabels() {
        return new String[]{
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        };
    }
    
    @Override
    protected InputReader getReader() throws IOException {
        return MNISTReader.fashion(batchSize);
    }
      
    public static void main (String [] args) throws IOException {
        new ZalandoExperiment(0.005f, 16, 15).go();
//        List<Float> learningRates = new ArrayList<Float>() {{
//            add(0.005f); add(0.01f); add(0.02f);
//        }};
//        List<Integer> batchSizes = new ArrayList<Integer>() {{
//            add(16); add(32); add(64);
//        }};
//        List<Integer> epochVals = new ArrayList<Integer>() {{
//            add(5); add(10); add(15);
//        }};
//        for (float learningRate : learningRates) {
//            for (int batchSize : batchSizes) {
//                for (int epochs : epochVals) {
//                    new ZalandoExperiment(learningRate, batchSize, epochs).go();
//                }
//            }
//        }

    }
}
