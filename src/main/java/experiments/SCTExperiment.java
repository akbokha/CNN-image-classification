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
import nl.tue.s2id90.dl.input.PrimitivesDataGenerator;


/**
 * @author Abdel K. Bokharouss
 * @author Adriaan Knapen
 */
public class SCTExperiment extends SgdExperimentTemplate {
    protected float learningRate = 0.2f;
    protected int batchSize = 64;
    protected int epochs = 5;
    
    static int FLATTEN_LINEAR = 1; // can be used to flatten the shape of an input image into a linear shape
    static int PIXELS_X = 28;
    static int PIXELS_Y = 28;
    static int PIXELS = PIXELS_X * PIXELS_Y;
    
    @Override
    protected String[] getLabels() {
        return new String[] {"S", "C", "T"};
    }
    
    @Override
    protected InputReader getReader() throws IOException {
        int seed=11081961, trainingDataSize=1200, testDataSize=200;
        return new PrimitivesDataGenerator(batchSize, seed, trainingDataSize, testDataSize);
    }
      
    public static void main (String [] args) throws IOException {
        new SCTExperiment().go();
    }
}
