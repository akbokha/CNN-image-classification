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
import java.util.Random;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.PrimitivesDataGenerator;


/**
 * @author Abdel K. Bokharouss
 * @author Adriaan Knapen
 */
public class SCTExperiment extends SgdExperimentTemplate {
    
    static int FLATTEN_LINEAR = 1; // can be used to flatten the shape of an input image into a linear shape
    static int PIXELS_X = 28;
    static int PIXELS_Y = 28;
    static int PIXELS = PIXELS_X * PIXELS_Y;
    static private InputReader reader;

    private SCTExperiment(float learningRate, int batchSize, int epochs) {
        super();
        
        this.learningRate = learningRate;
        this.batchSize = batchSize;
        this.epochs = epochs;
    }
    
    @Override
    protected String[] getLabels() {
        return new String[] {"Square", "Circle", "Triangle"};
    }
    
    @Override
    protected InputReader getReader() throws IOException {
        if (reader==null) {
            int seed=11081961, trainingDataSize=1200*4, testDataSize=200*4;
            reader = new PrimitivesDataGenerator(batchSize, seed, trainingDataSize, testDataSize);
        }
        
        return reader;
    }
      
    public static void main (String [] args) throws IOException {
        Random r = new Random();
        float l;
        int b, e;
        for (int i = 0; i < 50; i++) {
            l = (float) Math.pow(r.nextFloat()/3f, 2);
            b = r.nextInt(12) * 2 + 12;
            e = (int) Math.pow(1 + r.nextFloat() * 8f, 2);
            new SCTExperiment(l, b, e).go();
        }
    }
}
