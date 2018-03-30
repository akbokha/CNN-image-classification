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
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.optimizer.update.GradientDescent;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.experiment.Experiment;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;

/**
 * @author Abdel K. Bokharouss
 * @author Adriaan Knapen
 */
abstract public class SgdExperimentTemplate extends Experiment {
    protected float learningRate = 0.005f;
    protected int batchSize = 16;
    protected int epochs = 15;
    
    static final int FLATTEN_LINEAR = 1; // can be used to flatten the shape of an input image into a linear shape
    static final int PIXELS_X = 28;
    static final int PIXELS_Y = 28;
    static final int PIXELS = PIXELS_X * PIXELS_Y;
    
    static String[] labels;
    static int CLASSES;
    
    public SgdExperimentTemplate() {
        super(true); // true --> create and show GUI
        labels = getLabels();
        CLASSES = getLabels().length;
    }
    
    abstract protected String[] getLabels();
    
    abstract protected InputReader getReader() throws IOException;
    
    public void go() throws IOException {
        // read input and print information on the data
        InputReader reader = getReader();
        System.out.println("Reader info:\n" + reader.toString());
        reader.getValidationData(1).forEach(System.out::println);
        
        // pre-processing: mean-subtraction
        MeanSubtraction ms = new MeanSubtraction();
        ms.fit(reader.getTrainingData());
        ms.transform(reader.getTrainingData());
        ms.transform(reader.getValidationData());
        System.out.println("\nin : " + reader.getInputShape() + " out: " + reader.getOutputShape() + "\n");
        
        // show a set of images to get more acquinted with the dataset
        ShowCase showCase = new ShowCase(i -> labels[i]);
        FXGUI.getSingleton().addTab("show case " + learningRate + " " + batchSize + " " + epochs, showCase.getNode());
        showCase.setItems(reader.getValidationData(100));
        
        Model model = createModel(PIXELS_X, PIXELS_Y, PIXELS, FLATTEN_LINEAR, CLASSES);

        Optimizer sgd = SGD.builder()
                .model(model)
                .learningRate(learningRate)
                .validator(new Classification())
//                .updateFunction(GradientDescent::new)
//                .updateFunction(GradientDescentMomentum::new)
//                .updateFunction(GradientDescentNesterovMomentum::new)
                .updateFunction(() -> new L2Decay(GradientDescentMomentum::new, 0.000001f))
                .build();
        trainModel(model, reader, sgd, epochs, 0, batchSize);
    }
    
    private Model createModel(int inputX, int inputY, int pixels, int shape, int classes) {
        // network topology
        Model model = new Model(new InputLayer("In", new TensorShape(inputX, inputY, shape), true));
        model.addLayer(new Flatten("Flatten", new TensorShape(inputX, inputY, shape)));
        model.addLayer(new OutputSoftmax("Out", new TensorShape(pixels), classes, new CrossEntropy()));
        model.initialize(new Gaussian()); // initializing the weights
        System.out.println(model); // print summary of the model
        return model;    
    }
}
