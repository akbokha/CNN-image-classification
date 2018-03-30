/*
 * The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
 * There are 50000 training images and 10000 test images. 
 * The dataset is divided into five training batches and one test batch, each with 10000 images. 
 * The test batch contains exactly 1000 randomly-selected images from each class. 
 * The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. 
 * Between them, the training batches contain exactly 5000 images from each class. 
 *
 * The objective is to design a network using convolutional layers, possibly combined with pooling layers and
 * to tune the parameters, #layers, layer-sizes et cetera to achieve a relatively good performance (e.g. an accuracy of >70/80+ % as min. threshold)
 */
package experiments;
import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Convolution2D;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.FullyConnected;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.layer.PoolMax2D;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.experiment.Experiment;
import nl.tue.s2id90.dl.input.Cifar10Reader;
import static nl.tue.s2id90.dl.input.Cifar10Reader.getLabelsAsString;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;

/**
 * @author Abdel K. Bokharouss
 * @author Adriaan Knapen
 */
public class Cifar10Convolution extends Experiment {
    int batchSize = 64; // size of the batches in which the data is processed
    float learningRate = 0.1f; // parameter for the gradient descent optimization method
    int epochs = 5; // number of epochs that a training takes
    
    static final int RGB_DEPTH = 3; 
    static final int PIXELS_X = 32;
    static final int PIXELS_Y = 32;
    static final int PIXELS = PIXELS_X * PIXELS_Y;
    
    static String[] labels; // the labels in the dataset (e.g. airplane, dog, ship, truck)
    static int CLASSES; // number of labels 
    
    boolean l2Decay = true;
    
    public Cifar10Convolution() {
        super(true);  // true --> create and show GUI
        labels = getLabelsAsString().toArray(new String[getLabelsAsString().size()]);
        CLASSES = labels.length;
    }
    
    public Cifar10Convolution(boolean l2Decay) {
        this(); // call other constructor
        this.l2Decay = l2Decay;
    }
    
    public void go() throws IOException {
        // read input and print information on the data
        Cifar10Reader reader = new Cifar10Reader(batchSize, CLASSES);
        System.out.println("Reader info:\n" + reader.toString());
        
        // show a set of images to get more acquinted with the dataset
        ShowCase showCase = new ShowCase(i -> labels[i]);
        FXGUI.getSingleton().addTab("show case " + learningRate + " " + batchSize + " " + epochs, showCase.getNode());
        showCase.setItems(reader.getValidationData(100));
        
        // pre-processing: mean-subtraction
        MeanSubtraction ms = new MeanSubtraction();
        ms.fit(reader.getTrainingData());
        ms.transform(reader.getTrainingData());
        ms.transform(reader.getValidationData());
        
        System.out.println("\nin : " + reader.getInputShape() + " out: " + reader.getOutputShape() + "\n");
        
        Model model = createModel(PIXELS_X, PIXELS_Y, PIXELS, RGB_DEPTH, CLASSES);
        Optimizer sgd = SGD.builder()
                .model(model)
                .learningRate(learningRate)
                .validator(new Classification())
                .updateFunction(() -> new L2Decay(GradientDescentMomentum::new, 0.000001f))
                .build();
        trainModel(model, reader, sgd, epochs, 0);
    }
    
    private Model createModel(int inputX, int inputY, int pixels, int shape, int classes) {
        // network topology
        // INPUT => [[Conv => RELU]*N => MaxPool]*K => FC (Softmax)
        Model model = new Model(new InputLayer("In", new TensorShape(inputX, inputY, shape), true)); // input layer      
        // to do: design network
       
        model.initialize(new Gaussian()); // initializing the weights
        System.out.println(model); // print summary of the model
        return model;
    }
    
    public static void main (String [] args) throws IOException {
        new Cifar10Convolution().go();
    }
    
}
