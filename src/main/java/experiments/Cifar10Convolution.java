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
import nl.tue.s2id90.dl.NN.optimizer.update.GradientDescent;
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
    int batchSize = 128; // size of the batches in which the data is processed
    float learningRate = .0009f; // parameter for the gradient descent optimization method
    int epochs = 8; // number of epochs that a training takes
    
    static final int RGB_DEPTH = 3; 
    static final int PIXELS_X = 32;
    static final int PIXELS_Y = 32;
    static final int PIXELS = PIXELS_X * PIXELS_Y;
    static final int SHAPE = PIXELS * RGB_DEPTH;
    
    static String[] labels; // the labels in the dataset (e.g. airplane, dog, ship, truck)
    static int CLASSES; // number of labels 
    
    boolean l2Decay = true;
    
    final int convolKernelSize = 3;
    final int convolStride = 1;
    final int convolKernels = 32;
    final int convolZeroPadding = 0;
    
    final int poolStride = 2;
    
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
        Cifar10MeanSubtraction cms = new Cifar10MeanSubtraction();
        cms.fit(reader.getTrainingData());
        cms.transform(reader.getTrainingData());
        cms.transform(reader.getValidationData());
        
        System.out.println("\nin : " + reader.getInputShape() + " out: " + reader.getOutputShape() + "\n");
        
        Model model = createModel(PIXELS_X, PIXELS_Y, PIXELS, SHAPE, RGB_DEPTH, CLASSES);
        Optimizer sgd = SGD.builder()
                .model(model)
                .learningRate(learningRate)
                .validator(new Classification())
//                .updateFunction(GradientDescent::new)
//                .updateFunction(GradientDescentMomentum::new)
//                .updateFunction(GradientDescentNesterovMomentum::new)
                .updateFunction(() -> new L2Decay(GradientDescentMomentum::new, 0.0000001f))
                .build();
//        trainModel(model, reader, sgd, epochs, 0, batchSize);
        trainAndSaveModel(model, reader, sgd, epochs, 0, batchSize);
    }
    
    private Model createModel(int inputX, int inputY, int pixels, int shape, int depth, int classes) {
        // network topology
        Model model = new Model(new InputLayer("In", new TensorShape(inputX, inputY, depth), true)); // input layer      
        
        // 1st experiment (benchmarking purposes): linear classifier INPUT -> FC
//        model.addLayer(new Flatten("Flatten", new TensorShape(inputX, inputY, depth)));
//        model.addLayer(new OutputSoftmax("Out", new TensorShape(shape), classes, new CrossEntropy()));

        // 2nd experiment (benchmarking purposes): simple Convnet INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC  
        // params: batchSize = 128; learningRate = 0.001f; epochs = 5; convolKernelSize = 3; convolStride = 1; convolKernels = 32; convolZeroPadding = 0; poolStride = 2;
//        model.addLayer(new Convolution2D(String.format("conv%s", Integer.toString(1)), new TensorShape(inputX, inputY, depth), convolKernelSize, convolKernels, new RELU()));
//        model.addLayer(new PoolMax2D(String.format("pool%s", Integer.toString(1)), new TensorShape(inputX, inputY, convolKernels), poolStride));
//        model.addLayer(new Convolution2D(String.format("conv%s", Integer.toString(2)), new TensorShape(inputX / poolStride, inputY / poolStride, convolKernels), convolKernelSize, convolKernels, new RELU()));
//        model.addLayer(new PoolMax2D(String.format("pool%s", Integer.toString(2)), new TensorShape(inputX / poolStride, inputY / poolStride, convolKernels), poolStride));
//        model.addLayer(new Flatten("Flatten", new TensorShape(inputX / (poolStride * poolStride), inputY / (poolStride * poolStride), convolKernels)));        
//        model.addLayer(
//                new FullyConnected(String.format("fc%s", Integer.toString(1)), new TensorShape(inputX / (poolStride * poolStride) * inputY / (poolStride * poolStride) * convolKernels), 
//                inputX / (poolStride * poolStride) * inputY / (poolStride * poolStride) * convolKernels / 4, new RELU())
//        );
//        model.addLayer(
//                new FullyConnected(String.format("fc%s", Integer.toString(2)), new TensorShape(inputX / (poolStride * poolStride) * inputY / (poolStride * poolStride) * convolKernels / 4), 
//                inputX / (poolStride * poolStride) * inputY / (poolStride * poolStride) * convolKernels / 16, new RELU())
//        );
//        model.addLayer(new OutputSoftmax("Out", new TensorShape(inputX / (poolStride * poolStride) * inputY / (poolStride * poolStride) * convolKernels / 16), labels.length, new CrossEntropy()));
        
        // eventual design of the convolutional network     
        model.addLayer(
                new Convolution2D(String.format("conv%s", Integer.toString(1)), new TensorShape(inputX, inputY, depth), convolKernelSize, convolKernels, new RELU()));
        model.addLayer(
                new Convolution2D(String.format("conv%s", Integer.toString(2)), new TensorShape(inputX, inputY, convolKernels), convolKernelSize, convolKernels, new RELU()));
        model.addLayer(
                new PoolMax2D(String.format("pool%s", Integer.toString(1)), new TensorShape(inputX, inputY, convolKernels),
                        poolStride));
        model.addLayer(
                new Convolution2D(String.format("conv%s", Integer.toString(3)), new TensorShape(inputX / poolStride, inputY / poolStride, convolKernels),
                        convolKernelSize, convolKernels * 2, new RELU()));
        model.addLayer(
                new Convolution2D(String.format("conv%s", Integer.toString(4)), new TensorShape(inputX / poolStride, inputY / poolStride, convolKernels * 2),
                        convolKernelSize, convolKernels * 2, new RELU()));
        model.addLayer(
                new PoolMax2D(String.format("pool%s", Integer.toString(2)), new TensorShape(inputX / poolStride, inputY / poolStride, convolKernels * 2),
                        poolStride));
        model.addLayer(
                new Convolution2D(String.format("conv%s", Integer.toString(5)), new TensorShape(inputX / (poolStride * poolStride), inputY / (poolStride * poolStride), convolKernels * 2),
                convolKernelSize, convolKernels * 3, new RELU()));
        model.addLayer(
                new Convolution2D(String.format("conv%s", Integer.toString(6)), new TensorShape(inputX / (poolStride * poolStride), inputY / (poolStride * poolStride), convolKernels * 3),
                convolKernelSize, convolKernels * 3, new RELU()));
        model.addLayer(
                new PoolMax2D(String.format("pool%s", Integer.toString(3)), new TensorShape(inputX / (poolStride * poolStride), inputY / (poolStride * poolStride), convolKernels * 3),
                        poolStride));        
        model.addLayer(
                new Flatten("Flatten", new TensorShape(inputX / (poolStride * poolStride * poolStride), inputY / (poolStride * poolStride * poolStride), convolKernels * 3)));        
        model.addLayer(
                new FullyConnected(String.format("fc%s", Integer.toString(1)), new TensorShape(inputX / (poolStride * poolStride * poolStride) * inputY / (poolStride * poolStride * poolStride) * convolKernels * 3), 
                convolKernels * 3, new RELU()));
        model.addLayer(
                new OutputSoftmax("Out", new TensorShape(convolKernels * 3), labels.length, new CrossEntropy()));

        model.initialize(new Gaussian()); // initializing the weights
        System.out.println(model); // print summary of the model
        return model;
    }
    
    public static void main (String [] args) throws IOException {
//        new Cifar10Convolution().go();
        new Cifar10Convolution(true).go();
    }
    
}
