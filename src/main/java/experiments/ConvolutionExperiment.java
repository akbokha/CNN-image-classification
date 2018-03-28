package experiments;

import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Convolution2D;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.layer.PoolMax2D;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.optimizer.update.GradientDescent;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.experiment.Experiment;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.PrimitivesDataGenerator;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.javafx.ShowCase;

public class ConvolutionExperiment extends Experiment {
    protected float learningRate = 0.2f;
    protected int batchSize = 64;
    protected int epochs = 5;
    protected int kernelSize = 3;
    protected int noFilters = 1;
    protected int N = 1;
    protected int M = 1;
    protected int K = 1;
    
    static int FLATTEN_LINEAR = 1; // can be used to flatten the shape of an input image into a linear shape
    static int PIXELS_X = 28;
    static int PIXELS_Y = 28;
    
    static final String[] LABELS = {"Square", "Circle", "Triangle"};
    static final int CLASSES = LABELS.length;
    
    public ConvolutionExperiment() {
        super(true);
    }
    
    public void go() throws IOException {
        // read input and print information on the data
        int seed=11081961, trainingDataSize=1200*4, testDataSize=200*4;
        InputReader reader = new PrimitivesDataGenerator(batchSize, seed, trainingDataSize, testDataSize);
        System.out.println("Reader info:\n" + reader.toString());
        reader.getValidationData(1).forEach(System.out::println);
        
        // show a set of images to get more acquinted with the dataset
        ShowCase showCase = new ShowCase(i -> LABELS[i]);
        FXGUI.getSingleton().addTab("show case", showCase.getNode());
        showCase.setItems(reader.getValidationData(100));
        
        Model model = createModel(PIXELS_X, PIXELS_Y, FLATTEN_LINEAR, CLASSES);

        Optimizer sgd = SGD.builder()
                .model(model)
                .learningRate(learningRate)
                .validator(new Classification())
                .updateFunction(() -> new L2Decay(GradientDescent::new, .0001f))
                .build();
        trainModel(model, reader, sgd, epochs, 0);
    }
    
    private Model createModel(int inputX, int inputY, int shape, int classes) {
        // network topology
        // INPUT => Conv => RELU => MaxPool => FC (Softmax)
        
        // INPUT
        Model model = new Model(new InputLayer("In", new TensorShape(inputX, inputY, shape), true));
        
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                // Conv && RELU
                model.addLayer(new Convolution2D(
                        "Conv("+m+","+n+")", 
                        new TensorShape(inputX, inputY, shape), 
                        kernelSize, 
                        noFilters, 
                        new RELU()
                ));
            }
            
            // MaxPool
            model.addLayer(new PoolMax2D("max-pool", new TensorShape(inputX, inputY, shape), 1));
        }
        
        // FC
        model.addLayer(new Flatten("Flatten", new TensorShape(inputX, inputY, shape)));
        model.addLayer(new OutputSoftmax("Out", new TensorShape(inputX * inputY), classes, new CrossEntropy()));
        
        model.initialize(new Gaussian()); // initializing the weights
        System.out.println(model); // print summary of the model
        return model;
    }
    
    private Model createModelSimple(int inputX, int inputY, int shape, int classes) {
        // network topology
        // INPUT => Conv => RELU => MaxPool => FC (Softmax)
        
        // INPUT
        Model model = new Model(new InputLayer("In", new TensorShape(inputX, inputY, shape), true));
        
        // Conv && RELU
        model.addLayer(new Convolution2D(
                "Conv", 
                new TensorShape(inputX, inputY, shape), 
                kernelSize, 
                noFilters, 
                new RELU()
        ));
        
        // MaxPool
        model.addLayer(new PoolMax2D("max-pool", new TensorShape(inputX, inputY, shape), 1));
        
        // FC
        model.addLayer(new Flatten("Flatten", new TensorShape(inputX, inputY, shape)));
        model.addLayer(new OutputSoftmax("Out", new TensorShape(inputX * inputY), classes, new CrossEntropy()));
        
        model.initialize(new Gaussian()); // initializing the weights
        System.out.println(model); // print summary of the model
        return model;    
    }
    
    public static void main (String [] args) throws IOException {
        new ConvolutionExperiment().go();
    }
}
