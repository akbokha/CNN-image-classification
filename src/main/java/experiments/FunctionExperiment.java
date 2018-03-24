package experiments;
import nl.tue.s2id90.dl.experiment.Experiment;
import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.FullyConnected;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.loss.MSE;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Regression;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;

/**
 *
 * @author Abdel K. Bokharouss
 * @author Adriaan Knapen
 */
public class FunctionExperiment extends Experiment {
    // (hyper) parameters
    int batchSize = 16; // size of the batches in which the data is processed
    int epochs = 10; // number of epochs that a training takes
    float learningRate = 0.01f; // parameter for the gradient descent optimization method
    int numberOfNeuronsLayer = 8; // number of neurons of the layer
    int numberOfLayers = 10; // number of layers
    
    public void go() throws IOException {
        // read input and print information on the data
        InputReader reader = GenerateFunctionData.THREE_VALUED_FUNCTION(batchSize);
        System.out.println("Reader info:\n" + reader.toString());
        
        int inputs = reader.getInputShape().getNeuronCount();
        int outputs = reader.getOutputShape().getNeuronCount();
        Model model = createModel(inputs, outputs);
        
        Optimizer sgd = SGD.builder() // Stohastic Gradient Descent method
                .model(model)
                .validator(new Regression()) // computes mean squared error
                .learningRate(learningRate)
                .build();
        
        trainModel(model, reader, sgd, epochs, 0);
    }
    
    Model createModel(int inputs, int outputs) {
        /**
         * has an input layer of {@code: inputs} neurons
         * and is fully connected to an output layer that has {@code: inputs} connections
         * coming in to each of its {@code: outputs} neurons
         * The mean-squared-error function is employed as loss function
         */
        
        Model model = new Model(new InputLayer("In", new TensorShape(inputs), true));
   
        // FIRST EXAMINATION
//        model.addLayer(new SimpleOutput("Out", new TensorShape(inputs), outputs, new MSE(), true));
        
        // SECOND EXAMINATION: One additional layer
//        model.addLayer(new FullyConnected("fc1", new TensorShape(inputs), inputs, new RELU()));
//        model.addLayer(new SimpleOutput("Out", new TensorShape(inputs), outputs, new MSE(), true));
        
        // THIRD EXAMINATION: One additional layer: One additional layer with {@code: numberOfNeurons} neurons
//        model.addLayer(new FullyConnected("fc1", new TensorShape(inputs), numberOfNeuronsLayer, new RELU()));
//        model.addLayer(new SimpleOutput("Out", new TensorShape(numberOfNeuronsLayer), outputs, new MSE(), true));

        // FOURTH EXAMINATION: Multiple layers
        model.addLayer(new FullyConnected("fc1", new TensorShape(inputs), numberOfNeuronsLayer, new RELU()));
        for (int i = 0; i < numberOfLayers - 1; i++) {
            model.addLayer(new FullyConnected(String.format("fc%s", Integer.toString(i + 2)), new TensorShape(numberOfNeuronsLayer), numberOfNeuronsLayer, new RELU()));
        }
        model.addLayer(new SimpleOutput("Out", new TensorShape(numberOfNeuronsLayer), outputs, new MSE(), true));
        model.initialize(new Gaussian()); // initializing the weights
        System.out.println(model); // print summary of the model
        return model;
    }
    
    public static void main (String [] args) throws IOException {
        new FunctionExperiment().go();
    }
}
