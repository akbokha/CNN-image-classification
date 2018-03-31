package experiments;

import nl.tue.s2id90.dl.NN.activation.Activation;
import nl.tue.s2id90.dl.NN.initializer.Initializer;
import nl.tue.s2id90.dl.NN.layer.Layer;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Abdel K. Bokharouss
 * @author Adriaan Knapen
 */
public class DropOut extends Layer {
    
    private float probability;
    private final int inputs;
    protected     INDArray dropO;
    private boolean show_values = false;
    
    /**
     * Create Fully connected layer
     * 
     * @param name        layer name
     * @param input_shape 1D Tensor input shape 
     * @param p           dropout-probability
     */
    @SuppressWarnings("empty-statement")
    public DropOut(String name, TensorShape input_shape, float p){
        super(name, input_shape, input_shape);
        this.probability = p;
        // check that input shape is 1D
        if( input_shape.is3D() ){}; // see 
        this.inputs     = input_shape.getNeuronCount();
    }
    
    public DropOut(String name, TensorShape input_shape, float p, boolean showValues) {
        this(name, input_shape, p);
        this.show_values = showValues;
    }
    
    @Override
    public void updateLayer(float learning_rate, int batch_size){};
    
    @Override
    public INDArray backpropagation(INDArray input) {
        return input.muli(dropO);
    }
    
    @Override
    public Tensor inference(Tensor input) {
        INDArray inputValues = input.getValues();
        int [] input_shape = inputValues.shape();
        if(dropO == null) {
            dropO = Nd4j.rand(input_shape, Nd4j.getDistributions().createBinomial(1, this.probability)).divi(this.probability);
        }
        inputValues.muli(dropO);
        return new Tensor(inputValues, super.outputShape);
    }

    @Override
    public boolean showValues() {
        return this.show_values;
    }

    @Override
    public void initializeLayer(Initializer initializer) {};
}
