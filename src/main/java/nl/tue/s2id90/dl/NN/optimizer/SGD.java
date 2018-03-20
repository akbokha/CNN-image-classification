package nl.tue.s2id90.dl.NN.optimizer;

import static java.lang.String.format;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import nl.tue.s2id90.dl.NN.Model;
import java.util.function.Supplier;
import static java.util.stream.Collectors.joining;
import lombok.Builder;
import lombok.NonNull;
import nl.tue.s2id90.dl.NN.error.IllegalInput;
import nl.tue.s2id90.dl.NN.layer.Layer;
import nl.tue.s2id90.dl.NN.optimizer.update.GradientDescent;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.validate.ZeroValidator;
import nl.tue.s2id90.dl.experiment.BatchResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import nl.tue.s2id90.dl.NN.validate.Validator;

/**
 * SGD
 * 
 * @author Roel van Engelen
 */
public class SGD extends Optimizer{
    @NonNull private  Model    model = null;
    @NonNull private  Validator validator = new ZeroValidator();
    @NonNull private Supplier<UpdateFunction> updateFunctionConstructor=GradientDescent::new;
     
    private  int   batch_id = 0;
    /**
     * Initialize new SGD optimizer
     * 
     * @param model         Model to be trained
     * @param learningRate learning rate
     * @param validator     model validator
     * @param updateFunction a constructor for an update function, e.g. GradientDescent::new 
     */
    @Builder private SGD(@NonNull Model model, Float learningRate, Validator validator, Supplier<UpdateFunction> updateFunction ){
        super();
        
        this.model                       = model;
        if (validator!=null) this.validator = validator;
        if (learningRate!=null) this.learningRate = learningRate;
        if (updateFunction!=null) this.updateFunctionConstructor = updateFunction;
    }
    
    /**
     * Train model with single batch
     * 
     * @param batch List with Tensor_Pair training batch
     * @return 
     * @throws IllegalInput
     */
    @Override
    public BatchResult trainOnBatch( TensorPair batch ) throws IllegalInput{
        
        int batch_size = batch.model_input.getValues().shape()[0];        
        
        if (!model.isInTrainingMode()) {
            throw new IllegalStateException("Model trained while not in training Mode");
        }
        
        float accuracy = 0;
        float loss     = 0;
        
        // calculate inference
        Tensor prediction = model.inference( batch.model_input );

        // check if prediction is correct
        accuracy += validator.validate( batch.model_output, prediction );

        // calculate loss
        loss += model.calculateLoss( prediction, batch.model_output );

        // list with all layers
        List<Layer> layers = model.getLayers();

        // calculate backpropagation of output layer
        INDArray back = batch.model_output.getValues();

        // loop over all layers from ( last -1 ) to first layer
        for( int x = layers.size() - 1 ; x > 0 ; x-- ){

            back = layers.get( x ).backpropagation( back );
        }
        
        // update layer weights with calculated gradients
        for( Layer layer : model.getLayers() ){
            
            layer.updateLayer(updateFunctionConstructor, learningRate, batch_size );
        }
                
        return new BatchResult( ++batch_id, accuracy, loss, learningRate );
    }
    
    /**
     * validate model accuracy
     * 
     * @param batch List with Tensor_Pair validation batch
     * @return 
     * @throws IllegalInput
     */
    @Override
    public BatchResult validate(List<TensorPair> batch ) throws IllegalInput{
        
        if (model.isInTrainingMode()) {
            throw new IllegalStateException("Model validated while in training Mode");
        }
        
        float accuracy = 0;
        
        // loop over batch and calculate accuracy
        for( TensorPair sample : batch ){
            
            // predict image classification
            Tensor output = model.inference( sample.model_input );
            
            // check if prediction is correct
            accuracy += validator.validate( sample.model_output, output );
        }
        
        // calculate accuracy
        // accuracy = correct predictions / batch size
        accuracy /= batch.size();
        
        // update validation graph
        // loss and learning rate are irrelevant
        return new BatchResult( batch_id, accuracy);
    }
    
    /** @return a map with named informational objects for this model. */
    public Map<String,Object> getInfoMap() {
        Map result = new LinkedHashMap<>();
        result.put("optimizer", "Stochastic Gradient Descent");
        result.put("Validator", validator.getClass().getSimpleName());
        result.put("update function", updateFunctionConstructor.get().getClass().getSimpleName());
        result.put("learning rate", learningRate);
        return result;
    }
    
    @Override
    public String toString() {
        return getInfoMap().entrySet().stream()
            .map(e->format("%-20s: %s",e.getKey(), e.getValue()))
            .collect(joining("\n"));
    }
}
