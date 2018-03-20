package nl.tue.s2id90.dl.experiment;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.logging.Level;
import java.util.logging.Logger;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.javafx.Activations;
import nl.tue.s2id90.dl.javafx.GraphPanel;

/**
 * 
 * @author huub
 */
public class Experiment {   
    static final Logger LOGGER = Logger.getLogger(Experiment.class.getName());
    
    private GraphPanel accuracyGraph;
    private GraphPanel lossGraph;
    private GraphPanel validationGraph;
    private Activations activations;
    private BatchResult lastValidationResult;
 
    public Experiment() {
        this(false);    // don't startup GUI
    }
    
    public Experiment(boolean useGUI) { // create and show GUI
        if (useGUI) {
            
            // create javaFX widgets
            FXGUI fxGUI = FXGUI.getSingleton(); // initializes javafx platform!!
            fxGUI.setTitle(getClass().getSimpleName());
            fxGUI.addTab(accuracyGraph   = new GraphPanel("validation/batch")); 
            fxGUI.addTab(lossGraph       = new GraphPanel("loss/batch"));  
            fxGUI.addTab(validationGraph = new GraphPanel("test set validation/epoch")); 
            fxGUI.addTab(activations     = new Activations());
        }
    }
    
    /**
     * trains a neural network model, when finished saves the last model in the folder
     * "experiments/CLASSNAME-Model.json".
     * @param model       the neural network model
     * @param reader      the data source
     * @param sgd         the chosen optimizer, typically SGD
     * @param epochs      the number of epochs for training
     * @param activations every activations batches the activation of the network is made available to the gui;
     *                     if less or equal to zero, this is ignored.
     */
    public void trainAndSaveModel(Model model, InputReader reader, Optimizer sgd, int epochs, int activations) {
        try {
            trainModel(model, reader, sgd,epochs, activations);
        } finally {
            try {
                File file = Paths.get("experiments", getClass().getSimpleName()+"-Model.json").toFile();                
                LOGGER.log(Level.INFO, "Saving model to {0} ...", file.getCanonicalPath());
                model.saveModel(file);
            } catch (IOException ex) {
                LOGGER.severe(ex.getMessage());
            }
        }
    }

    /**
     * trains the neural network model.
     * @param model       the neural network model
     * @param reader      the data source
     * @param sgd         the chosen optimizer, typically SGD
     * @param epochs      the number of epochs for training
     * @param activations every activations batches the activation of the network is made available to the gui;
     *                     if less or equal to zero, this is ignored.
     */
    public void trainModel(Model model, InputReader reader, Optimizer sgd, int epochs, int activations) {
        // loop over all training batches for #epochs times
        for( int epoch = 1 ; epoch <= epochs ; epoch++ ){
            
            // iterator randomizes data and loops over all training data
            Iterator<TensorPair> batches = reader.getTrainingBatchIterator();
            
            // loop over all training batches
            System.out.format( "\nTraining epoch %d ..,\n", epoch);
            while( batches.hasNext() ){
                TensorPair batch = batches.next();
                
                // train model with one batch
                model.setInTrainingMode(true);
                BatchResult result = sgd.trainOnBatch(batch);
                
                onBatchFinished(model, reader, sgd, epochs, activations, batch, result);
            }
            
            onEpochFinished(model, reader, sgd, epochs, activations, epoch);
        }
        
        System.out.format( "Training of model finished after %d epochs.\n.", epochs );
    }
    
    /** called after a batch has been trained in method trainModel().
     * The arguments are the arguments as presented to trainModel, supplemented with
     * the last batch and its result.
     * @param model       the neural network model
     * @param reader      the data source
     * @param sgd         the chosen optimizer, typically SGD
     * @param epochs      the number of epochs for training
     * @param activations every activations batches the activation of the network is made available to the gui;
     *                     if less or equal to zero, this is ignored.
     * @param batch       the last batch handled during training.
     * @param result      the result of the last batch training.
     */
    public void onBatchFinished(Model model, InputReader reader, Optimizer sgd, int epochs, int activations, TensorPair batch, BatchResult result) {
        System.out.println(result);

        // add to gui
        addTrainingResults(result);
        if (activations > 0 && result.batch_id % activations == 0) {
            model.setInTrainingMode(false);
            ActivationData activation = model.getActivations(batch, result.batch_id);
            addActivations(activation);
        }
    }
    
    /** called after an epoch has been trained in method trainModel.
     * The arguments are the arguments as presented to trainModel, supplemented with
     * the index of the last finished epoch..
     * 
     * @param model       the neural network model
     * @param reader      the data source
     * @param sgd         the chosen optimizer, typically SGD
     * @param epochs      the number of epochs for training
     * @param activations every activations batches the activation of the network is made available to the gui;
     *                     if less or equal to zero, this is ignored.
     * @param epoch       index of the last finished epoch.
     */
    public void onEpochFinished(Model model, InputReader reader, Optimizer sgd, int epochs, int activations, int epoch) {
        // validate model after each epoch
        System.out.println("\nValidating ...");
        model.setInTrainingMode(false);
        lastValidationResult = sgd.validate(reader.getValidationData());
        System.out.format("Validation after epoch %3d: %s \n", epoch, lastValidationResult);

        // add to gui
        addValidationResult(lastValidationResult);
    }
    
    /** returns the result of the last validation, or null if no validation was performed.
      *@return BatchResult 
      */
    public BatchResult getLastValidationResult() {
        return lastValidationResult;
    }
    

    private void addTrainingResults(BatchResult result) {
        if (accuracyGraph!=null) accuracyGraph.add(result.batch_id, result.validation);
        if (lossGraph!=null) lossGraph.add(result.batch_id, result.loss);
    }

    private void addValidationResult(BatchResult result) {
        if (validationGraph!=null) validationGraph.add(result.batch_id, result.validation);
    }
    
    private void addActivations(ActivationData activation) {
        activations.add(activation);
    }
}