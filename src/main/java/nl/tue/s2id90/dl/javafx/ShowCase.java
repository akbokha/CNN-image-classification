package nl.tue.s2id90.dl.javafx;

import java.awt.image.BufferedImage;
import static java.lang.String.format;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import static java.util.stream.Collectors.toList;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Node;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Text;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.Nd4jUtil;
import static nl.tue.s2id90.dl.javafx.Images.getFXImage;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Shows image, label and scores of a list of Tensor_Pairs.
 * @author huub
 */
public class ShowCase {
    Function<Integer,String> toString;
    private final FlowPane flowPane;    // javafx node containing images and the like.
    public ShowCase(Function<Integer,String> toString) {
        this.toString = toString;
        
        /* initializes main layout: FlowPane */
        flowPane = new FlowPane();
        flowPane.setVgap(15);
        flowPane.setHgap(15);
    }
    
    /** 
     * @return  javafx node containing this showCase.
     */
    public Node getNode() {
        ScrollPane pane = new ScrollPane(flowPane);
        pane.setFitToWidth(true);
        
        // make flowPane as wide as possible without need for scrolling horizontally
        //flowPane.prefWrapLengthProperty().bind(pane.widthProperty());
        
        return pane;
    }
    
    /** adds gui representations (Items) of the tensor pairs to the flow pane.
     * @param pairs list of tensor pairs
     **/
    public void setItems(List<TensorPair> pairs) {
        Platform.runLater( () ->
            flowPane.getChildren().setAll(
                pairs.stream()
                .sorted((o1,o2)-> Integer.compare(
                        label(o1.model_output.getValues()),
                        label(o2.model_output.getValues()))
                ).map(tp-> new Item(
                        image(tp.model_input),
                        label(tp.model_output.getValues()),
                        scores(tp.model_output.getValues())))
                .collect(toList())
                )
        );
    }
    
    /** converts tensor to a javafx image. */
    private Image image(Tensor t) {
        return getFXImage(Images.image_from_tensor_3d(t));
    }
    
    /** computes label id. */
    private int label(INDArray array) {
        return Nd4jUtil.argMax(array);
    }
    
    /** converts class scores to list of floats. */
    private List<Float> scores(INDArray array) {
        List<Float> result = new ArrayList<>();
        INDArray a = array.reshape(1,array.length());
        for(int i=0;i<a.length();i++) {
            result.add(a.getFloat(i));
        }
        return result;
    }
    
    /** widget for showing image, label and label scores. */
    class Item extends VBox {
        public Item(Image image, int label, List<Float> scores) {
            String labelString = format("%d - %s", label, toString.apply(label) );
            getChildren().addAll(
                new ImageView(image), 
                rectangles(label,scores),
                new Text(labelString)
            );
            this.setSpacing(2);
        }
    }
    
    /** creates horizontal histogram. */
    private static HBox rectangles(int label, List<Float> scores) {
        List<Rectangle> rectangles = new ArrayList<>();
        for(int i=0; i<scores.size(); i = i+1) {
            Float s = scores.get(i);
            rectangles.add(new Rectangle(8,Math.max(1,25*s),i==label?Color.GREEN:Color.RED));
        }
        HBox box = new HBox(rectangles.toArray(new Rectangle[0]));
        box.setSpacing(2);
        return box;
    } 
}
