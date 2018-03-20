package nl.tue.s2id90.dl.javafx;

import java.util.List;
import static java.util.stream.Collectors.toList;
import javafx.application.Platform;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.text.Text;
import javafx.stage.Stage;

/**
 *
 * @author huub
 */
public class FXGUI extends FXBase {
    static FXGUI gui;
    TabPane tabPane = new TabPane();
    Stage stage;
    
    @Override public void init() {
        countDown(); // you have to call this!
    }

    @Override public void start(Stage primaryStage) {
        gui = this;
        this.stage = primaryStage;

        Scene scene = new Scene(tabPane, 300, 250);
        //tabPane.getTabs().add(new Tab("About",new Text("hello world")));
        primaryStage.setScene(scene);
        primaryStage.setOnCloseRequest(e -> System.exit(0));
        primaryStage.show();
        
        countDown();    // you have to do this!
    }
    
    public void setTitle(String title) {
        Platform.runLater(()->stage.setTitle(title));
    }
    
    public static boolean isAvailable() {
        return gui!=null;
    }
    
    public static FXGUI getSingleton() {
        if (gui==null) {
            FXBase.launchFXAndWait(FXGUI.class);
        }
        return gui;
    }
    
    /** Adds a tab with the given label and node.
     * @param label label of the tab
     * @param node  widget shown under the tab
     */
    public void addTab(String label, Node node) {
        Platform.runLater(()-> tabPane.getTabs().add(new Tab(label,node)));
    }
    
    /** convenience method for adding a tab with a GraphPanel.
     * @param gp panel, will be added with label gp.getLabel()
     */
    public void addTab(GraphPanel gp) {
        addTab(gp.getLabel(), gp.getNode()); 
    };
    
    /** convenience method for adding a tab with a GraphPanel.
     * @param activations Activations panel, will be added with label "activations".
     */
    public void addTab(Activations activations) {
        addTab("activations", activations); 
    };
}
