package nl.tue.s2id90.dl;

import static java.util.stream.Collectors.toList;
import java.util.stream.IntStream;

/**
 *
 * @author huub
 */
public class Util {
    
    /**
     * @param a integer array 
     * @return string representation of a.
     */
    public static String arrayToString(int[] a) {
        return IntStream.of(a).boxed().collect(toList()).toString();
    }
}
