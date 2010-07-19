import java.math.BigInteger;
import java.util.ArrayList;

public class fib {
    public static void main(String[] args) {
        ArrayList<BigInteger> l = new ArrayList<BigInteger>();
        l.add(BigInteger.ONE);
        l.add(BigInteger.ONE);
        int e = new Integer(args[0]).intValue();
        for (int i = 0; i < e; i++) {
            l.add(l.get(i).add(l.get(i+1)));
        }
    }
}