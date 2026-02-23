package bg.fmi.ai;

public class Main {

  public static void main(String[] args) {

    int n = 13;
    LinearLevelAncestor la = new LinearLevelAncestor(n);
    la.addEdge(0, 1);
    la.addEdge(1, 2);
    la.addEdge(2, 3);
    la.addEdge(3, 4);
    la.addEdge(4, 5);
    la.addEdge(5, 6);

    la.addEdge(3, 7);
    la.addEdge(7, 8);

    la.addEdge(2, 9);
    la.addEdge(9, 10);
    la.addEdge(10, 11);
    la.addEdge(1, 12);

    la.preprocess(0);

    System.out.println(la.query(8, 4));
    System.out.println(la.query(6, 1));
  }
}
