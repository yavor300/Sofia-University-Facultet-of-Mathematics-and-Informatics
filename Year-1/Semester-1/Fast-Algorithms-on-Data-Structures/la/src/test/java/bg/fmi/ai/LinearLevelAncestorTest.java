package bg.fmi.ai;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import java.util.Random;

public class LinearLevelAncestorTest {

  /**
   * Linear tree (List)
   * 0 -> 1 -> 2 -> 3 -> 4 -> 5
   */
  @Test
  public void testLinearPath() {

    int n = 6;
    LinearLevelAncestor la = new LinearLevelAncestor(n);

    for (int i = 0; i < n - 1; i++) {
      la.addEdge(i, i + 1);
    }

    la.preprocess(0);

    assertEquals(0, la.query(5, 0));
    assertEquals(3, la.query(5, 3));
    assertEquals(4, la.query(4, 4));

    assertEquals(-1, la.query(2, 5));
  }

  /**
   *   0
   * /   \
   * 1     2
   * / \   / \
   * 3   4 5   6
   */
  @Test
  public void testBinaryTree() {
    int n = 7;
    LinearLevelAncestor la = new LinearLevelAncestor(n);

    la.addEdge(0, 1); la.addEdge(0, 2);
    la.addEdge(1, 3); la.addEdge(1, 4);
    la.addEdge(2, 5); la.addEdge(2, 6);

    la.preprocess(0);

    assertEquals(0, la.query(3, 0));
    assertEquals(1, la.query(3, 1));
    assertEquals(3, la.query(3, 2));

    assertEquals(0, la.query(6, 0));
    assertEquals(2, la.query(6, 1));
  }

  /**
   * Deep Tree (Micro/Macro/Jump Logic)
   * Goal: To force the algorithm to use:
   * 1. Micro Table (for leaf nodes)
   * 2. Jump Pointers (for traversing large distances)
   * 3. Ladders (for finalizing the search)
   */
  @Test
  public void testDeepTreeLogic() {

    int n = 25;
    LinearLevelAncestor la = new LinearLevelAncestor(n);

    for(int i = 0; i < 19; i++) {
      la.addEdge(i, i + 1);
    }

    la.addEdge(5, 20);
    la.addEdge(10, 21); la.addEdge(21, 22);
    la.addEdge(18, 23); la.addEdge(23, 24);

    la.preprocess(0);

    assertEquals(23, la.query(24, 19));

    assertEquals(10, la.query(24, 10));

    assertEquals(0, la.query(24, 0));
    assertEquals(0, la.query(19, 0));
  }

  /**
   * "Star" Graph (Wide, shallow tree)
   * Structure: Node 0 is connected to all other nodes (1..N-1).
   * Goal: Tests if the algorithm works correctly when the tree has minimal depth.
   */
  @Test
  public void testStarGraph() {

    int n = 10;
    LinearLevelAncestor la = new LinearLevelAncestor(n);

    for (int i = 1; i < n; i++) {
      la.addEdge(0, i);
    }

    la.preprocess(0);

    for (int i = 1; i < n; i++) {
      assertEquals(0, la.query(i, 0), "Parent of " + i + " should be 0");
      assertEquals(i, la.query(i, 1), "Node " + i + " at depth 1 should be itself");
    }
  }

  @Test
  public void testStructureWithLongSpine() {

    int n = 16;
    LinearLevelAncestor la = new LinearLevelAncestor(n);
    //       0
    //     /   \
    //    1     2
    //   / \
    //  3   4
    //      |
    //      5
    //     / \
    //    6   7
    //       / \
    //      8   9
    //          |
    //          10
    //          |
    //          11 -> 12 -> 13 -> 14 -> 15

    la.addEdge(0, 1); la.addEdge(0, 2);
    la.addEdge(1, 3); la.addEdge(1, 4);
    la.addEdge(4, 5);
    la.addEdge(5, 6); la.addEdge(5, 7);
    la.addEdge(7, 8); la.addEdge(7, 9);
    la.addEdge(9, 10);
    la.addEdge(10, 11);
    la.addEdge(11, 12);
    la.addEdge(12, 13);
    la.addEdge(13, 14);
    la.addEdge(14, 15);

    la.preprocess(0);

    assertEquals(0, la.query(2, 0));
    assertEquals(1, la.query(3, 1));
    assertEquals(0, la.query(3, 0));

    assertEquals(4, la.query(6, 2));
    assertEquals(5, la.query(8, 3));

    assertEquals(14, la.query(15, 10));
    assertEquals(13, la.query(15, 9));
    assertEquals(9, la.query(15, 5));
    assertEquals(7, la.query(15, 4));
    assertEquals(1, la.query(15, 1));
    assertEquals(0, la.query(15, 0));

    assertEquals(15, la.query(15, 11));
    assertEquals(-1, la.query(15, 12));
    assertEquals(10, la.query(10, 6));
  }

  /**
   * TEST: Asymmetric Branching Tree
   * N=13
   * <p>
   * Visual Representation:
   * 0
   * |
   * 1
   * /   \
   * 2     12
   * / \
   * 3   9
   * / \   \
   * 4   7   10
   * |   |    |
   * 5   8   11
   * |
   * 6
   * <p>
   * Max Depth: 6 (Node 6)
   */
  @Test
  public void testCustomAsymmetricStructure() {

    int n = 13;
    LinearLevelAncestor la = new LinearLevelAncestor(n);

    la.addEdge(0, 1);

    la.addEdge(1, 2);
    la.addEdge(1, 12);

    la.addEdge(2, 3);
    la.addEdge(2, 9);

    la.addEdge(3, 4);
    la.addEdge(3, 7);
    la.addEdge(9, 10);

    la.addEdge(4, 5);
    la.addEdge(7, 8);
    la.addEdge(10, 11);

    la.addEdge(5, 6);

    la.preprocess(0);

    assertEquals(6, la.query(6, 6));
    assertEquals(5, la.query(6, 5));
    assertEquals(3, la.query(6, 3));
    assertEquals(2, la.query(6, 2));
    assertEquals(1, la.query(6, 1));
    assertEquals(0, la.query(6, 0));

    assertEquals(7, la.query(8, 4));
    assertEquals(3, la.query(8, 3));
    assertEquals(2, la.query(8, 2));

    assertEquals(10, la.query(11, 4));
    assertEquals(9, la.query(11, 3));
    assertEquals(2, la.query(11, 2));
    assertEquals(1, la.query(11, 1));

    assertEquals(1, la.query(12, 1));
    assertEquals(0, la.query(12, 0));

    assertEquals(2, la.query(5, 2));

    assertEquals(1, la.query(7, 1));

    assertEquals(-1, la.query(12, 3));
    assertEquals(-1, la.query(0, 1));
  }

  @Test
  public void testBranchesWithThreeChildrenAndShortLeavesAndDeepPaths() {

    int n = 17;
    LinearLevelAncestor la = new LinearLevelAncestor(n);

    la.addEdge(0, 1);
    la.addEdge(0, 2);
    la.addEdge(0, 3);

    la.addEdge(1, 4);
    la.addEdge(3, 7);
    la.addEdge(2, 5);
    la.addEdge(2, 6);

    la.addEdge(4, 8);
    la.addEdge(5, 9);
    la.addEdge(5, 10);

    la.addEdge(9, 11);
    la.addEdge(9, 12);
    la.addEdge(9, 13);

    la.addEdge(10, 14);

    la.addEdge(13, 15);
    la.addEdge(14, 16);

    la.preprocess(0);

    assertEquals(4, la.query(8, 2));
    assertEquals(1, la.query(8, 1));
    assertEquals(0, la.query(8, 0));

    assertEquals(3, la.query(7, 1));
    assertEquals(0, la.query(7, 0));

    assertEquals(2, la.query(6, 1));
    assertEquals(0, la.query(6, 0));

    assertEquals(9, la.query(11, 3));
    assertEquals(9, la.query(12, 3));
    assertEquals(9, la.query(15, 3));

    assertEquals(13, la.query(15, 4));
    assertEquals(9, la.query(15, 3));
    assertEquals(5, la.query(15, 2));
    assertEquals(2, la.query(15, 1));

    assertEquals(14, la.query(16, 4));
    assertEquals(10, la.query(16, 3));
    assertEquals(5, la.query(16, 2));

    int ancestor15 = la.query(15, 2);
    int ancestor16 = la.query(16, 2);
    assertEquals(5, ancestor15);
    assertEquals(5, ancestor16);
    assertEquals(ancestor15, ancestor16);

    assertEquals(15, la.query(15, 5));
    assertEquals(-1, la.query(15, 6));
    assertEquals(0, la.query(0, 0));
  }

  @Test
  public void testLongPathDecomposition() {

    int n = 16;
    LinearLevelAncestor la = new LinearLevelAncestor(n);

    la.addEdge(0, 1);
    la.addEdge(0, 2);
    la.addEdge(0, 3);

    la.addEdge(1, 4);
    la.addEdge(1, 5);
    la.addEdge(4, 9);

    la.addEdge(2, 6);
    la.addEdge(6, 10);
    la.addEdge(10, 12);
    la.addEdge(10, 13);
    la.addEdge(13, 15);

    la.addEdge(3, 7);
    la.addEdge(3, 8);
    la.addEdge(7, 11);

    la.preprocess(0);

    assertEquals(0, la.query(15, 0));
    assertEquals(2, la.query(15, 1));
    assertEquals(6, la.query(15, 2));
    assertEquals(10, la.query(15, 3));
    assertEquals(13, la.query(15, 4));
    assertEquals(15, la.query(15, 5));

    assertEquals(0, la.query(9, 0));
    assertEquals(1, la.query(9, 1));
    assertEquals(4, la.query(9, 2));
    assertEquals(9, la.query(9, 3));

    assertEquals(0, la.query(12, 0));
    assertEquals(2, la.query(12, 1));
    assertEquals(6, la.query(12, 2));
    assertEquals(10, la.query(12, 3));

    assertEquals(0, la.query(11, 0));
    assertEquals(3, la.query(11, 1));
    assertEquals(7, la.query(11, 2));

    assertEquals(1, la.query(5, 1));
    assertEquals(3, la.query(8, 1));

    assertEquals(-1, la.query(15, 6));
    assertEquals(-1, la.query(0, 1));
  }

  @Test
  public void testWorstCasePathDecompositionStructure() {

    int n = 21;
    LinearLevelAncestor la = new LinearLevelAncestor(n);

    la.addEdge(0, 1);
    la.addEdge(1, 2);
    la.addEdge(2, 3);
    la.addEdge(3, 4);
    la.addEdge(4, 5);

    la.addEdge(0, 6);
    la.addEdge(6, 7);
    la.addEdge(7, 8);
    la.addEdge(8, 9);
    la.addEdge(9, 10);

    la.addEdge(6, 11);
    la.addEdge(11, 12);
    la.addEdge(12, 13);
    la.addEdge(13, 14);

    la.addEdge(11, 15);
    la.addEdge(15, 16);
    la.addEdge(16, 17);

    la.addEdge(15, 18);
    la.addEdge(18, 19);

    la.addEdge(18, 20);

    la.preprocess(0);

    assertEquals(0, la.query(5, 0));
    assertEquals(1, la.query(5, 1));
    assertEquals(4, la.query(5, 4));
    assertEquals(5, la.query(5, 5));

    assertEquals(0, la.query(10, 0));
    assertEquals(6, la.query(10, 1));
    assertEquals(7, la.query(10, 2));
    assertEquals(10, la.query(10, 5));

    assertEquals(0, la.query(14, 0));
    assertEquals(6, la.query(14, 1));
    assertEquals(11, la.query(14, 2));
    assertEquals(12, la.query(14, 3));
    assertEquals(14, la.query(14, 5));

    assertEquals(0, la.query(17, 0));
    assertEquals(6, la.query(17, 1));
    assertEquals(11, la.query(17, 2));
    assertEquals(15, la.query(17, 3));
    assertEquals(16, la.query(17, 4));
    assertEquals(17, la.query(17, 5));

    assertEquals(0, la.query(19, 0));
    assertEquals(6, la.query(19, 1));
    assertEquals(11, la.query(19, 2));
    assertEquals(15, la.query(19, 3));
    assertEquals(18, la.query(19, 4));
    assertEquals(19, la.query(19, 5));

    assertEquals(0, la.query(20, 0));
    assertEquals(6, la.query(20, 1));
    assertEquals(11, la.query(20, 2));
    assertEquals(15, la.query(20, 3));
    assertEquals(18, la.query(20, 4));
    assertEquals(20, la.query(20, 5));

    assertEquals(-1, la.query(5, 6));
    assertEquals(-1, la.query(20, 6));
    assertEquals(-1, la.query(0, 1));
  }

  @Test
  public void testLargeRandomTreeWithSequentialIds() {

    int n = 1000;
    LinearLevelAncestor la = new LinearLevelAncestor(n);

    int[] verificationParents = new int[n];
    verificationParents[0] = 0;

    // fixed seed
    Random rand = new Random(42);

    for (int i = 1; i < n; i++) {
      int parent = rand.nextInt(i);
      la.addEdge(parent, i);
      verificationParents[i] = parent;
    }


    assertDoesNotThrow(() -> la.preprocess(0));

    int queriesCount = 100_000;

    for (int k = 0; k < queriesCount; k++) {

      int u = rand.nextInt(n);

      int trueDepth = getDepthNaive(u, verificationParents);
      if (trueDepth == 0) {
        assertEquals(0, la.query(u, 0));
        continue;
      }

      int targetDepth = rand.nextInt(trueDepth + 1);
      int fastResult = la.query(u, targetDepth);

      int expectedResult = getAncestorNaive(u, targetDepth, verificationParents);
      assertEquals(expectedResult, fastResult,
        String.format("Mismatch at iter %d: Node %d, Depth %d, Target %d",
          k, u, trueDepth, targetDepth));
    }
  }

  @Test
  public void testWithN257_EnsuringBlockSizeTwo() {

    int n = 257;
    LinearLevelAncestor la = new LinearLevelAncestor(n);

    int[] verificationParents = new int[n];
    verificationParents[0] = 0;

    Random rand = new Random(257);

    for (int i = 1; i < n; i++) {
      int parent = rand.nextInt(i); // p < i
      la.addEdge(parent, i);
      verificationParents[i] = parent;
    }

    assertDoesNotThrow(() -> la.preprocess(0));

    int queriesCount = 5000;

    for (int k = 0; k < queriesCount; k++) {
      int u = rand.nextInt(n);

      int trueDepth = getDepthNaive(u, verificationParents);

      if (trueDepth == 0) {
        assertEquals(0, la.query(u, 0));
        continue;
      }

      int targetDepth = rand.nextInt(trueDepth + 1);

      int fastResult = la.query(u, targetDepth);
      int expectedResult = getAncestorNaive(u, targetDepth, verificationParents);

      assertEquals(expectedResult, fastResult,
        String.format("Грешка при N=257! Възел %d (дълб. %d) към цел %d",
          u, trueDepth, targetDepth));
    }
  }

  private int getDepthNaive(int u, int[] parents) {

    int d = 0;
    while (u != 0) {
      u = parents[u];
      d++;
    }
    return d;
  }

  private int getAncestorNaive(int u, int targetDepth, int[] parents) {

    int curr = u;
    int currentDepth = getDepthNaive(u, parents);

    while (currentDepth > targetDepth) {
      curr = parents[curr];
      currentDepth--;
    }
    return curr;
  }
}
