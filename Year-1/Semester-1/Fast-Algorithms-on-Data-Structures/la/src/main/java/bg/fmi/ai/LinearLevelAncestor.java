package bg.fmi.ai;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LinearLevelAncestor {

  // Nodes structure and data
  private final int nodesCount;
  private final List<List<Integer>> adjacencyList;
  private final int[] depth, parent, height;

  // Ladder
  private final int[] longPathChild;
  private final List<int[]> ladders;
  private final int[] ladderId, ladderPos;

  // Macro-Micro
  private final boolean[] isJumpNode;
  private final List<Integer> jumpNodesList;
  // Used to find the Jump node for a given Macro node
  private final int[] jumpNodeDescendant;
  private final int[][] jumpPointers;
  private final int logN;
  private final int microBlockSize; // B = logN / 4

  // Micro-Tree
  private final int[] microRoot;
  // Node index inside micro-tree
  private final int[] microDfsRank;
  // Map: [microId][localIndex] -> globalNodeId
  private final int[][] microToGlobal;
  private final int[] globalToLocalBuffer;
  // Contains only the unique shapes
  // Since the block is small, the possible shapes are very few
  private final List<int[][]> shapesLibrary;
  // The serial number of the shape
  private final  int[] microShapeId;
  private int currentShapeMask;

  public LinearLevelAncestor(int nodesCount) {

    this.nodesCount = nodesCount;
    this.adjacencyList = new ArrayList<>(nodesCount);
    for (int i = 0; i < nodesCount; i++) {
      adjacencyList.add(new ArrayList<>());
    }
    depth = new int[nodesCount];
    parent = new int[nodesCount];
    height = new int[nodesCount];
    longPathChild = new int[nodesCount];
    Arrays.fill(longPathChild, -1);
    ladderId = new int[nodesCount];
    ladderPos = new int[nodesCount];
    ladders = new ArrayList<>();

    isJumpNode = new boolean[nodesCount];
    jumpNodesList = new ArrayList<>();

    jumpNodeDescendant = new int[nodesCount];
    Arrays.fill(jumpNodeDescendant, -1);
    microRoot = new int[nodesCount];
    Arrays.fill(microRoot, -1);
    microDfsRank = new int[nodesCount];

    // Determining the block size according to the article (log N / 4)
    // log2N + 1 (because for 2^4 we need size of 5 0,1,2,3,4)
    if (nodesCount > 1) {
      logN = 32 - Integer.numberOfLeadingZeros(nodesCount);
    } else {
      logN = 1;
    }
    // TODO: When n is small number there aren't any micro nodes, manual change for testing purposes if needed
    microBlockSize = Math.max(1, logN / 4); // B

    jumpPointers = new int[nodesCount][];
    globalToLocalBuffer = new int[nodesCount];
    Arrays.fill(globalToLocalBuffer, -1);
    shapesLibrary = new ArrayList<>();
    microShapeId = new int[nodesCount];
    microToGlobal = new int[nodesCount][];
  }

  public void addEdge(int parent, int child) {
    adjacencyList.get(parent).add(child);
  }

  /**
   * Main preprocessing method - O(N)
   */
  public void preprocess(int root) {

    parent[root] = root; // Safety check

    // 1. Initial Analysis: Heights, Subtree Sizes, Ladders
    // This corresponds to the standard part of Lemma 7
    int[] successorsPerNode = new int[nodesCount];
    dfsBasic(root, 0, successorsPerNode);
    buildLadders(root);

    // 2. Identify Jump Nodes and Macro/Micro decomposition
    // According to Section 4.1: "maximally deep vertices having at least log n/4 descendants"
    identifyJumpNodes(root, successorsPerNode);

    // 3. Compute Jump Pointers only for Jump Nodes - O(N) total
    // According to Lemma 10
    buildSparseJumpPointers();

    // 4. Link Macro nodes to their Jump Descendants
    // According to Lemma 11
    dfsMacroLink(root);

    // 5. Process Micro-Trees (Encoding and Lookup Tables)
    // According to Section 4.3 and Lemma 12
    processMicroTrees(root);
  }

  private void dfsBasic(int node, int depth, int[] successorsPerNode) {

    this.depth[node] = depth;
    height[node] = 1;
    successorsPerNode[node] = 1;
    int maxH = -1;

    for (int child : adjacencyList.get(node)) {
      if (child == parent[node]) continue; // TODO: Do we need that check?
      parent[child] = node;
      dfsBasic(child, depth + 1, successorsPerNode);
      successorsPerNode[node] += successorsPerNode[child];

      if (height[child] > maxH) {
        maxH = height[child];
        longPathChild[node] = child;
      }
      height[node] = Math.max(height[node], height[child] + 1);
    }
  }

  private void buildLadders(int root) {

    int id = 0;

    for (int i = 0; i < nodesCount; i++) {
      boolean isHead = (i == root) || (longPathChild[parent[i]] != i);
      if (isHead) {
        List<Integer> path = new ArrayList<>();
        int current = i;
        while (current != -1) {
          path.add(current);
          current = longPathChild[current];
        }

        int length = path.size();
        int[] ladder = new int[length * 2];

        // Ancestors
        int anc = parent[path.getFirst()];
        for (int k = 0; k < length; k++) {
          ladder[length - 1 - k] = anc;
          anc = parent[anc];
        }

        // Path
        for (int k = 0; k < length; k++) {
          ladder[length + k] = path.get(k);
        }

        ladders.add(ladder);
        for (int k = 0; k < length; k++) {
          int node = path.get(k);
          ladderId[node] = id;
          ladderPos[node] = length + k;
        }
        id++;
      }
    }
  }

  //  Jump Nodes and Macro/Micro Split
  private void identifyJumpNodes(int node, int[] successorsPerNode) {

    // At the beginning, we assume that the current node is a leaf in the context of the Macro-skeleton
    boolean isLeafInMacroSense = true;

    for (int child : adjacencyList.get(node)) {
      if (child == parent[node]) continue; // TODO: Do we need that check?
      identifyJumpNodes(child, successorsPerNode);
      if (successorsPerNode[child] >= microBlockSize) {
        isLeafInMacroSense = false;
      }
    }

    // Def. 4.1
    if (successorsPerNode[node] >= microBlockSize && isLeafInMacroSense) {
      isJumpNode[node] = true;
      jumpNodesList.add(node);
    }
  }

  // Fill in Jump pointers for Jump nodes
  // Lemma 10 from article
  private void buildSparseJumpPointers() {
    for (int node : jumpNodesList) {
      jumpPointers[node] = new int[logN];
      // We use Ladders for direct filling
      for (int i = 0; i < logN; i++) {
        int dist = 1 << i;
        jumpPointers[node][i] = queryLadderOnly(node, dist);
      }
    }
  }

  // Utility method: search only by Ladder
  private int queryLadderOnly(int node, int distToNextPredecessor) {

    if (distToNextPredecessor == 0) return node;

    // If the jump goes beyond the root -> return the root (0)
    // This is important because parent[root] == root
    if (distToNextPredecessor >= depth[node]) return 0;

    int[] ladder = ladders.get(ladderId[node]);
    int position = ladderPos[node];

    // CHECK: Does the jump exceed the bounds of the ladder array
    if (position - distToNextPredecessor < 0) {
      // The ladder is too short for this jump.
      // 1. Jump to the highest point of this ladder (ladder[0])
      int topOfLadder = ladder[0];

      // 2. Calculate the distance covered to reach the top
      // 'position' is effectively the distance from the current node to the top of the ladder
      // 3. Recursively continue upwards from the top with the remaining distance
      return queryLadderOnly(topOfLadder, distToNextPredecessor - position);
    }

    // Standard case (O(1))
    return ladder[position - distToNextPredecessor];
  }

  // Lemma 11 from article
  // In the implementation, we simply check if there is a Jump Node below us
  private int dfsMacroLink(int node) {

    int foundJumpNode = -1;

    if (isJumpNode[node]) foundJumpNode = node;

    for (int child : adjacencyList.get(node)) {
      if (child == parent[node]) continue;
      int res = dfsMacroLink(child);
      // Use nearest jump node
      if (res != -1) foundJumpNode = res;
    }

    if (foundJumpNode != -1) {
      // Macro Node or Jump Node
      jumpNodeDescendant[node] = foundJumpNode;
      return foundJumpNode;
    } else {
      // Micro Node (no jump node underneath)
      return -1;
    }
  }

  // Micro Trees Processing
  private void processMicroTrees(int root) {

    List<Integer> microRoots = new ArrayList<>();
    for (int i = 0; i < nodesCount; i++) {
      if (jumpNodeDescendant[i] == -1) {
        if (i == root || jumpNodeDescendant[parent[i]] != -1) {
          microRoots.add(i);
        }
      }
    }

    Map<Integer, Integer> tempShapeRegistry = new HashMap<>();
    for (int mRoot : microRoots) {

      List<Integer> nodes = new ArrayList<>();
      currentShapeMask = 1; // anchor because of leading zeros
      dfsMicroBitmask(mRoot, nodes);
      int bitmaskCode = currentShapeMask;

      int shapeId;
      if (tempShapeRegistry.containsKey(bitmaskCode)) {
        shapeId = tempShapeRegistry.get(bitmaskCode);
      } else {
        shapeId = shapesLibrary.size();
        int[][] table = computeMicroTable(nodes);
        shapesLibrary.add(table);
        tempShapeRegistry.put(bitmaskCode, shapeId);
      }

      int[] mapping = new int[nodes.size()];
      for (int k = 0; k < nodes.size(); k++) {
        int node = nodes.get(k);
        microRoot[node] = mRoot;
        microShapeId[node] = shapeId;
        microDfsRank[node] = k;
        mapping[k] = node;
      }
      microToGlobal[mRoot] = mapping;
    }
  }

  private void dfsMicroBitmask(int node, List<Integer> nodes) {

    nodes.add(node);
    for (int child : adjacencyList.get(node)) {
      if (child == parent[node]) continue;
      // Continue only if the child is micro node
      if (jumpNodeDescendant[child] == -1) {
        currentShapeMask = (currentShapeMask << 1);
        dfsMicroBitmask(child, nodes);
        currentShapeMask = (currentShapeMask << 1) | 1;
      }
    }
  }

//  private void dfsMicro(int node, List<Integer> nodes, StringBuilder shape) {
//
//    nodes.add(node);
//    for (int child : adjacencyList.get(node)) {
//      if (child == parent[node]) continue;
//      // Continue only if the child is micro node
//      if (jumpNodeDescendant[child] == -1) {
//        shape.append('0'); // Down
//        dfsMicro(child, nodes, shape);
//        shape.append('1'); // Up
//      }
//    }
//  }

  // Calculates the table for a given shape (Brute force, but on a small size B)
  private int[][] computeMicroTable(List<Integer> nodes) {

    int size = nodes.size();
    int[][] table = new int[size][size + 1]; // [localNode][k-th ancestor]

    for (int i = 0; i < size; i++) {
      int node = nodes.get(i);
      globalToLocalBuffer[node] = i;
    }

    int[] localParent = new int[size];
    for (int i = 0; i < size; i++) {
      int node = nodes.get(i);
      int parent = this.parent[node];
      if (globalToLocalBuffer[parent] != -1) {
        // It's in the buffer -> we get its local number
        localParent[i] = globalToLocalBuffer[parent];
      } else {
        // Root of micro-tree points to self locally
        localParent[i] = i;
      }
    }

    for (int i = 0; i < size; i++) {
      int curr = i;
      for (int dist = 0; dist <= size; dist++) {
        table[i][dist] = curr;
        if (curr != localParent[curr]) {
          curr = localParent[curr];
        }
      }
    }

    for (int i = 0; i < size; i++) {
      int u = nodes.get(i);
      globalToLocalBuffer[u] = -1;
    }

    return table;
  }

  // --- QUERY: O(1) ---
  public int query(int u, int targetDepth) {

    if (depth[u] < targetDepth) return -1;
    if (depth[u] == targetDepth) return u;

    // CASE 1: We are inside a Micro-Tree
    if (jumpNodeDescendant[u] == -1) {
      int mRoot = microRoot[u];

      // Check if the target is within the same Micro-Tree
      if (targetDepth >= depth[mRoot]) {
        // The target is inside -> Use the Precomputed Lookup Table
        int dist = depth[u] - targetDepth;
        int shapeID = microShapeId[u];
        int localIdx = microDfsRank[u];

        // Retrieve the local index of the result from the "Four Russians" table
        int resultLocalIdx = shapesLibrary.get(shapeID)[localIdx][dist];

        // Map the local index back to the global Node ID
        return microToGlobal[mRoot][resultLocalIdx];

      } else {
        // The target is above the Micro-Tree -> Jump to the Macro Skeleton
        u = parent[mRoot];
        // --- FIX ---
        // Immediately check if the transition to the parent landed us exactly on the target.
        // This prevents 'dist' becoming 0 in the next step (Case 2), which would cause an IndexOutOfBoundsException.
        if (depth[u] == targetDepth) return u;
        // ----------------
      }
    }

    // CASE 2: We are at a Macro Node (or just transitioned from Micro)
    // We apply Theorem 8 logic (JumpDescendant + JumpPointer + Ladder)

    // Step A: Delegate to the nearest descendant Jump Node
    // (This node 'v' is guaranteed to have Jump Pointers initialized)
    int v = jumpNodeDescendant[u];
    int dist = depth[v] - targetDepth;

    // Step B: Use Jump Pointer (Largest power of 2)
    // Find k such that 2^k is the largest power of 2 fitting in 'dist'
    int k = Integer.numberOfTrailingZeros(Integer.highestOneBit(dist));
    int mid = jumpPointers[v][k];

    // Step C: Use Ladder to climb the remaining distance
    // The Ladder at 'mid' is guaranteed to cover the remaining height
    int remaining = depth[mid] - targetDepth;
    return queryLadderOnly(mid, remaining);
  }
}
