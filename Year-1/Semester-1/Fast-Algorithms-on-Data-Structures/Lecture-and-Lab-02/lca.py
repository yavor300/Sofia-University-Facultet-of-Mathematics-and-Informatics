class TreeNode:
  def __init__(self, val):
    self.val = val
    self.children = []

def build_sample_tree():
  A = TreeNode("A")
  B = TreeNode("B")
  C = TreeNode("C")
  D = TreeNode("D")
  E = TreeNode("E")
  F = TreeNode("F")

  A.children = [B, C]
  B.children = [D, E]
  C.children = [F]

  return A

def print_tree(node, level=0):
    print("  " * level + str(node.val))
    for child in node.children:
        print_tree(child, level + 1)

def euler_tour(root):
  E = []
  L = []
  label_to_index = {}
  R = []
  all_nodes = []
  def collect_nodes(node):
     all_nodes.append(node.val)
     for child in node.children:
        collect_nodes(child)
  
  collect_nodes(root)
  n = len(all_nodes)
  label_to_index = {label: i for i, label in enumerate(all_nodes)}
  R = [-1] * n

  def dfs(node, depth):
    idx = len(E)
    E.append(node.val)
    L.append(depth)
    node_idx = label_to_index[node.val]
    if R[node_idx] == -1:
      R[node_idx] = idx
  
    for child in node.children:
      dfs(child, depth + 1)
      E.append(node.val)
      L.append(depth)
  
  dfs(root, 0)
  return E, L, R, label_to_index

def build_rmq_naive(L):
    n = len(L)
    RMQ = [[0] * n for _ in range(n)]
    for i in range(n):
       RMQ[i][i] = i
       for j in range(i + 1, n):
          prev_min_idx = RMQ[i][j-1]
          RMQ[i][j] = prev_min_idx if L[prev_min_idx] <= L[j] else j
    return RMQ

def build_lca_naive(root):
   E, L, R, label_to_index = euler_tour(root)
   RMQ = build_rmq_naive(L)
   def lca(u_label, v_label):
      iu = label_to_index[u_label]
      iv = label_to_index[v_label]
      left = R[iu]
      right = R[iv]
      if left > right:
         left, right = right, left
      idx_min = RMQ[left][right]
      return E[idx_min]
   return lca, (E, L, R)

if __name__ == "__main__":
    root = build_sample_tree()
    print_tree(root)
    E, L, R, label_to_index = euler_tour(root)
    print(E)
    print(L)
    print(R)
    for label, idx in label_to_index.items():
        print(f"R[{label}] = {R[idx]}")
    lca, (E, L, R) = build_lca_naive(root)
    print("\nLCA(D, E) =", lca("D", "E"))
    print("LCA(D, F) =", lca("D", "F"))
    print("LCA(B, F) =", lca("B", "F"))
