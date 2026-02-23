# Linear Level Ancestor Algorithm

A Java implementation of the **Level Ancestor Problem** solved in **constant time $O(1)$** using **linear space $O(N)$**. This project implements the state-of-the-art algorithm described by **Bender & Farach-Colton (2004)**, often referred to as "The Method of Four Russians" applied to trees.

## The Problem

Given a rooted tree $T$ with $N$ nodes, the **Level Ancestor Problem** asks to find the ancestor of a given node $u$ at a specific depth $d$.

* **Input:** A node $u$ and a target depth $d$ (where $d \le \text{depth}(u)$).
* **Output:** The unique ancestor of $u$ that is located at depth $d$.

### Why is this hard?
Standard approaches offer a trade-off:
* **Naive Parent Pointers:** $O(1)$ space, $O(N)$ query. (Too slow)
* **Binary Lifting (Jump Pointers):** $O(N \log N)$ space, $O(\log N)$ query. (Good, but not optimal)
* **Direct Lookup Table:** $O(N^2)$ space, $O(1)$ query. (Too much memory)

**The Goal:** $O(N)$ Preprocessing, $O(N)$ Space, and **$O(1)$ Query time**.

---

## The Solution (Algorithm Overview)

This implementation combines three powerful techniques to achieve theoretical optimality:

### 1. Ladder Decomposition (Long Path Decomposition)
The tree is decomposed into disjoint paths (ladders). Each path is extended upwards by a factor of 2.
* **Benefit:** Allows climbing $O(1)$ after a large jump.
* **Limitation:** Alone, it requires logarithmic jumps.

### 2. Jump Pointers (Binary Lifting)
Selected nodes ("Jump Nodes") store pointers to ancestors at distances $1, 2, 4, 8, \dots, 2^k$.
* **Benefit:** Allows covering half the remaining distance in one step.
* **Optimization:** We only store these pointers for a small subset of nodes ($N / \log N$), keeping memory linear.

### 3. Macro-Micro Tree Decomposition ("The Four Russians")
To achieve true linearity, the tree is split into two parts:
* **Macro Tree (Skeleton):** Contains only the top $\approx N / \log N$ nodes. We run the "heavy" algorithms (Jump Pointers) here. The reduced size allows us to spend more time per node without exceeding $O(N)$ total.
* **Micro Trees:** The remaining small subtrees at the bottom (size $< \frac{1}{4} \log N$). These are so small that we can precompute all possible "shapes" and store their answers in a lookup table.

---

## 🚀 Complexity Analysis

| Operation | Complexity | Explanation |
| :--- | :--- | :--- |
| **Preprocessing** | **$O(N)$** | We traverse the tree a constant number of times. The heavy Jump Pointers are built only for a small fraction of nodes ($N/\log N$). |
| **Query** | **$O(1)$** | No loops. The answer is found using bitwise operations (Micro) or 2 array lookups (Macro). |
| **Space** | **$O(N)$** | All auxiliary structures (arrays, tables) are proportional to the number of nodes. |

---
