"""Train and persist node2vec embeddings for Article-MeSH graphs."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from gensim.models import KeyedVectors
from node2vec import Node2Vec


class Node2VecTrainer:
    def __init__(
        self,
        dimensions: int = 128,
        walk_length: int = 20,
        num_walks: int = 100,
        workers: int = 2,
        window: int = 10,
        min_count: int = 1,
    ):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.window = window
        self.min_count = min_count

    def train(self, graph: nx.Graph):
        """Train node2vec and return gensim KeyedVectors embeddings."""
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot train node2vec on an empty graph.")

        node2vec = Node2Vec(
            graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            quiet=False,
        )
        model = node2vec.fit(
            window=self.window,
            min_count=self.min_count,
            batch_words=4,
        )
        return model.wv

    def save(self, model, path: str) -> None:
        """Save KeyedVectors, or a gensim Word2Vec model's .wv, to disk."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        keyed_vectors = model.wv if hasattr(model, "wv") else model
        keyed_vectors.save(str(output_path))

    def load(self, path: str):
        return KeyedVectors.load(str(path), mmap="r")
