# Temporal Graph Network (TGN)

> A minimal and clean PyTorch Geometric implementation of **Temporal Graph Networks (TGN)** for beginners and researchers exploring dynamic graph neural networks.

---


This project demonstrates the **core ideas** of TGN — a framework for learning on dynamic and temporal graphs by combining node embeddings with memory states.

TGN is especially useful for:
- Evolving graph data (e.g., financial transactions, user-item interactions)
- Temporal message passing and memory-augmented GNNs

---

## Features

- Lightweight & educational: no external datasets required
- Clean modular design (model + example runner)
- Demonstrates how to combine node features with temporal memory
- Built using `torch` and `torch-geometric`

---

## Installation 

```bash
pip install torch torch-geometric
```
---

## How to Start
Run the example script:

```bash
python run_example.py
```

This will:
- Initialize a random graph of 100 nodes and 150 edges
- Assign each node a random feature and memory vector
- Run TGN message passing on nodes [0, 1, 2]
- Output their updated embeddings

## Core Idea (How it Works)
Each node has:
An input feature vector, A memory vector

The TGN does:

1.Filter edge_index to include only selected nodes

2.For each message:

    Message = [sender_feature | sender_memory]

3.Aggregate all messages using mean

4.Use a multi-layer perceptron (MLP) to produce updated node embeddings

The output is the new embedding for the selected node subset.

## References
- Rossi et al., 2020 - Temporal Graph Networks
- PyTorch Geometric Documentation

## Contributing
This is an introductory implementation. Feel free to open issues or pull requests to:

- Add memory update mechanisms (e.g., GRU-based)
- Integrate real datasets
- Extend temporal handling



