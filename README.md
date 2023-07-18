# MolGAT Model

The MolGAT model is a deep learning model for molecular property prediction. It is based on the Graph Attention Networks (GAT) architecture and is designed to work with molecular graphs.

Unlike the GAT model, the MolGAT model takes n-dimensional edge features as input, which allows it to incorporate additional information about the chemical bonds in the molecular graph.

This model takes as input a molecular graph represented as a node feature matrix, an edge feature matrix, and an adjacency matrix, and learns to predict a molecular property. It consists of several Graph Attention Layers (GATs) followed by fully connected layers, and its architecture is defined by the MolGAT class.




The __init__ method of the MolGAT class initializes the layers of the model. It takes as input the following parameters:

- `node_features`: the dimensionality of the input node features.
- `hidden_dim`: the dimensionality of the hidden representations.
- `edge_features`: the dimensionality of the input edge features.
- `num_heads`: the number of attention heads used in each GAT layer.
- `dropout`: the dropout probability used in the GAT layers.
- `num_conv_layers`: the number of GAT layers.
- `num_fc_layers`: the number of fully connected layers.

The forward method of the MolGAT class performs a forward pass through the layers of the model. It takes as input the following arguments:

- `x`: the node feature matrix of shape (num_nodes, node_features).
- `edge_index`: the edge index tensor of shape (2, num_edges).
- `batch_index`: the batch index tensor of shape (num_nodes,).
- `edge_attr`: the edge feature matrix of shape (num_edges, edge_features).

The forward method performs the following steps:

- Applies a GAT layer to the input node features and edge features.
- ReLU activation function is applied to the output of the GAT layer.
- Dropout is applied to the output of the GAT layer.
- Graph-level max pooling and mean pooling are applied to the output of the GAT layer.
- The output of the GAT layer is concatenated with the output of the graph-level pooling.
- Fully connected layers with ReLU activation functions are applied to the concatenated output.
- Dropout is applied to the output of the fully connected layers.
- The output of the fully connected layers is passed through a linear layer to obtain the final output.

Overall, the MolGAT model learns a hierarchical representation of the molecular graph using GAT layers and fully connected layers, and uses attention-based graph aggregation to summarize the node-level features and generate a graph-level representation, which is then used to make a prediction of the molecular property.


## Requirements

The following packages are required to run the MolGAT model:

- Python 3.x
- PyTorch
- RDKit
- PyTorch_geometric(PyG)
- numpy
- pandas
- scikit-learn


## Model Architecture

The MolGAT model consists of multiple graph attention convolutional layers, followed by fully connected layers. The number of convolutional layers and fully connected layers can be specified as input parameters. Each convolutional layer is followed by a batch normalization layer and ReLU activation. The output of the last convolutional layer is concatenated with the global maximum pooling and global average pooling of node features, and passed through the fully connected layers to obtain the final prediction.

## Training

The model is trained using mean squared error (MSE) loss, and optimized using the Adam optimizer with a custom learning rate schedule. The training and testing data are loaded using PyTorch data loaders. During training, the model is put into training mode, and gradients are computed and updated after each batch. During testing, the model is put into evaluation mode, and no gradients are computed.

## Usage

To use the MolGAT model, create an instance of the MolGAT class with the desired input parameters. Then, train the model using the train function, passing in the training data loader. To evaluate the model on test data, call the test function, passing in the test data loader.

Example usage:


```

edge_attr = mol_reddb.data.edge_attr.shape[1]
num_features=mol_reddb.num_features

# Initialize model
model = MolGAT(node_features=num_features, hidden_dim=512, edge_features=edge_attr , num_heads=4, dropout=0.1, num_conv_layers=3, num_fc_layers=3)

# Train model
train_loss = []
test_loss = []
for epoch in range(1, args.epochs):
    train_mse = train(train_loader)
    test_mse = test(test_loader)
    train_loss.append(train_mse)
    test_loss.append(test_mse)
    if epoch % 1 == 0:
        print(f'Epoch: {epoch:d}, Loss: {train_mse:.7f}, test MSE: {test_mse:.7f}')
```





=======
# MolGAT Model

The MolGAT model is a deep learning model for molecular property prediction. It is based on the Graph Attention Networks (GAT) architecture and is designed to work with molecular graphs.

Unlike the GAT model, the MolGAT model takes n-dimensional edge features as input, which allows it to incorporate additional information about the chemical bonds in the molecular graph.

This model takes as input a molecular graph represented as a node feature matrix, an edge feature matrix, and an adjacency matrix, and learns to predict a molecular property. It consists of several Graph Attention Layers (GATs) followed by fully connected layers, and its architecture is defined by the MolGAT class.




The __init__ method of the MolGAT class initializes the layers of the model. It takes as input the following parameters:

- `node_features`: the dimensionality of the input node features.
- `hidden_dim`: the dimensionality of the hidden representations.
- `edge_features`: the dimensionality of the input edge features.
- `num_heads`: the number of attention heads used in each GAT layer.
- `dropout`: the dropout probability used in the GAT layers.
- `num_conv_layers`: the number of GAT layers.
- `num_fc_layers`: the number of fully connected layers.

The forward method of the MolGAT class performs a forward pass through the layers of the model. It takes as input the following arguments:

- `x`: the node feature matrix of shape (num_nodes, node_features).
- `edge_index`: the edge index tensor of shape (2, num_edges).
- `batch_index`: the batch index tensor of shape (num_nodes,).
- `edge_attr`: the edge feature matrix of shape (num_edges, edge_features).

The forward method performs the following steps:

- Applies a GAT layer to the input node features and edge features.
- ReLU activation function is applied to the output of the GAT layer.
- Dropout is applied to the output of the GAT layer.
- Graph-level max pooling and mean pooling are applied to the output of the GAT layer.
- The output of the GAT layer is concatenated with the output of the graph-level pooling.
- Fully connected layers with ReLU activation functions are applied to the concatenated output.
- Dropout is applied to the output of the fully connected layers.
- The output of the fully connected layers is passed through a linear layer to obtain the final output.

Overall, the MolGAT model learns a hierarchical representation of the molecular graph using GAT layers and fully connected layers, and uses attention-based graph aggregation to summarize the node-level features and generate a graph-level representation, which is then used to make a prediction of the molecular property.


## Requirements

The following packages are required to run the MolGAT model:

- Python 3.x
- PyTorch
- RDKit
- PyTorch_geometric(PyG)
- numpy
- pandas
- scikit-learn



## Installation

To install MolGAT from the source code in GitHub, follow these steps:

Clone the repository:

```
git clone https://github.com/mesfind/molgnn.git
```

Navigate to the directory:

```
cd molgat

```

Install the required dependencies:


```
pip install -r requirements.txt
```

Install the package:

```
pip install -e .
```



## Model Architecture

The MolGAT model consists of multiple graph attention convolutional layers, followed by fully connected layers. The number of convolutional layers and fully connected layers can be specified as input parameters. Each convolutional layer is followed by a batch normalization layer and ReLU activation. The output of the last convolutional layer is concatenated with the global maximum pooling and global average pooling of node features, and passed through the fully connected layers to obtain the final prediction.

## Training

The model is trained using mean squared error (MSE) loss, and optimized using the Adam optimizer with a custom learning rate schedule:

```
python train.py

```

Please cite our [paper](https://pubs.acs.org/doi/full/10.1021/acsomega.3c01295) (and the respective papers of the methods used) if you use this code in your own work:

```
@article{doi:10.1021/acsomega.3c01295,
author = {Chaka, Mesfin Diro and Geffe, Chernet Amente and Rodriguez, Alex and Seriani, Nicola and Wu, Qin and Mekonnen, Yedilfana Setarge},
title = {High-Throughput Screening of Promising Redox-Active Molecules with MolGAT},
journal = {ACS Omega},
volume = {8},
number = {27},
pages = {24268-24278},
year = {2023},
doi = {10.1021/acsomega.3c01295},
}
```
