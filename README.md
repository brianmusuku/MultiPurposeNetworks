## Dual Purpose Weights

This repository contains code for multi purpose weights. A neural network optimised for multiple tasks depending on the weights transformation. In this experiment, the default configuration of the model is a classifier but when weights are transformed (pseudoinverse / transpose) it becomes a generator.


## Results

We train variants on the MNIST dataset for 100 epochs. We evaluate on digit classifier accuracy and mean absolute error of the data generator.

| Transformation    | Classifier    | Generation (MAE)|
| ------------------|:-------------:|:-------------:|
| transpose         | 91.2%         |  0.4057       |
| psuedoinverse     | 91.7%         |  0.3753       |
| shuffled rows     | ~             |  ~            |

## Usage 


For transpose transformation: 
```
python main.py --type transpose
```


For psuedoinverse transformation: 
```
python main.py --type pinv
```

## To Do

* Implement more weight transformations
* Implement deeper layered networks.
* Test more tasks and datasets.
