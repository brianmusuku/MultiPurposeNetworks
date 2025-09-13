# Dual Purpose Weights

This repository provides code for dual pupose weights. A neural network whose weights are optimised for multiple tasks depending on the transformation. In the default configuration, it is a classifier and when weights are transformed (pseudoinverse / transpose) it becomes a generator.


## Results

| Transformation    | Classifier    | Generation (MSE)    |
| ------------------|:-------------:|:-------------:|
| Transpose         | 92%           |  0.6          |
| psuedoinverse     | 89%           |  0.9          |
| shuffled rows     | 13%           |  ~            |

## Usage 

```
python main.py
```

## To Do

* Implement more weight transformations
* Implement deeper layered networks.
* Test more tasks.
