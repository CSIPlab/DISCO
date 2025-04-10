This repository aims to automate the discovery of algorithms through the use of various machine learning techniques. We start by exploring sparse coding and see how a Neural Architecture Search (NAS) based training framework can recreate algorithms such as the Fast Iterative Shrinkage Thresholding Algorithm (FISTA). 

## **NAS FISTA**

We rediscover FISTA by creating sufficiently large DARTS cells and have the model learn operations matching the FISTA algorithm's operations. The main choices to consider are the proximal operator (shrinkage), the gradient's choice for acceleration (momentum), and the number of layers in our model (iterations of the algorithm). 

## **Momentum and Preconditioning**
We also investigated two possible approaches for discovery of acceleration: (1) Add **momentum** terms in the unrolled network, which are equivalent to adding the skip connections in the network. (2) Add a **preconditioning** operator in the forward/adjoint steps.

## **Shift Varying Systems**
In this section, we explore efficient modeling approaches for shift-varying systems. We focus on imaging applications such as shift-varying blur and atmospheric turbulence. These systems are typically large and computationally expensive to represent directly. Our objective is to discover novel and efficient factorizations starting from minimal assumptions.

## **Citation**
```
@misc{disco,
  author = {Sarthak Gupta, Yaoteng Tan, Nebiyou Yismaw, Patrick Yubeaton, Salman Asif, Chinmay Hegde},
  month = april,
  title = {{DISCO: Discovery of Sparsity Constrained Optimization Algorithms}},
  year = {2025}
```
