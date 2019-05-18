# Pytorch-Tutorial
Learning how to use this shit in Google Colab


# Backpropagation 

## Output layer

$\frac{\partial J}{\partial W_{ji}} = \frac{\partial J}{\partial z^r_j} \frac{\partial z^r_j}{\partial W^r_{ij}}$

$\frac{\partial J}{\partial W_{ji}} = \delta^r_j a^{r-1}_i$

$\frac{\partial J}{\partial W_{ji}} = (\hat{y_j} - y_j) a^{r-1}_i$

## Hidden layers

$\delta^r_j = \frac{\partial J}{\partial z^r_j}$

$\delta^r_j = \sum_k \frac{\partial J}{\partial z^{r+1}_k} \frac{\partial z^{r+1}_j}{\partial z^r_j}$

$\delta^r_j = \sum_k \delta^{r+1}_k W^{r+1}_{ji} f'(z^r_j)$

$\delta^r_j = f'(z^r_j) \sum_k \frac{\partial J}{\partial z^{r+1}_k} W^{r+1}_{ji}$
