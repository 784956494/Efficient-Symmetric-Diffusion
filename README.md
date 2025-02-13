# Implementation for the paper "Efficient DIffusion Models for Symmetric Manifolds"

## Installation
Run the following commands to create the virtual environemnt for running this code:
```
python3.10 -m venv symm_diff
source symm_diff/bin/activate
pip installl -r requirements.txt
```

## Training: example usage
See the *examples* folder for an example of how to generate data, train the model, and evaluating the generated samples. Pseudo command examples are shown below. For the specifics of each argument please see the relevant files, namely *run.py* and *generate_data.py*.

### Generating data
```
python generate_data.py --dim $DIM --num_samples $NUM_SAMPLES --seed $SEED \
    --data_path $DATA_PATH --manifold $MANIFOLD --distribution $DISTRIBUTION
```

### Model Traing
```
python run.py --dim $DIM --epoch $EPOCH --device $DEVICE \
    --hidden_dim $HIDDEN_DIM --num_layers $NUM_LAYERS --lr 5e-4 --act $ACT \
    --batch_size $BATCH_SIZE --seed $SEED --beta_s 0.1\
    --beta_f 0.3 --mode train --model_directory $MODEL_DIRECTORY \
    --data_path $DATA_PATH --manifold $MANIFOLD --model_type $MODEL_TYPE\
    --data_type torch.float32 --ckpt_directory $CKPT_DIRECTORY\
    --figure_path $FIGURE_PATH
```

### Evaluation
```
python run.py --dim $DIM --epoch $EPOCH --device $DEVICE \
    --hidden_dim $HIDDEN_DIM --num_layers $NUM_LAYERS --lr 5e-4 --act $ACT \
    --batch_size $BATCH_SIZE --seed $SEED --beta_s 0.1\
    --beta_f 0.3 --mode eval --model_directory $MODEL_DIRECTORY \
    --data_path $DATA_PATH --manifold $MANIFOLD --model_type $MODEL_TYPE\
    --data_type torch.float32 --ckpt_directory $CKPT_DIRECTORY\
    --figure_path $FIGURE_PATH --eval_type EVAL_TYPE
```

Notably, the evaluation metrics allowed are loglikelihood for the torus and C2ST for SO(n) and U(n). Additionally, note that for the unitary group set the data type to *torch.complex64*. The trained model can either be a simple MLP or a ResNet model for better learning. In the training and evaluation scripts, replace the *$MANIFOLD* field with one of *torus, special_orthogonal, or unitary* and the *$DIM$ field with the appropriate dimensions (*d* for d-dimsional torus and *n* for *SO(n)* and *U(n)*) for reproducing the experiments. 
* For the result in Table 2, train with the script with *--manifold torus --model_type MLP*
    > For evaluation, set *--eval_type log_likelihood*
* For the result in Figure 1, run the script with *--manifold unitary --model ResNet --data_type torch.float32*
   > For the result in Table 3, record the per-iteration runtime with the above entries and *--batch_size 32*
   > For evaluation, set *--eval_type C2ST*
* For the result in Table 4, run the script with *--manifold special_orthgonal --model MLP*

### File Structure
The *manifold* folder contains files that implement manifold operations, including the loss function on each manifold. The *model folder contains files for implementing the neural networks and a *sampler* that implements our sampling algorithm. The *utils* folder contains several files for utility functions for visualization, evaluation metric, and mathematical functions.

### Baselines
For completeness, we also include the files for which the baseline models were trained at. The files for TDM can be found *TDM_files*, where the training scheme is handled by *main.py, runner.py*. Set *--problem-name* to SOn or Un. For RSGM, a relevant functions in PyTorch is in the *rsgm_files* folder. The code were derived from their official implementations, which can be found at:
* For RSGM: https://github.com/oxcsml/riemannian-score-sde
* For TDM: see the supplementary material at https://openreview.net/forum?id=DTatjJTDl1

## Results
Create a folder of your choosing to place the results of the training and evaluation. Specifying each save path (e.g. model directory) would automatically enable saving the relevant result.

## Synthetic Data Generation
For the torus and special orthogonal group, a wrapped Gaussian distribution is generated:
*  Given a $d$-dimensional Riemmanian manifold $\mathcal{M}$, a number of mixture components $k \in \mathbb{N}$, points $m_1,\cdots, m_k \in \mathcal{M}$ and covariance matrices $C_1,\cdots, C_k \in \mathbb{R}^{d \times d}$, we say that a random variable is distributed according to a wrapped Gaussian distribution with means  $m_1,\cdots, m_k$ and covariances $C_1,\cdots, C_k$ (with equal weights on each component) if its distribution is equal to that of a random variable $X$ sampled as follows:
    > Sample an index $i$ at random from $\{1,\cdots, k\}$.
    > Sample $Z \sim N(0, C_i)$
    > Set $X = \mathrm{exp}_{m_i}(Z)$, where $\mathrm{exp}_{x}(\cdot)$ denotes the exponential map at any point $x \in \mathcal{M}$.

For the unitary, the generated dataset models the Hamiltonian of a quantum oscillator
* $\mathrm{U}(n)$ matrix are of the form $e^{itH}$, where $t = 1$ is time, $H = \Delta_h - V_h$ is the discretized Hamiltonian for quantum oscillator. $V_h$ comes from the random potential function $V(x) = \frac{1}{2} \omega^2 \| x - x_0 \|^2$. $\omega \sim \mathcal{U}(\text{angular min}, \text{angular max})$, $x_0 \sim \mathcal{N}(\textbf{mean}, \textbf{var}^2)$