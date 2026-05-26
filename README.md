# ⚙️ Requirements

This environment configuration is shared across the entire project. To ensure full reproducibility, please install the following primary dependencies. The code has been tested under **Python 3.10** and **PyTorch 2.6.0**.

> torch==2.6.0
> torch-geometric==2.6.1
> torchaudio==2.6.0
> torchvision==0.21.0
> ogb==1.3.6
> numpy==1.26.4
> pandas==2.2.3
> scipy==1.15.3
> matplotlib==3.10.0
> scikit-learn==1.5.2

*(Note: Ensure your PyTorch installation matches your specific CUDA version.)*



# Part 1: Transfer Error (Cora, PubMed, OGBN-Arxiv)

This directory contains the core implementation for empirically validating the size transferability of Graph Convolutional Networks (GCNs) across varying sparsity schemes. It includes the complete pipeline for training, testing, and visualizing both the transfer errors and the empirical edge densities on three real-world datasets: Cora, PubMed, and OGBN-Arxiv.

Our experimental design aligns with the mathematical construction of the generalized graphon, simulating the expansion from a "dense core" to a "sparse tail" via degree-ordered sampling.


## 📂 File Structure

* `Stretched_models.py`: Defines the custom GCN architectures and the sparse aggregation layer (`SpecificAggrLayer`) necessary for unnormalized transferability analysis.
* `Stretched_sampling.py`: Implements the degree-ordered subgraph sampling algorithms and density computation modules.
* `Stretched_GCN_train.py`: Trains the base GCN model on a fixed-size subgraph and saves the pretrained weights.
* `Stretched_GCN_test.py`: Evaluates the transfer error (L2 discrepancy) between the pretrained subgraph and continuously expanding target graphs under three distinct sparsity schemes (Scheme I, II, III).
* `Plot_testResults.py`: Reads the evaluation results and generates publication-ready convergence plots.
* `Plot_Density.py`: Empirically samples subgraph edge densities across different schemes and fits mathematical curves ($\Theta(1/n)$, $\Theta(\log n/n)$, $\Theta(1)$) to validate the underlying sparsity settings.

## 🚀 How to Run

The pipeline is fully parameterized. You can seamlessly switch between datasets and model architectures using command-line arguments. 

### 1. Transfer Error Evaluation Pipeline
To reproduce the baseline transfer error results (e.g., on the Cora dataset with 2 layers and 32 hidden channels), run sequentially:

`python Stretched_GCN_train.py --dataset Cora --num_layers 2 --hidden_channels 32`

`python Stretched_GCN_test.py --dataset Cora --num_layers 2 --hidden_channels 32`

`python Plot_testResults.py --dataset Cora --num_layers 2 --hidden_channels 32`

*Outputs: A saved model weight `.pt`, a results `.csv` file, and a standard SCI-styled `.pdf` figure.*

### 2. Edge Density Sampling & Curve Fitting
To validate the sparsity schemes and generate density convergence curves, run the density plotting script. You can switch between a `standard` layout or an `icml` single-column optimized layout.

`python Plot_Density.py --dataset Cora --style standard`

`python Plot_Density.py --dataset ogbn-arxiv --style icml`

## 📊 Sparsity Schemes Explained

During both the testing and density sampling phases, the model evaluates structural scaling under three schemes:
* **Scheme I**: $f(n) \propto 1/n$ (Extremely sparse, density vanishes rapidly)
* **Scheme II**: $f(n) \propto \log(n)/n$ (Moderately sparse)
* **Scheme III**: $f(n) = \Theta(1)$ (Expected edge density remains constant)

The generated plots will demonstrate that across all schemes, the transferability error consistently converges to zero as the graph size increases and expected edge density decreases, validating the theoretical upper bounds.



# 📈 Part 2: Synthetic Experiment

This section validates the continuous limit and convergence properties of standard c-GCNs versus the proposed Generalized Graphon Convolutional Networks (GWCN). The experiments are conducted on synthetic geometric random graphs dynamically generated via a Gaussian kernel.

## 📂 File Structure (Part 2)

* `models.py`: Defines the standard c-GCN architecture with symmetric Laplacian normalization (`MyGCN`).
* `graph_utils.py`: Efficient utilities for generating nodes on a 3D surface and sampling edges via a Gaussian distance kernel.
* `run_convergence.py`: Evaluates the convergence error of standard c-GCNs. It computes the continuous limit on a dense graph ($\alpha=1$) and measures discrepancies as graphs grow and become sparser.
* `run_stretched_convergence.py`: Evaluates the proposed Stretched GWCN. It computes specific continuous limits for each sparsity scaling independently before measuring convergence errors.
* `plot_convergence.py`: Reads the combined experimental results from both models and generates publication-ready plots.

## 🚀 How to Run

To reproduce the synthetic graph experiments, run the following scripts sequentially:

**Step 1: Run standard c-GCN experiments**
> python run_convergence.py

*Output: `convergence_results_combined.csv`*

**Step 2: Run Stretched GWCN experiments**
> python run_stretched_convergence.py

*Output: `stretched_convergence_results_combined.csv`*

**Step 3: Generate Convergence Plots**
> python plot_convergence.py

*Output: 4 SCI-styled PDF figures (`fig_cGCN_Error.pdf`, `fig_cGCN_Density.pdf`, `fig_GWCN_Error.pdf`, `fig_GWCN_Density.pdf`).*

## 🧠 Experimental Design

The experiments dynamically generate synthetic graphs sized $n \in [100, 1000]$ and compare the finite graph model outputs against a large continuous limit approximation ($N=3000$). We evaluate the models under four distinct sparsity scaling schemes:
1. $\alpha(n) = n^{-1/4}$
2. $\alpha(n) = n^{-1/2}$
3. $\alpha(n) = \log(n)/n$
4. $\alpha(n) = n^{-1}$

The generated plots will illustrate that while standard c-GCNs may struggle or exhibit scaling biases under extreme sparsity, the proposed unnormalized GWCN correctly converges to its generalized continuous limits across all tested sparse regimes.


# 🎯 Part 3: Supplementary Performance Metrics (Cora Dataset)

This section extends the transferability analysis by evaluating the real-world downstream utility of the pretrained models. Instead of solely relying on the L2 feature discrepancy, this module rigorously assesses Node Classification Accuracy, Macro-F1 Score, and Cross-Entropy Loss across varying subgraph sizes and sparsity schemes.

## 📂 File Structure (Part 3)

* `Test_acc_f1.py`: The comprehensive evaluation script. It samples subgraphs, runs the forward pass, and computes multiple downstream metrics (Transfer Error, Loss, Accuracy, Macro-F1) across multiple random trials. It rigorously records both the mean and standard deviation for statistical significance.
* `Plot_acc_f1.py`: The visualization script. It parses the comprehensive metrics CSV and automatically generates four independent, publication-ready error-bar plots (one for each metric).

## 🚀 How to Run

Ensure you have already trained the base model using the scripts in Part 1 (e.g., `Stretched_GCN_train.py`). Then, run the following scripts sequentially to evaluate downstream metrics (e.g., for Cora, 3-layer, 64-hidden):

**Step 1: Evaluate all metrics**
> python Test_acc_f1.py --dataset Cora --num_layers 3 --hidden_channels 64

*Output: A combined metrics dataset `Cora_Test_3_64_metrics.csv` containing means and standard deviations.*

**Step 2: Generate Metric Plots**
> python Plot_acc_f1.py --dataset Cora --num_layers 3 --hidden_channels 64

*Output: 4 SCI-styled PDF figures (`Cora_plot_Accuracy_3_64.pdf`, `Cora_plot_F1Score_3_64.pdf`, `Cora_plot_Loss_3_64.pdf`, `Cora_plot_TransferError_3_64.pdf`).*

## 🧠 Experimental Design

While the unnormalized GWCN theoretically guarantees the convergence of the L2 representation error, it is crucial to empirically verify that this mathematical property translates to stable downstream task performance. 
* **Robustness Evaluation:** The Macro-F1 Score is included alongside standard Accuracy to provide a robust evaluation metric, ensuring the model's performance remains stable even under potential class imbalances in the randomly sampled subgraphs.
* **Statistical Significance:** Every point on the generated curves is averaged over multiple random trials (default: 50), with error bars representing the standard deviation. This confirms that the scaling stability across extreme sparsity ($\Theta(1/n)$) is a persistent structural property of the architecture, rather than an artifact of lucky sampling.
