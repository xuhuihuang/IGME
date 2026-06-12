
# A Tensor-Based Approach for Identifying Slow Collective Variables in Biomolecular Dynamics with a Non-Markovian Framework

  

### Abstract

  

Understanding conformational dynamics is essential for elucidating the mechanisms of many biological processes. A major challenge in analyzing molecular dynamics (MD) simulations is identifying collective variables (CVs) that capture the slow dynamical modes governing long-timescale conformational changes. Existing approaches for CV discovery are largely based on Markovian dynamical models, which may be inadequate when the underlying dynamics exhibit memory effects. Here, we develop a framework that integrates nonlinear CV discovery with non-Markovian dynamical modeling. The method combines AMUSET-based time-lagged independent component analysis (AMUSET-TICA), which constructs nonlinear feature representations using tensor-based basis expansions, with the integrative generalized master equation (IGME) framework for modeling non-Markovian dynamics. To efficiently estimate the multiple Koopman matrices required for IGME fitting, we introduce a one-step multi-lag Koopman (OSMLK) fitting scheme. The resulting AMUSET-TICA-IGME framework identifies continuous nonlinear CVs directly from MD trajectories while incorporating memory effects through the IGME formalism. Applications to alanine dipeptide, the folding dynamics of the FIP35 WW domain, and the dimerization of 9-(diphenylmethylene)-9H-fluorene (9d9f) demonstrate that our framework accurately captures slow dynamical modes and provides stable kinetic models for systems with complex conformational dynamics. These results establish the AMUSET-TICA-IGME framework as a general strategy for integrating nonlinear feature representations with non-Markovian dynamical modeling, opening new opportunities for the analysis of long-timescale biomolecular dynamics.

  

### References

AMUSET-TICA-IGME:
[Jingcheng Dai, Siqin Cao, Zige Liu, Xuhui Huang, ChemRxiv, 2026](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15004602/v1)

AMUSET-TICA:

[Siqin Cao, Feliks Nüske, Bojun Liu, Micheline B. Soley, Xuhui Huang, J. Chem. Theory Comput. 2025, 21, 9, 4855–4866](https://doi.org/10.1021/acs.jctc.5c00076)

IGME: 

[Siqin Cao, Yunrui Qiu, Michael L. Kalin, Xuhui Huang, Integrative generalized master equation: A method to study long-timescale biomolecular dynamics via the integrals of memory kernels, J. Chem. Phys. 159, 134106 (2023)](https://doi.org/10.1063/5.0167287)

[Bojun Liu, Siqin Cao, Jordan G. Boysen, Mingyi Xue, Xuhui Huang, Memory kernel minimization-based neural networks for discovering slow collective variables of biomolecular dynamics, Nature Computational Science volume 5, pages 562–571 (2025)](https://doi.org/10.1038/s43588-025-00815-8)


### Illustration

  
<img src="https://github.com/xuhuihuang/IGME/blob/main/AMUSET-TICA-IGME/release/Figure1.png" height=300></img>

  

## Quick start

  For each system, the whole process starts with the tICs (always from tICA) as input. 

  Steps:
  1. Generation of Koopman matrices (done with library AMUSET-TICA: https://github.com/xuhuihuang/amusettica) at different short lag times;
  2. One-Step-forward Multi-Lag Koopman (OSMLK) Fitting is applied, based on IGME theory, to scan RMSE of different combinations of hyperparameters;
  3. CVs are generated from SVD of the tensor network, which are constructed with the hyperparameters determined in step 2;
 
 Here are the descriptions of all the files, with a brief workflow of each system at the beginning of every sections.
 

### Alanine Dipeptide:

Workflow: it starts with Ala2-dim25-KoopmanGeneration.ipynb for Koopman matrices generation, followed with RMSE scanning of hyperparameters in Ala2-RMSE_n_CV.ipynb, and plots are output from Ala2-AllPlots.ipynb. 
  
Ala2-dim25-KoopmanGeneration.ipynb:
Generates rank 25 Koopman matrices and saves them into Ks25.npy. Also saves comparisons with rank 125 and rank 343.

Ala2-RMSE_n_CV.ipynb:
Runs over many combinations of tau_k and L to identify the best choices. Uses them to build the CVs and also the plots. The ITS comparisons with previous methods like tICA and AMUSET-tICA are also in.

Ala2-tICsGaussian_n_bootstraping.ipynb:
Two parts. 1. Bootstrapping for CK test; 2. Plotting the tICs expansions. 

Ala2-AllPlots.ipynb:
Collects the data from all previous codes and generates the plots ready for publications. Has color-friendly plans, and designs neat plots.


### FIP35 WW domain:
Workflow: it starts with FIP35-Koopman.ipynb for Koopman matrices generation, followed with RMSE scanning of hyperparameters in FIP35-RMSE_n_CV.ipynb, and plots are output from FIP35-AllPlots.ipynb. The folding path with the analysis of the MD trajectories is done in FIP35-foldingPath.ipynb.

FIP35-Koopman.ipynb:
Uses tICs to generate Koopman matrices of rank 27 with AMUSET-TICA library and outputs the result into "FIP35-Ks.npy". Contains the loading and usage of file "FIP35-tICA.npz".

FIP35-savings-comparison.ipynb:
Two parts. 1. Comparing rank 27, 125, 343 for the minimal trajectory length needed for ITS reaching gold standard; 2. At rank 27, compare ITS results at different lag times for AMUSET-TICA and A.T.-IGME.

FIP35-ITS-comparison.ipynb:
Does tICA, AMUSET-TICA, A.T.-IGME at different lag times for demonstrating savings over different methods.

FIP35-RMSE_n_CV.ipynb:
Goes over hyperparameters tau_k and L for fitting. Creates CV plot against RMSD 1 & 2 for predicted CV with propagation parameters tau_k=42, L=25, at 300ns lag time; saved to "FIP35_pred300lag-mode*_data.txt".

FIP35-foldingPath.ipynb:
Creates the CVs for tau_k=30 and L=15 and outputs into "CVs_pred30-45.npy". Performs 3D plot for CV1, CV3 and CV4. Plots are reproduced in a nicer way in "FIP35-AllPlots.ipynb".

FIP35-tICsGaussian_n_bootstraping.ipynb:
Two parts. 1. Bootstrapping for CK test; 2. Plotting the tICs expansions.

FIP35-AllPlots.ipynb:
Collects the data from all previous codes and generates the plots ready for publications. Has color-friendly plans, and designs neat plots.


  

### 9d9f:
Workflow: it starts with 9d9f_tICA_AT_IGME-comparison.ipynb for Koopman matrices generation, followed with RMSE scanning of hyperparameters in 9d9f_RMSE.ipynb. CVs computed with different methods are generated and compared in 9d9f_cv_plots.ipynb. All the plots are finally summarized in Ala2-AllPlots.ipynb.

9d9f_tICA_AT_IGME-comparison.ipynb:
Uses sorted tICs in "9d9f_inv_2tics.npy", which has top 2 tICs for overall 40 trajectories with 50,000 frames in each (1 frame = 2 ps) and apply tICA, AMUSET-TICA, AMUSET-TICA-IGME methods for doing comparison. Koopman matrices for from lag_time = 1 to lag_time = 300 frames in AMUSET-TICA are saved in "9d9f_Ks.npy". ITS comparisons are saved in "Panel_1d_ITS.npz". 

9d9f_RMSE.ipynb:
Goes over hyperparameters tau_k and L for fitting and the results are saved to "Panel_1e*_RMSE.npz".

9d9f_cv_plots.ipynb:
Generates CV against theta plots from 5 methods. 1. GraphVAMPnets (using "graphvampnets_cvs_50epochs.npy"); 2. Unsorted tICA ("9d9f_pwd_2tics.npy"); 3. Sorted tICA ("9d9f_inv_2tics.npy"); 4. AMUSET-TICA (at lag_time = 86 frames); 5. AMUSET-TICA-IGME (tau_k=65, L=20, propagation steps=100 frames). Output all the CV results into "9d9f_[Method name]_mode[1/2]_data.txt". Also contains usage of theta trajectories (folder 9d9f_theta) and generation of SVD results ("9d9f-dim25-SVD-Vt.npy").

9d9f-tICsGaussian_n_bootstraping.ipynb:
Two parts. 1. Bootstrapping for CK test; 2. Plotting the tICs expansions.

9d9f-AllPlots.ipynb:
Collects the data from all previous codes and generates the plots ready for publications. Has color-friendly plans, and designs neat plots.


  
  
