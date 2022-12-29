# IGME
Integrative Generalized Master Equation (IGME): A Theory to Study Long-timescale Biomolecular Dynamics via the Integrals of Memory
Kernels.

This code is used to generate Integrative-Generalized-Master-Equation (IGME) models for large molcular systems with the steepest
descent optimization based on the initial guess of least-square-fitting. Three theories have been provided:

do_igme_sd2.py : based on the second-order solution of IGME: T_IGME(t) = A T_hat^t

do_igme_sd3a.py : based on a three-order solution of IGME: T_IGME(t) = (A T_hat^t + T_hat^t A) / 2

do_igme_sd3a.py : based on another three-order solution of IGME: T_IGME(t) = sqrt{A} T_hat^t sqrt{A}

# Usage

## perform IGME with specified beginning and end points of fitting:

```console
input=TPM_file_name [begin=fit_range_begin end=fit_range_end epoch=number_of_epochs] do_igme_sd2.py
```

```console
export input=TPM_file_name 
export begin=fit_range_begin
export end=fit_range_end
export epoch=number_of_epochs
python do_igme_sd2.py
```

## scan parameters of IGME

```console
export TPM_file=...;
for ((i=2;i<=`cat $TPM_file|wc -l`;i++)); do
  for ((j=1;j<i=1;j++)); do
    input=$TPM_file begin=$j end=$i python do_igme_sd2.py
  done
done
```

# note

## format for input TPM_file

The TPM_file is a single file contains multiple TPMs. Each line in TPM_file represent a transition probability matrix.
Each line has N^2 elements (N is the dimension of TPMs or number of states) seperated by spaces or tabs.

## required packages

numpy

pyTorch

scipy

# Reference

[1] TBA

