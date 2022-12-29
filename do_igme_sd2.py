#!python
"""
The python script to build the Integrative Generalized Master Equation (IGME). 
  (c) Dec 17, 2021 Yunrui QIU

Usage:
  input=TPM_file_name [begin=fit_range_begin end=fit_range_end epoch=number_of_epochs] do_igme_sd2.py 
or:
  export input=TPM_file_name
  export begin=15   # optional, tau_K, default: 1 (first frame of input file)
  export end=30     # optional, tau_G, default: number of input TPMs
  export epoch=1000 # optional, epochs, default 1000
  python do_igme_sd2.py

Output:
  tau_K tau_G  T=  T_hat_matrix_in_csv_format  A=  A_matrix_in_csv_format

This scipt is based on the following theories:
  (1) the 3rd order expansion of the solution of IGME
        e^(ln(A)+ln(B)) = A B
  (2) enforce the row/column sums of TPMs are always 1
  (3) force detailed balance of predicted TPMs

Package dependence:
  numpy
  pyTorch
  scipy
"""

import numpy as np
import gc
import os
import sys
import torch
import math
import scipy.linalg

class IntegralGME(object):

    def __init__(self, input_len=1, dimension=1, delta_time=1.0):
        self.delta_time = delta_time
        self.input_len = input_len
        self.dimension = dimension
        self.TPM = np.zeros((input_len, dimension, dimension))
        self.lag_time = np.zeros(input_len)
        self.tau_ini = 0
        self.tau_end = 10
        self.A_matrix = np.zeros((dimension, dimension))
        self.T_hat_matrix = np.zeros((dimension, dimension))
        self.__raw_data = np.zeros((input_len, dimension, dimension))
        self.__row_norm = False
        self.__col_norm = False
        self.__get_data = False
        self.__pre_set_data = False
        self.__initial_seeds = False

    def get_data(self, input_data):

        self.__raw_data = input_data
        if not isinstance(self.__raw_data, np.ndarray):
            raise TypeError("Loading input data is not np.array type")
        elif not self.__raw_data.ndim == 2:
            raise IOError("Dimension of input TPM data is not 2")
        else:
            self.__get_data = True

    def pre_set_data(self):

        if self.input_len != len(self.__raw_data):
            raise IOError("Input length is inconsistent with real data length")
        elif not self.dimension == np.sqrt(len(self.__raw_data[0])):
            raise IOError("Input dimension is inconsistent with real data dimension")
        else:
            self.TPM = np.reshape(self.__raw_data, (self.input_len, self.dimension, self.dimension))
            for i in range(self.input_len):
                self.lag_time[i] = self.delta_time * (i + 1)

        #if abs((np.sum(self.TPM[3, 0])) - 1) < 1e-3:
        #    self.__row_norm = True
        #    print("# The Transition Probability Matrix is row normalized and row normalization algorithm is used !")
        #elif abs(np.sum(self.TPM[3, :, 0]) - 1) < 1e-3:
        #    self.__col_norm = True
        #    print("# The Transition Probability Matrix is column normalized and column normalization algorithm is used !")
        #    for i in range(len(self.TPM)):
        #        self.TPM[i] = self.TPM[i].T
        #else:
        #    raise IOError("# Transition Probability Matrix is not normalized, cannot do qMSM")

        self.__pre_set_data = True
        del self.__raw_data
        gc.collect()

    def linear_regression_fitting(self, tau_ini, tau_end, output_file=False):

        self.tau_ini = tau_ini
        self.tau_end = tau_end
        if self.tau_ini > tau_end:
            raise IOError("tau_end parameter should be longer than tau_int parameter")
        if self.tau_end > len(self.TPM):
            raise IOError("tau_end parameter should be shorter than length of input TPM")
        if not self.__get_data:
            raise ValueError('Please use get_data method to get appropriate TPM data')
        if not self.__pre_set_data:
            raise NotImplementedError('Please use pre_set_data method to reset TPM')
        TPM_eigenval = np.zeros((self.tau_end - self.tau_ini, self.dimension))
        TPM_rightvec = np.zeros((self.tau_end - self.tau_ini, self.dimension, self.dimension))
        Q_matrix = np.zeros((self.tau_end - self.tau_ini, self.dimension, self.dimension))
        log_TPM = np.zeros((self.tau_end - self.tau_ini, self.dimension, self.dimension))
        A_matrix = np.zeros((self.dimension, self.dimension))
        T_hat_matrix = np.zeros((self.dimension, self.dimension))
        X_data = np.zeros((self.tau_end - self.tau_ini, 2))
        for i in range(0, self.tau_end - self.tau_ini):
            TPM_eigenval[i], TPM_rightvec[i] = np.linalg.eig(self.TPM[i + tau_ini])
            sorted_indices = np.argsort(TPM_eigenval[i])
            TPM_rightvec[i] = TPM_rightvec[i][:, sorted_indices[: -self.dimension - 1: -1]]
            TPM_eigenval[i] = TPM_eigenval[i][sorted_indices[: -self.dimension - 1: -1]]
            Q_matrix[i] = np.linalg.inv(TPM_rightvec[i])
            log_TPM[i] = np.dot(np.dot(TPM_rightvec[i], np.diag(np.log(TPM_eigenval[i]))), Q_matrix[i])
        X_data[:, 0] = np.ones(self.tau_end - self.tau_ini)
        X_data[:, 1] = self.lag_time[self.tau_ini:self.tau_end] / self.delta_time
        tmp = np.dot(np.linalg.inv(np.dot(X_data.T, X_data)), X_data.T)
        for j in range(self.dimension ** 2):
            Y_data = log_TPM[:, j // self.dimension, j % self.dimension]
            weight = np.dot(tmp, Y_data)
            A_matrix[j // self.dimension, j % self.dimension] = weight[0]
            T_hat_matrix[j // self.dimension, j % self.dimension] = weight[1]
        A_eigenval, A_rightvec = np.linalg.eig(A_matrix)
        T_hat_eigenval, T_hat_rightvec = np.linalg.eig(T_hat_matrix)
        A_eigenval = np.exp(A_eigenval)
        A_matrix = np.dot(np.dot(A_rightvec, np.diag(A_eigenval)), np.linalg.inv(A_rightvec))
        T_hat_eigenval = np.exp(T_hat_eigenval)
        T_hat_matrix = np.dot(np.dot(T_hat_rightvec, np.diag(T_hat_eigenval)), np.linalg.inv(T_hat_rightvec))

        if output_file:
            with open('LinearRegression_A_matrix.txt', 'ab') as file1:
                if not os.path.getsize('LinearRegression_A_matrix.txt'):
                    np.savetxt(file1, A_matrix, delimiter=' ')
                else:
                    raise IOError('Output A_matrix already exists, please create another!!')
            with open('LinearRegression_T_hat_matrix.txt', 'ab') as file2:
                if not os.path.getsize('LinearRegression_T_hat_matrix.txt'):
                    np.savetxt(file2, T_hat_matrix, delimiter=' ')
                else:
                    raise IOError('Output File already exists, please create another!!')

        del TPM_eigenval, TPM_rightvec, Q_matrix, log_TPM, X_data, Y_data, \
            A_eigenval, A_rightvec, T_hat_eigenval, T_hat_rightvec
        gc.collect()
        return A_matrix, T_hat_matrix

    def geometric_fitting(self, tau_ini, tau_end, output_file=False):
        self.tau_ini = tau_ini
        self.tau_end = tau_end
        if self.tau_ini > tau_end:
            raise IOError("tau_end parameter should be longer than tau_int parameter")
        if self.tau_end > len(self.TPM):
            raise IOError("tau_end parameter should be shorter than length of input TPM")
        if not self.__get_data:
            raise ValueError('Please use get_data method to get appropriate TPM data')
        if not self.__pre_set_data:
            raise NotImplementedError('Please use pre_set_data method to reset TPM')

        summation = np.zeros((self.dimension, self.dimension))
        for i in range(self.tau_ini, self.tau_end):
            summation += self.TPM[i]
        weight_T_mat = self.TPM[tau_ini] - self.TPM[tau_end]
        summation_inv = np.linalg.inv(summation)
        T_hat_matrix = np.identity(self.dimension) - np.dot(summation_inv, weight_T_mat)
        A_matrix = (self.tau_end * self.TPM[tau_ini] - self.tau_ini * self.TPM[tau_end]) / (self.tau_end - self.tau_ini)
        if output_file:
            with open('GeometricFitting_A_matrix.txt', 'ab') as file1:
                if not os.path.getsize('GeometricFitting_A_matrix.txt'):
                    np.savetxt(file1, A_matrix, delimiter=' ')
                else:
                    raise IOError('Output File already exists, please create another!!')
            with open('GeometricFitting_T_hat_matrix.txt', 'ab') as file2:
                if not os.path.getsize('GeometricFitting_T_hat_matrix.txt'):
                    np.savetxt(file2, T_hat_matrix, delimiter=' ')
                else:
                    raise IOError('Output File already exists, please create another!!')

        del summation, summation_inv, weight_T_mat
        gc.collect()
        return A_matrix, T_hat_matrix

    def initial_for_gradient_descent(self, tau_ini, tau_end, initial_seeds='linear', A_matrix=None, T_hat_matrix=None):

        self.tau_ini = tau_ini
        self.tau_end = tau_end
        if self.tau_ini > tau_end:
            raise IOError("tau_end parameter should be longer than tau_int parameter")
        if self.tau_end > len(self.TPM):
            raise IOError("tau_end parameter should be shorter than length of input TPM")
        if initial_seeds == 'customization' and A_matrix is None and T_hat_matrix is None:
            raise IOError("If the customization method is chosen for initialization, A_matrix and T_hat_matrix must "
                          "be given.")
        if not self.__get_data:
            raise ValueError('Please use get_data method to get appropriate TPM data')
        if not self.__pre_set_data:
            raise NotImplementedError('Please use pre_set_data method to reset TPM')
        if initial_seeds == 'linear':
            self.A_matrix, self.T_hat_matrix = self.linear_regression_fitting(tau_ini=self.tau_ini,
                                                                              tau_end=self.tau_end)
            self.__initial_seeds = True
        elif initial_seeds == 'geometric':
            self.A_matrix, self.T_hat_matrix = self.geometric_fitting(tau_ini=self.tau_ini, tau_end=self.tau_end)
            self.__initial_seeds = True
        elif initial_seeds == 'customization':
            self.A_matrix = A_matrix
            self.T_hat_matrix = T_hat_matrix
            self.__initial_seeds = True
        else:
            raise IOError('Method used to generate initial seeds is not valid')

    @staticmethod
    def frobenius_loss(prop_tpm, train_data, row_rate, stat_pop, detail_rate):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        diff = prop_tpm - train_data
        row_sum = torch.sum(prop_tpm, dim=1) - torch.ones(len(prop_tpm)).to(device)
        prop_tcm = torch.mm(stat_pop, prop_tpm)
        trans_sum = prop_tcm - prop_tcm.T
        fro_norm = torch.mul(diff, diff).sum() + row_rate * torch.mul(row_sum, row_sum).sum() + detail_rate * torch.mul(
            trans_sum, trans_sum).sum()
        return fro_norm

    def gradient_descent_opt(self, learning_rate=1e-6, row_rate=0.0, detail_rate=0.0, momentum=0.1, epochs=500,
                             output_file=False):
        if not self.__initial_seeds:
            raise NotImplementedError(
                "Should use initial_for_gradient_descent method to give an initial seeds for GD algorithm")
        if epochs < 0 or learning_rate < 0 or row_rate < 0 or detail_rate < 0 or momentum < 0:
            raise IOError("Input parameters should be larger than zero")
        if momentum > 1:
            raise IOError("Value of momentum should be in range [0, 1)")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        T_hat_matrix = self.T_hat_matrix.astype(np.float32)
        A_matrix = self.A_matrix.astype(np.float32)
        eigenval, eigenvec = scipy.linalg.eig(self.TPM[self.tau_end], right=False, left=True)
        eigenval = eigenval.real
        eigenvec = eigenvec.real
        tolerance = 1e-10
        mask = abs(max(eigenval) - eigenval) < tolerance
        stat_pop = eigenvec[:, mask].T
        stat_pop /= np.sum(stat_pop)
        station = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            station[i, i] = stat_pop[0, i]
        station = torch.from_numpy(station).to(device)
        station = station.to(torch.float32)
        TPM = torch.from_numpy(self.TPM).to(device)
        T_hat_matrix = torch.from_numpy(T_hat_matrix).to(device)
        T_hat_matrix.requires_grad = True
        A_matrix = torch.from_numpy(A_matrix).to(device)
        A_matrix.requires_grad = True

        optimizer = torch.optim.SGD([A_matrix, T_hat_matrix], lr=learning_rate, momentum=momentum)
        for epoch in range(epochs):
            loss = 0
            T_hat_power_n = T_hat_matrix
            for i in range(1, self.tau_end):
            #if self.tau_ini > 1:
            #    T_hat_power_n = torch.matrix_power(T_hat_matrix, self.tau_ini)
            #else :
            #    T_hat_power_n = T_hat_matrix
            #for i in range(self.tau_ini, self.tau_end):
                T_hat_power_n = torch.mm(T_hat_power_n, T_hat_matrix)
                prediction = torch.mm(A_matrix, T_hat_power_n)
                if i >= self.tau_ini:
                    loss += self.frobenius_loss(prediction, TPM[i], row_rate=row_rate, stat_pop=station,
                                                      detail_rate=detail_rate)
            loss /= (self.tau_end - self.tau_ini)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if epoch % 100 == 0:
            #    print("epoch= {}".format(epoch), "loss= {}".format(loss))

        if output_file:
            with open('GradientDescent_A_matrix.txt', 'ab') as file1:
                if not os.path.getsize('GradientDescent_A_matrix.txt'):
                    np.savetxt(file1, A_matrix.cpu().detach().numpy(), delimiter=' ')
                else:
                    raise IOError('Output File already exists, please create another!!')
            with open('GradientDescent_T_hat_matrix.txt', 'ab') as file2:
                if not os.path.getsize('GradientDescent_T_hat_matrix.txt'):
                    np.savetxt(file2, T_hat_matrix.cpu().detach().numpy(), delimiter=' ')
                else:
                    raise IOError('Output File already exists, please create another!!')
        return A_matrix.cpu().detach().numpy(), T_hat_matrix.cpu().detach().numpy()

    @staticmethod
    def calculate_its(tpm, lag_time, num_modes):
        eign_val, eign_vec = np.linalg.eig(tpm)
        idx = np.argsort(np.real(eign_val))
        ITS = np.zeros(num_modes)
        for i in range(num_modes):
            ITS[i] = -lag_time / np.log(np.real(eign_val[idx[-2 - i]]))
        return ITS

    @staticmethod
    def move_row_col(matrix, i):
        tmp = np.delete(matrix, i, axis=0)
        move_matrix = np.delete(tmp, i, axis=1)
        return move_matrix

    def mean_first_passage_time(self, T_hat_matrix=None, output_file=False):
        if T_hat_matrix is None:
            raise IOError("T_hat_matrix must be given to compute the mean first passage time")
        mfpt_a = np.zeros((self.dimension * (self.dimension - 1), self.dimension * (self.dimension - 1)))
        for j in range(self.dimension):
            index_ini = j * (self.dimension - 1)
            index_end = index_ini + self.dimension - 1
            mfpt_a[index_ini:index_end, index_ini:index_end] = self.move_row_col(T_hat_matrix, j)
        mfpt_a -= np.identity(self.dimension * (self.dimension - 1))
        mfpt_b = np.ones(self.dimension * (self.dimension - 1)) * -self.delta_time
        mfpt = np.linalg.solve(mfpt_a, mfpt_b)
        for i in range(self.dimension):
            mfpt = np.insert(mfpt, i * self.dimension + i, 0)
        mfpt = np.reshape(mfpt, (self.dimension, self.dimension)).T
        print("mean first passage time computed from T_hat_matrix is: \n", mfpt)
        if output_file:
            with open("MFPT_from_T_hat_matrix.txt", 'ab') as file1:
                if not os.path.getsize("MFPT_from_T_hat_matrix.txt"):
                    np.savetxt(file1, mfpt)
                else:
                    raise IOError('Output File already exists, please create another!!')
        del mfpt_a, mfpt_b
        gc.collect()
        return mfpt

    def squared_difference(self, A_matrix, T_hat_matrix, end_frame=100, station_point=10, tpm_ref=None):
        eigenval, eigenvec = scipy.linalg.eig(self.TPM[station_point], right=False, left=True)
        eigenval = eigenval.real
        eigenvec = eigenvec.real
        tolerance = 1e-10
        mask = abs(max(eigenval) - eigenval) < tolerance
        station_pop = eigenvec[:, mask].T
        station_pop /= np.sum(station_pop)
        # print("%.20f" % station_pop[0, 0])
        # print("%.20f" % station_pop[0, 1])
        # print("%.20f" % station_pop[0, 2])
        # print("%.20f" % station_pop[0, 3])
        station_pop = np.diag(np.reshape(station_pop, self.dimension))
        TPM_prop_igme = np.zeros((end_frame, self.dimension, self.dimension))
        TPM_prop_igme[0] = np.dot(A_matrix, T_hat_matrix)
        error = np.sum(np.power(np.dot(station_pop, (tpm_ref[0] - TPM_prop_igme[0])), 2))

        for i in range(1, end_frame):
            TPM_prop_igme[i] = np.dot(TPM_prop_igme[i - 1], T_hat_matrix)
            error += np.sum(np.power(np.dot(station_pop, (tpm_ref[i] - TPM_prop_igme[i])), 2))
        error = 100 * np.sqrt(error / end_frame / self.dimension ** 2)
        return error

    def root_mean_squared_error(self, ini_frame=10, end_frame=100, fitting_length=30, end_rmse=200, slice_point=1,
                                initial_seeds='linear', learning_rate=1e-6, row_rate=0.0, detail_rate=0.0, momentum=0.0,
                                epochs=500, tpm_ref=None, station_point=10, figure=False, output_file=False):
        if end_frame <= 1 or ini_frame < 1:
            raise IOError("Initial frame number end frame number should be larger than one!")
        if end_frame + fitting_length > self.input_len:
            raise ValueError("Required fitting period exceed the original input data!")
        if tpm_ref is None:
            raise IOError("tpm_ref should be given a TPM Reference!")
        error = np.zeros((math.ceil((end_frame - ini_frame) / slice_point), 3))
        for i in range(ini_frame, end_frame, slice_point):
            self.initial_for_gradient_descent(tau_ini=i, tau_end=i + fitting_length,
                                              initial_seeds=initial_seeds)
            print("********************************************************************")
            print("optimization for the " + str(i) + "____" + str(i + fitting_length) + " is going : ")
            A_matrix, T_hat_matrix = self.gradient_descent_opt(learning_rate=learning_rate, row_rate=row_rate,
                                                               detail_rate=detail_rate, momentum=momentum,
                                                               epochs=epochs)

            error[int((i - ini_frame) / slice_point), 0] = (i+1) * self.delta_time
            error[int((i - ini_frame) / slice_point), 1] = (i + fitting_length) * self.delta_time
            error[int((i - ini_frame) / slice_point), 2] = self.squared_difference(A_matrix, T_hat_matrix,
                                                                                   end_frame=end_rmse,
                                                                                   station_point=station_point,
                                                                                   tpm_ref=tpm_ref)
        #if figure:
        #    plt.figure()
        #    plt.plot(error[:, 0], error[:, 2], color='red', linewidth=2.5)
        #    plt.ylabel('RMSE(%)')
        #    plt.xlabel('Time')
        #    plt.show()
        if output_file:
            with open("igme_rmse_data.txt", 'ab') as file1:
                if not os.path.getsize("igme_rmse_data.txt"):
                    np.savetxt(file1, error)
                else:
                    raise IOError('Output File already exists, please create another!!')
        return error


#==============================================================================
#  Entry of the script  =======================================================
#==============================================================================


#-----------------------------------------------------------
#  Part 1. input parameters  -------------------------------
#-----------------------------------------------------------

env_fit_input = os.getenv('input');
env_fit_begin = os.getenv('begin');
env_fit_end = os.getenv('end');
env_epoch = os.getenv('epoch');

#-----------------------------------------------------------
#  Part 2. analysis parameters  ----------------------------
#-----------------------------------------------------------

if not env_fit_input:
    print("do_igme_sd2.py : perform IGME based on 3rd order solution of IGME")
    print("    e^(ln(A)+ln(B)) = A B")
    print("  usage: input=TPM_file_name [begin=1 end=100 epoch=1000] python do_igme_sd2.py")
    sys.exit()

if not os.path.exists(env_fit_input):
    print("do_igme_sd2.py : error : cannot open `"+env_fit_input+"'")
    sys.exit()

input_data = np.loadtxt(env_fit_input, dtype=float)
TPM = np.reshape(input_data, (len(input_data), int(np.sqrt(len(input_data[0]))), int(np.sqrt(len(input_data[0])))))
input_len = len(TPM)
dimension = len(TPM[0])

fit_begin=1
fit_end=input_len-1
fit_epoch=1000
if env_fit_begin:
    fit_begin = int(env_fit_begin)
if env_fit_end:
    fit_end = int(env_fit_end)
if env_epoch:
    fit_epoch = int(env_epoch)
if fit_epoch < 10:
    fit_epoch = 10

#-----------------------------------------------------------
#  Part 3. build IGME  -------------------------------------
#-----------------------------------------------------------

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
igme = IntegralGME(input_len=input_len, dimension=dimension, delta_time=1)
igme.get_data(input_data=input_data)
igme.pre_set_data()

igme.initial_for_gradient_descent(tau_ini=fit_begin, tau_end=fit_end, initial_seeds='linear')
A_matrix, T_hat_matrix = igme.gradient_descent_opt(learning_rate=1e-7, row_rate=1, detail_rate=0.1, momentum=0.8, epochs=fit_epoch+1, output_file=False)

#-----------------------------------------------------------
#  Part 4. output: only output tau_K, tau_G, T_hat and A  --
#-----------------------------------------------------------

print(fit_begin, fit_end, end=" ")

print("  T=", end=" ")
nele=0;
for i in T_hat_matrix:
  for j in i:
    nele += 1;
    if nele == dimension*dimension:
        print(format(j,".16g"), end=" ")
    else:
        print(format(j,".16g"), end=",")

print("  A=", end=" ")
nele=0;
for i in A_matrix:
  for j in i:
    nele += 1;
    if nele == dimension*dimension:
        print(format(j,".16g"), end=" ")
    else:
        print(format(j,".16g"), end=",")

print(" ")





