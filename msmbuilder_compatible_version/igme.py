""" Integrative Generalized Master Equation (IGME)
Use the Least Square fitting algorithm to build IGME model(s) from given
time-dependent transition probability matrixes. The IGME can provide
numerically stable approaches to compute long-time dynamics and time integrated
memory kernels.
"""
# Author: Siqin Cao <siqincao@gmail.com>
# Contributors:
# Copyright (c) 2023, University of Wisconsin-Madison
# All rights reserved.

import numpy
import sys

class IGMENS :
    @staticmethod
    def matrix_log_accurate(m) :
        ev, vr = numpy.linalg.eig(m)
        return numpy.matmul(vr, numpy.matmul(numpy.diag(numpy.log(ev)), numpy.linalg.inv(vr)))

    @staticmethod
    def matrix_log_approx(m, order=10) :
        res = m - numpy.identity(len(m))
        ret = 0
        temp = res
        for i in range(1, order+1) :
            ret += temp / i;
            temp = numpy.matmul(temp, -res)
        return ret
        
    @staticmethod
    def matrix_log(m, logarithm_approx_order=0) :
        if (logarithm_approx_order>=2) & (logarithm_approx_order<=100) :
            return IGMENS.matrix_log_approx(m, logarithm_approx_order)
        else :
            return IGMENS.matrix_log_accurate(m)

    @staticmethod
    def matrixes_log(ms, logarithm_approx_order=0) :
        ret = []
        for i in range(len(ms)) :
            ret.append(IGMENS.matrix_log(ms[i], logarithm_approx_order))
        return ret

    @staticmethod
    def matrix_exp(m) :
        ev, vr = numpy.linalg.eig(m)
        return numpy.matmul(vr, numpy.matmul(numpy.diag(numpy.exp(ev)), numpy.linalg.inv(vr)))

    @staticmethod
    def matrixes_exp(ms) :
        ret = []
        for i in range(len(ms)) :
            ret.append(IGMENS.matrix_exp(ms[i]))
        return ret

    @staticmethod
    def lsf_xy(x, y) :
        length = len(x)
        if (length>len(y)) :
            length = len(y)
        sx = numpy.mean(x)
        sy = numpy.mean(y)
        sxx = numpy.dot(x, x) / length
        syy = numpy.dot(y, y) / length
        sxy = numpy.dot(x, y) / length
        if (syy == sy*sy) :
            return sy, 0, 1
        if (sxx == sx*sx) :
            return sy, 0, 0
        k = (sxy - sx*sy) / (sxx - sx*sx)
        b = sy - k * sx
        r = (sxy - sx*sy) / numpy.sqrt((sxx-sx*sx) * (syy - sy*sy))
        return b, k, r

    @staticmethod
    def matrixes_lsf(mat, dimension, begin, end) :
        A = numpy.zeros([dimension,dimension])
        B = numpy.zeros([dimension,dimension])
        R = numpy.zeros([dimension,dimension])
        for i in range(dimension) :
            for j in range(dimension) :
                x = []
                y = []
                for k in range(begin, end+1) :
                    x.append(k+1)
                    y.append(mat[k][i][j])
                A[i,j], B[i,j], R[i,j] = IGMENS.lsf_xy(x, y)
        return A, B, R

    @staticmethod
    def print_matrix_in_csv(m, end="\n") :
        for i in range(len(m)) :
            for j in range(len(m[i])) :
                if (i>0) | (j>0) :
                    print(",", end="")
                print(m[i][j], end="")
        print(end, end="")


class IGME(object) :

    """ Integrative Generalized Master Equation

    Parameters
    ----------
    logarithm_approx_order : int, default: 0
        When this parameter is set >=2 and <=100, the logarithm will be
        computed with Taylor expansions up to the order of this parameter.
        When this parameter is below 2 or above 100, the approximation will be
        disabled and the accurate logarithm will be used.
        This parameter is used to compute matrix logarithm when eigenvalues
        contain negative components. In such cases, logarithm of matrix will
        lead to a lot of complex elements.

    Attributes
    ----------
    lnTh : array like, shape = (n_state, n_state)
        The logorithm of \hat{T} matrix
    Th : array like, shape = (n_state, n_state)
        The \hat{T} matrix
    lnA : array like, shape = (n_state, n_state)
        The logorithm of A matrix
    R : array like, shape = (n_state, n_state)
        A matrix of Pearson Correlation Coefficients
    rmse : real
        The RMSE between input and IGME predictions
        rmsew : the RMSE is weighted by population
    timescales : array like, shape = (n_state - 1)
        The implied timescales of \hat{T} matrix. This is also the long-time
        limit of implied timescales of IGME
    M0 : array like, shape = (n_state, n_state)
        The time integral of memory kernels.
    mik : real
        The root square mean of all elements of M0

    Reference
    ----------

    """

    def __init__(self, logarithm_approx_order=0) :
        self.logarithm_approx_order = logarithm_approx_order
        self._begin = 0
        self._end   = 0
        self._lnTh  = 0
        self._Th    = 0
        self._lnA   = 0
        self._R     = 0
        self._M0    = 0
        self._mik   = 0
        self._timescales = 0
        self._rmse  = 0

    @property
    def lnTh(self):
        return self._lnTh

    @property
    def lnA(self):
        return self._lnA

    @property
    def Th(self):
        return self._Th

    @property
    def R(self):
        return self._R

    @property
    def rmse(self):
        return self._rmse

    @property
    def timescales(self):
        return self._timescales

    @property
    def M0(self):
        return self._M0

    @property
    def mik(self):
        return self._mik

    def fit(self, input_data, begin, end, rmse_weighted_by_sp=True) :
        """ Construct an IGME model based on the input data

        Parameters
        ----------
        input_data : array like
            shape = (n_lagtime, n_state^2) or (n_lagtime, n_state, n_state)
            Time dependent TPMs from \delta t to n_lagtime \delta t 
            i.e., input_data[0] = trajectory_TPM(\delta t),
                  input_data[1] = trajectory_TPM(2 \delta t),
                  input_data[n_lagtime-1] = trajectory_TPM(n_lagtime \delta t),
        begin : int
            The begin time used to fit IGME. Begin with 1. Should exceed
            the memory relaxation time.
        end : int
            The end time used to fit IGME. Begin with 1. Length of input_data
            will be used instead if end is smaller than begin
        rmse_weighted_by_sp : boolean, default: True
            Allow RMSE weighted by stationary populations

        Returns
        -------
        self : object
            returns the instance itself

        """
        if (numpy.ndim(input_data)==2) :
            self._dim = int(numpy.sqrt(len(input_data[0])))
        elif (numpy.ndim(input_data)==3) :
            if (len(input_data[0]) != len(input_data[0][0])) :
                self._dim = int(numpy.sqrt(len(input_data[0]) * len(input_data[0][0])))
            else :
                self._dim = len(input_data[0])
        else :
            self._dim = 0
        reshaped_input_data = input_data.reshape([len(input_data), self._dim, self._dim]) 
        self._begin = begin
        self._end = end
        if (self._dim <= 0) :
            return 0
        lnTPMs = IGMENS.matrixes_log(reshaped_input_data, self.logarithm_approx_order)
        self._lnA, self._lnTh, self._R = IGMENS.matrixes_lsf(lnTPMs, self._dim, self._begin-1, self._end-1)
        self._Th = IGMENS.matrix_exp(self._lnTh)
        ev = -numpy.linalg.eigvals(self._lnTh)
        ev.sort()
        self._timescales = (1 / ev[1:])
        self._rmse = self._compute_rmse(reshaped_input_data, rmse_weighted_by_sp)
        self._M0 = IGMENS.matrix_log(reshaped_input_data[0], self.logarithm_approx_order) - self._lnTh
        self._mik = numpy.linalg.norm(self._M0) / self._dim
        return self

    def predict(self, begin, end) :
        """ Generate IGME predictions of TPMs at given range of lagtime

        Parameters
        ----------
        begin : int
            The begin time used to fit IGME. Begin with 1. Should exceed
            the memory relaxation time.
        end : int
            The end time used to fit IGME. Begin with 1. Length of input_data
            will be used instead if end is smaller than begin

        Returns
        -------
        predicted_TPMs : array like, shape = (end-begin+1, n_state, n_state)
            The predicted TPMs at given range of lagtime

        """
        if (self._dim <= 0) :
            return 0
        ret = [];
        for i in range(begin-1, end) :
            ret.append(IGMENS.matrix_exp(numpy.multiply(self._lnTh, i) + self._lnA))
        return ret

    def _compute_rmse(self, data, rmse_weighted_by_sp=True) :
        if (self._dim <= 0) :
            return 0
        diff = self.predict(1, len(data)) - data
        if (rmse_weighted_by_sp) :
            sp_matrix = numpy.diag(IGMENS.matrix_exp(numpy.multiply(IGMENS.matrix_log(data[int((self._begin+self._end)/2)], self.logarithm_approx_order), 1e9).reshape([self._dim, self._dim]))[0]);
            for i in range(len(diff)) :
                diff[i] = numpy.matmul(sp_matrix, diff[i])
        return numpy.sqrt(numpy.square(numpy.mean(diff)) + numpy.square(numpy.std(diff)))

    def _init_output_dic(self) :
        dic = {}
        dic['begin']    = []
        dic['end']      = []
        dic['lnTh']     = []
        dic['Th']       = []
        dic['lnA']      = []
        dic['rmse']     = []
        dic['timescales'] = []
        dic['M0']       = []
        dic['mik']      = []
        return dic

    def output(self, dic=0) :
        """ Output the IGME model to a dictionary

        Parameters
        ----------
        dic : dictionary, optional, default: 0
            The dictionary to append current IGME model
            Ignore this parameter if you want to generate a new dictionary

        Returns
        -------
        dic : dictionary
            A dictionary include all the fitted results of IGME
            If input parameter "dic" is given, then output will be the same as
            the input parameter

        """
        if (dic == 0):
            dic = self._init_output_dic()
        dic['begin'].append(self._begin)
        dic['end'].append(self._end)
        dic['lnTh'].append(self._lnTh)
        dic['Th'].append(self._Th)
        dic['lnA'].append(self._lnA)
        dic['rmse'].append(self._rmse)
        dic['timescales'].append(self._timescales)
        dic['M0'].append(self._M0)
        dic['mik'].append(self._mik)
        return dic

    def fit_output(self, input_data, begin, end, rmse_weighted_by_sp=True) :
        """ Construct an IGME model based on the input data

        Parameters
        ----------
        input_data : array like
            shape = (n_lagtime, n_state^2) or (n_lagtime, n_state, n_state)
            Time dependent TPMs from \delta t to n_lagtime \delta t 
            i.e., input_data[0] = trajectory_TPM(\delta t),
                  input_data[1] = trajectory_TPM(2 \delta t),
                  input_data[n_lagtime-1] = trajectory_TPM(n_lagtime \delta t),
        begin : int
            The begin time used to fit IGME. Begin with 1. Should exceed
            the memory relaxation time.
        end : int
            The end time used to fit IGME. Begin with 1. Length of input_data
            will be used instead if end is smaller than begin
        rmse_weighted_by_sp : boolean, default: True
            Allow RMSE weighted by stationary populations

        Returns
        -------
        dic : dictionary
            A dictionary include all the fitted results of IGME

        """
        return self.fit(input_data, begin, end, rmse_weighted_by_sp).output()

    def _print_number_array(self, a, end=",", file=sys.stdout) :
        print("'", end="", file=file)
        for i in range(len(a)) :
            print(a[i], end = " " if i<len(a)-1 else "", file=file)
        print("'", end=end, file=file)

    def print_output(self, out, begin=0, end=0, comma=",", file=sys.stdout, _print_title=True) :
        if _print_title :
            print("#begin,end,rmse,timescales,lnTh,lnA,mik", file=file)
        if (end<=begin) | (end>len(out['begin'])-1) :
            end = len(out['begin']) - 1
        for i in range(begin, end+1) :
            print(out['begin'][i], end=comma, file=file)
            print(out['end'][i], end=comma, file=file)
            print(out['rmse'][i], end=comma, file=file)
            self._print_number_array(out['timescales'][i], end=comma, file=file)
            self._print_number_array(out['lnTh'][i].reshape([self._dim*self._dim]), end=comma, file=file)
            self._print_number_array(out['lnA'][i].reshape([self._dim*self._dim]), end=comma, file=file)
            print(out['mik'][i], end='\n', file=file)

    def scan(self, input_data, begin, end, stride=1, rmse_weighted_by_sp=True, debug=False) :
        """ Scan IGME models within the given range of lagtime

        Parameters
        ----------
        input_data : array like
            shape = (n_lagtime, n_state^2) or (n_lagtime, n_state, n_state)
            Time dependent TPMs from \delta t to n_lagtime \delta t 
            i.e., input_data[0] = trajectory_TPM(\delta t),
                  input_data[1] = trajectory_TPM(2 \delta t),
                  input_data[n_lagtime-1] = trajectory_TPM(n_lagtime \delta t),
        begin : int
            The begin time used to scan IGME models. Begin with 1. Should exceed
            the memory relaxation time.
        end : int
            The end time used to scan IGME models. Begin with 1. Length of
            input_data will be used instead if end is smaller than begin
        stride : int, default: 1
            The stride steps int the scanning.
        rmse_weighted_by_sp : boolean, default: True
            Allow RMSE weighted by stationary populations
        debug : boolean, default: False
            If turned on, then all scanned IGME models will be displayed on the
            screen immediately (in CSV format)

        Returns
        -------
        output : dictionary
            returns a dictionary containing all IGME models

        """
        output = self._init_output_dic()
        _print_title = True
        for iend in range(begin+stride, end+1, stride) :
            for ibegin in range(begin, iend, stride) :
                output_itself = self.fit(input_data, ibegin, iend, rmse_weighted_by_sp).output(output)
                if debug :
                    self.print_output(output, begin=len(output['begin'])-1, _print_title=_print_title)
                    _print_title = False
        return output

