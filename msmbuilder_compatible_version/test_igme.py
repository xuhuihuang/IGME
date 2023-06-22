import os
import sys
import numpy
from igme import IGME

# -----------------------------------------------------------------------------
# followings are the code to run this script in terminal

# code entry

# input

env_fit_input = os.getenv('input');
env_fit_begin = os.getenv('begin');
env_fit_end = os.getenv('end');
env_fit_stride = os.getenv('stride');
env_fit_scan = os.getenv('scan');
env_log_order = os.getenv('logarithm_approx_order');

if not env_fit_input:
    print("test_igme.py : perform IGME based on least square fitting")
    print("  usage: input=... begin=... end=... [scan=TRUE/true/false stride=...] [logarithm_approx_order=0] python test_igme.py")
    sys.exit()

if not os.path.exists(env_fit_input):
    print("test_igme.py : error : cannot open `"+env_fit_input+"'")
    sys.exit()

raw_TPMs = numpy.loadtxt(env_fit_input)
# raw_TPMs = numpy.loadtxt("ala2-pccap-4states-0.1ps-50ps.txt")
dimension = int(numpy.sqrt(len(raw_TPMs[0])))
input_len = len(raw_TPMs)

fit_begin = 1
fit_end = input_len-1
fit_stride = 1
logarithm_approx_order = 0
if env_fit_begin:
    fit_begin = int(env_fit_begin)
if env_fit_end:
    fit_end = int(env_fit_end)
if env_fit_stride:
    fit_stride = int(env_fit_stride)
if env_log_order:
    logarithm_approx_order = int(env_log_order)

# perform IGME

if not env_fit_scan:
    env_fit_scan = "False"

rmse_weighted_by_sp=True

if (env_fit_scan=="TRUE") | (env_fit_scan=="YES"):
  # scan hyperparameters
    igme = IGME(logarithm_approx_order=logarithm_approx_order)
    result = igme.scan(raw_TPMs, begin=fit_begin, end=fit_end, stride=fit_stride, debug=False, rmse_weighted_by_sp=rmse_weighted_by_sp)
    igme.print_output(result)
elif (env_fit_scan=="True") | (env_fit_scan=="true") | (env_fit_scan=="Yes") | (env_fit_scan=="yes"):
  # scan hyperparameters
    igme = IGME(logarithm_approx_order=logarithm_approx_order)
    result = igme.scan(raw_TPMs, begin=fit_begin, end=fit_end, stride=fit_stride, debug=True, rmse_weighted_by_sp=rmse_weighted_by_sp)
else :
  # perform IGME once
    igme = IGME(logarithm_approx_order=logarithm_approx_order)
    result = igme.fit_output(raw_TPMs, begin=fit_begin, end=fit_end, rmse_weighted_by_sp=rmse_weighted_by_sp)
    igme.print_output(result)

