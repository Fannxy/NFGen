"""Non-linear functions micro-benchmark, totally 15 functions.

Machine Learning functions: 'soft_plus', 'tanh', 'sigmoid', 'elu', 'selu', 'gelu', 'soft_sign', 'isru'
Probability functions: 'snormal_dis', 'scauchy_dis', 'gamma_dis', 'chi_square', 'sexp_dis', 'slog_dis', 'bs_dis'
"""
from NFGen.main import generate_nonlinear_config
import NFGen.CodeTemplet.templet as temp
import NFGen.PerformanceModel.time_ops as to
import sympy as sp
import copy
import math
import numpy as np
import os

# constant factors
PAI = 3.1415926
TAU_2 = 0.959502
ALPHA1 = 1.0
ALPHA2 = 1.6732632
LAMBDA = 1.0507010
E = 2.7182818
C1 = 0.044715
TAU_half = 1.7725
G = 0.5

# exp_list = ["Rep3", "RepPrime", "PsRepRing", "PsRepPrime", "Shamir3", "PrivPy"]
exp_list = ["Rep3"] # test platforms.

# micro_benchmark of activation functions.
ml_config_list = [
    'soft_plus', 'tanh', 'sigmoid', 'elu', 'selu', 'gelu', 'soft_sign', 'isru'
]
# micro_benchmark of probability distribution functions.
prob_config_list = [
    'snormal_dis', 'scauchy_dis', 'gamma_dis', 'chi_square', 'sexp_dis',
    'slog_dis', 'bs_dis'
]

for exp in exp_list:
    print("Exp: ", exp)

    if exp == "PrivPy":
        f = 45
        n = 128
        config_folder = './Demo/PrivPy/'
    else:
        f = 44
        n = 96
        config_folder = './Demo/SPDZ/'
    
    if not os.path.exists(config_folder):
        os.makedirs(config_folder)

    profiler_file = '../src/NFGen/PerformanceModel/' + exp + '_kmProfiler.pkl' # path of profiler models, change to your project file location.

    if exp == "PrivPy":
        ml_config_dict_tmp = {
            'k_max': 10,
            'tol': 1e-3,
            'n': n,
            'f': f,
            'zero_mask': 1e-6,
            # 'test_graph': config_folder # if you want to generate the graph of target generated functions and errors, add this key.
            'profiler': profiler_file,
            'config_file': config_folder + exp + '_non_linear.cpp',
            'code_templet': temp.templet_privpy_cpp,
            'code_language': "cpp",
            'time_dict': to.basic_time[exp],
            'not_check': True
        }
        prob_config_dict_tmp = {
            'k_max': 10,
            'tol': 1e-3,
            'n': n,
            'f': f,
            'zero_mask': 1e-6,
            'profiler': profiler_file,
            'config_file': config_folder + exp + '_non_linear.cpp',
            'code_templet': temp.templet_privpy_cpp2,
            'code_language': "cpp2",
            'ms': 1000,
            'time_dict': to.basic_time[exp],
            'not_check': True
        }
    else:
        ml_config_dict_tmp = {
            'k_max': 10,
            'tol': 1e-3,
            'n': n,
            'f': f,
            'zero_mask': 1e-6,
            'profiler': profiler_file,
            'config_file': config_folder + exp + '_code.py',
            'code_templet': temp.templet_spdz,
            'code_language': "python",
            'time_dict': to.basic_time[exp],
        }
        prob_config_dict_tmp = {
            'k_max': 10,
            'tol': 1e-3,
            'n': n,
            'f': f,
            'zero_mask': 1e-6,
            'profiler': profiler_file,
            'config_file': config_folder + exp + '_code.py',
            'code_templet': temp.templet_spdz,
            'code_language': "python",
            'ms': 1000,
            'time_dict': to.basic_time[exp],
        }

    # Learning related functions, refering to:
    # 1. https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
    # 2. http://www.gabormelli.com/RKB/Inverse_Square_Root_Unit_(ISRU)_Activation_Function

    def tanh(x):
        ep = func_exp(x)
        en = func_exp(-x)
        return (ep - en) * func_reciprocal(ep + en)

    tanh_config = copy.deepcopy(ml_config_dict_tmp)
    tanh_config['function'] = tanh
    tanh_config['range'] = (-50, 50)

    def soft_plus(x):
        return func_log(1 + func_exp(x))

    soft_plus_config = copy.deepcopy(ml_config_dict_tmp)
    soft_plus_config['function'] = soft_plus
    soft_plus_config['range'] = (-20, 50)

    def sigmoid(x):
        return 1 * func_reciprocal((1 + func_exp(-x)))

    sigmoid_config = copy.deepcopy(ml_config_dict_tmp)
    sigmoid_config['function'] = sigmoid
    sigmoid_config['range'] = (-50, 50)

    def elu(x):
        """Reference: https://arxiv.org/pdf/1511.07289.pdf
        """
        pos_flag = x > 0
        res = x * pos_flag + (1 - pos_flag) * ALPHA1 * (func_exp(x, lib=np) -
                                                        1)
        return res

    elu_config = copy.deepcopy(ml_config_dict_tmp)
    elu_config['function'] = elu
    elu_config['range'] = (-50, 20)
    elu_config['derivative_flag'] = False

    def selu(x):
        """Reference: https://mlfromscratch.com/activation-functions-explained/#/
        """
        pos_flag = x > 0
        res = LAMBDA * x * pos_flag + (1 - pos_flag) * LAMBDA * (
            ALPHA2 * func_exp(x, lib=np) - ALPHA2)
        return res

    selu_config = copy.deepcopy(ml_config_dict_tmp)
    selu_config['function'] = selu
    selu_config['range'] = (-50, 20)
    selu_config['derivative_flag'] = False

    def gelu(x):
        constant = math.sqrt(2 / PAI)
        x1 = constant * (x + C1 * x * x * x)
        ep = func_exp(x1, lib=np)
        en = func_exp(-x1, lib=np)
        return 0.5 * x * (1 + ((ep - en) * func_reciprocal(ep + en)))

    gelu_config = copy.deepcopy(ml_config_dict_tmp)
    gelu_config['function'] = gelu
    gelu_config['range'] = (-20, 20)
    gelu_config['ms'] = 1000
    gelu_config['derivative_flag'] = False

    def soft_sign(x):
        pos_flag = x > 0
        abs_x = (pos_flag * x) + (1 - pos_flag) * (-x)
        return x * func_reciprocal(1 + abs_x)

    soft_sign_config = copy.deepcopy(ml_config_dict_tmp)
    soft_sign_config['function'] = soft_sign
    soft_sign_config['range'] = (-50, 50)
    soft_sign_config['derivative_flag'] = False

    def isru(x):
        """Inverse Square Root Unit.
        """
        return x * func_reciprocal(func_sqrt(1 + ALPHA1 * x**2))

    isru_config = copy.deepcopy(ml_config_dict_tmp)
    isru_config['function'] = isru
    isru_config['range'] = (-50, 50)

    # probability functions.
    def snormal_dis(x):
        """https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm
        """
        return func_exp(((-x**2) / 2)) * func_reciprocal(func_sqrt(2 * PAI))

    snormal_dis_config = copy.deepcopy(prob_config_dict_tmp)
    snormal_dis_config['function'] = snormal_dis
    snormal_dis_config['range'] = (-10, 10)

    def scauchy_dis(x):
        """https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
        """
        return 1 * func_reciprocal((PAI * (1 + x**2)))

    scauchy_dis_config = copy.deepcopy(prob_config_dict_tmp)
    scauchy_dis_config['function'] = scauchy_dis
    scauchy_dis_config['range'] = (-40, 40)

    def gamma_dis(x):
        """fix gamma = 2
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda366b.htm
        """
        return (func_sqrt(x) * func_exp(-x)) / TAU_half

    gamma_dis_config = copy.deepcopy(prob_config_dict_tmp)
    gamma_dis_config['function'] = gamma_dis
    gamma_dis_config['range'] = (10**(-6), 50)

    def chi_square(x):
        """fix v = 4.
        """
        return (func_exp(-x / 2) * x) / (4 * TAU_2)

    chi_square_config = copy.deepcopy(prob_config_dict_tmp)
    chi_square_config['function'] = chi_square
    chi_square_config['range'] = (0, 30)

    def sexp_dis(x):
        """https://www.itl.nist.gov/div898/handbook/eda/section3/eda3667.htm
        """
        # return E**(-x)
        return func_exp(-x)

    sexp_dis_config = copy.deepcopy(prob_config_dict_tmp)
    sexp_dis_config['function'] = sexp_dis
    sexp_dis_config['range'] = (0, 10)

    def slog_dis(x):
        return (func_exp(-(func_log(x)**2) / 2)) * func_reciprocal(
            (x * math.sqrt(2 * PAI)))

    slog_dis_config = copy.deepcopy(prob_config_dict_tmp)
    slog_dis_config['function'] = slog_dis
    slog_dis_config['range'] = (10**-4, 40)

    def bs_dis(x):
        x1 = (func_sqrt(x) - func_sqrt(1 / x)) / G
        x1 = snormal_dis(x1)
        x2 = (func_sqrt(x) + func_sqrt(1 / x)) / 2 * G * x
        return x1 * x2

    bs_dis_config = copy.deepcopy(prob_config_dict_tmp)
    bs_dis_config['function'] = bs_dis
    bs_dis_config['range'] = (1e-6, 30)

    # fundenmental functions.
    def func_reciprocal(x):
        return 1 / x

    def func_exp(x, lib=sp):
        return lib.exp(x)

    def func_sqrt(x, lib=sp):
        return lib.sqrt(x)

    def func_log(x, lib=sp):
        return lib.log(x)

    def func_pow(a, x):
        return a**x

    for func_name in ml_config_list:
        print("Func: ", func_name)
        config = eval(func_name + '_config')
        generate_nonlinear_config(config)
        print("\n\n")

    for func_name in prob_config_list:
        print("Func: ", func_name)
        config = eval(func_name + '_config')
        generate_nonlinear_config(config)
        print("\n\n")