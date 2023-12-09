# NFGen
NFGen: Automatic Non-Linear Function Evaluation Code Generator for General-purpose MPC Platforms

## Introduction

NFGen is designed to generate non-linear function evaluation code for general-purpose MPC platforms. 

The requirement for the underlying MPC platform is three basic routines *addition*, *multiplication* and *greater-than* are provided and these three routines can be composed sequentially.

This repo contains all the source files for developers to build their improvements with. For users, you can simply install NFGen with

```
pip install NFGen
```

## Example

We use one function ``sigmoid`` and one MPC deployment ``MP-SPDZ/rep-ring`` protocol for example. 

The following demo is in ``./tests/example.py``. 

1) Import the required package
```python
import sympy as sp
from NFGen.main import generate_nonlinear_config
import NFGen.CodeTemplet.templet as temp
import NFGen.PerformanceModel.time_ops as to
```

2) Write the non-linear function definition config (NFD)
```python
platform = "Rep3" # using MP-SPDZ Rep3 protocol as an example.
f = 48
n = 96
profiler_file = './NFGen/PerformanceModel/' + platform + "_kmProfiler.pkl"

# fundenmental functions, indicating they are cipher-text non-linear operations.
def func_reciprocal(x):
        return 1 / x

def func_exp(x, lib=sp):
    return lib.exp(x)

# target function.
def sigmoid(x):
    return 1 * func_reciprocal((1 + func_exp(-x)))

# define NFD
sigmoid_config = {
    "function": sigmoid, # function config.
    "range": (-10, 10),
    "k_max": 10, # set the maximum order.
    "tol": 1e-3, # percision config.
    "ms": 1000, # maximum samples.
    "zero_mask": 1e-6,
    "n": n, # <n, f> fixed-point config.
    "f": f,
    "profiler": profiler_file, # profiler model source file.
    "code_templet": temp.templet_spdz, # spdz templet.
    "code_language": "python", # indicating the templet language.
    "config_file": "./sigmoig_spdz.py", # generated code file.
    "time_dict": to.basic_time[platform], # basic operation time cost.
    # "test_graph": "./graph/" # (optional, need mkdir for this folder first), whether generate the graph showing the approximation and the real function.
}
```
3) Generate the evaluation code using NFGen.
```python
# using NFGen to generate the target function code.
generate_nonlinear_config(sigmoid_config)
```

## Help

Feel free to contact Xiaoyu Fan fxy23@mails.tsinghua.edu.cn for any assistance about NFGen or raise issues in this repo.

## Citing

NFGen is described in [this paper](https://dl.acm.org/doi/pdf/10.1145/3548606.3560565), for academic usage, please cite:
```
@inproceedings{fan2022nfgen,
  title={NFGen: Automatic Non-linear Function Evaluation Code Generator for General-purpose MPC Platforms},
  author={Fan, Xiaoyu and Chen, Kun and Wang, Guosai and Zhuang, Mingchun and Li, Yi and Xu, Wei},
  booktitle={Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security (CCS)},
  year={2022}
}
```
