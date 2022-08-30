"""Genrate the final execution code for given MPC system M.
"""
import sympy as sp
import numpy as np
from dill.source import getsource
import functools
import operator


basic_building_blocks = {
    'func_reciprocal': "mpc_reciprocal",
    'func_sqrt': "mpc_sqrt",
    'func_log': "mpc_log",
    'func_exp': "mpc_exp",
    'func_pow': "mpc_pow"
}
library_prefix = {'sp.', 'np.'}



def predict_sequential_time(func, basic_time):
    func_string = getsource(func)
    count_bb = {key: func_string.count(key) for key in basic_time.keys()}
    time_seq = 0
    for key in count_bb:
        time_seq += count_bb[key] * basic_time[key]
    
    return time_seq


def code_generate_flp(km_config, func_name, code_templet, save_file):
    """generate the floating-point computation code for mp-spdz.
    """
    templet = code_templet.split("# insert here")
    templet[0] = templet[0].replace("general_non_linear_flp", func_name)
    config_code = "\n    breaks = " + str(km_config['breaks']) + "\n    coeffA = " + str(km_config['coeffA'])
    exec_code = templet[0] + config_code + templet[1]
    
    exec_code += "\n\n"
    with open(save_file, 'a') as f:
        f.write(exec_code)
    f.close()
    print("Write "+func_name+" SUCCESS~")
    
    return

    
def code_generate(km_config, profiler, func, basic_time, code_templet, basic_building_blocks, save_file, nick_name=None, not_check=False, code_language="python"):
    """Generate the performant execution code with minimal profiled execution time.
    """
    k, m = len(km_config['coeffA'][0]), len(km_config['coeffA'])
    time_km = profiler.predict(np.array([[m, k]]))
    time_seq = predict_sequential_time(func, basic_time)
    exec_code = getsource(func)
    exp_flag = (exec_code.count("func_exp")) > 0
    if nick_name is None:
        func_name = exec_code[exec_code.index("def ")+4:exec_code.index("(")]
    else:
        func_name = nick_name
        
    # templet = code_templet.split("# insert here")
    # code selection -> 1) efficiency, which perform more efficient; 2) accuracy, avoiding expensive cases.
    if not_check:
        flag_km = True
    else:
        flag_km = ((time_km < time_seq) or exp_flag)
        
    print("Time pred:\n  Approx: %.3g s\n  Direct Evaluation: %.3g s"%(time_km, time_seq))
    print("Whether use approximation =>", flag_km)
    exec_code = None
    if flag_km:
        if code_language == "python": # python templet
            templet = code_templet.split("# insert here")
            templet[0] = templet[0].replace("general_non_linear_func", func_name)
            config_code = "\n    breaks = " + str(km_config['breaks'][:-1]) + "\n    coeffA = " + str(km_config['coeffA']) + "\n    scaler = " + str(km_config['scaler'])
            exec_code = templet[0] + config_code + templet[1]
            
        elif code_language == "cpp": # sequential cpp templet
            templet = code_templet.split("# insert here")
            m = len(km_config['breaks']) - 1
            k = len(km_config['coeffA'][0])
            templet[0] = templet[0].replace("general_non_linear_func", func_name)
            config_code = "\n  const size_type M = "+ str(m) + ";\n  const size_type K = " + str(k) + ";\n  const double Breaks[M] = {" + str(km_config['breaks'][:-1])[1:-1] + "};" + "\n  const double CoeffA[M * K] = {" + str(functools.reduce(operator.iconcat, km_config['coeffA'], []))[1:-1] + "};" + "\n  const double Scaler[M * K] = {" + str(functools.reduce(operator.iconcat, km_config['scaler'], []))[1:-1] + "};\n"
            exec_code = templet[0] + config_code + templet[1]
            
        elif code_language == "cpp2": # multi-thread cpp templet
            code_templet = code_templet.replace("general_non_linear_func", func_name)
            templet = code_templet.split("# insert here")
            config_code = "\n  const size_type M = "+ str(m) + ";\n  const size_type K = " + str(k) + ";\n  const double Breaks[M] = {" + str(km_config['breaks'][:-1])[1:-1] + "};" + "\n  const double CoeffA[M * K] = {" + str(functools.reduce(operator.iconcat, km_config['coeffA'], []))[1:-1] + "};" + "\n  const double Scaler[M * K] = {" + str(functools.reduce(operator.iconcat, km_config['scaler'], []))[1:-1] + "};\n"
            k_code = "const size_type K = " + str(k) + ";\n"
            exec_code = templet[0] + config_code + templet[1] + k_code + templet[2]
    else:
        exec_code = getsource(func)
        for key in basic_building_blocks:
            exec_code = exec_code.replace(key, basic_building_blocks[key])
    exec_code += "\n\n"
    with open(save_file, 'a') as f:
        f.write(exec_code)
    f.close()
    print("Write "+func_name+ " in " + save_file +" SUCCESS!")
    return exec_code
   
    
    
if __name__ == "__main__":
    
    TAU_2 = 0.959502
    # ## test function
    def func_exp(x, lib=sp):
        return lib.exp(x)

    def func_reciprocal(x):
        return 1/x
    
    def func_log(x, lib=sp):
        return lib.log(x)
    
    def func_sqrt(x, lib=sp):
        return lib.sqrt(x)
    


    
   