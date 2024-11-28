from multiprocessing import Pool, cpu_count
import numpy as np
from pathlib import Path
import pickle

from ds_test2_user_script import g_func


data_path = Path.cwd()
data_fpne = data_path / "temp_evaluation_data.pkl" 
with open(data_fpne, 'rb') as file:
    data = pickle.load(file)


p_samples = data['p_samples']


def calculate_output_for(ichunk):
    p_num = len(p_samples)
    p_dim = len(p_samples[0]['c'])
    p_mat = np.empty((p_num, p_dim))
    for i, p_sample in enumerate(p_samples):
        p_mat[i, :] = p_sample['c']

    d_shape = np.shape(ichunk)
    if len(d_shape) == 1:
        d_num, d_dim = 1, d_shape
    else:
        d_num, d_dim = d_shape

    n_model_evals = p_num
    g_mat_list = g_func(ichunk, p_mat)
    ochunk = []
    for i, g_mat in enumerate(g_mat_list):
        efp = 0.0
        for j, g_vec in enumerate(g_mat):
            if np.all(g_vec >= 0.0):
                efp = round(efp + p_samples[j]['w'], ndigits=15)
        item = {'efp': efp, 'nme': n_model_evals}
        ochunk.append(item)
    return ochunk


if __name__ == "__main__":

    inputs = data['in']
    n_processes = cpu_count()
    n_inputs = int(len(inputs))
    chunk_size = int(n_inputs/n_processes)
    input_chunks = [inputs[i*chunk_size:(i+1)*chunk_size] for i in range(n_processes-1)]
    input_chunks.append(inputs[(n_processes-1)*chunk_size:])
    with Pool(n_processes) as the_pool:
        output_chunks = the_pool.map(calculate_output_for, input_chunks)


    outputs = []
    for chunk in output_chunks:
        outputs.extend(chunk)

    data['out'] = [item['efp'] for item in outputs]
    data['n_model_evals'] = [item['nme'] for item in outputs]
    data['g_list'] = []


    with open(data_fpne, 'wb') as file:
        pickle.dump(data, file)
