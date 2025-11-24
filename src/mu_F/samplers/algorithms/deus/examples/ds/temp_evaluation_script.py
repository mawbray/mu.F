from multiprocessing import Pool, cpu_count
import numpy as np
from pathlib import Path
import pickle

from batch_script import g_func


data_path = Path.cwd()
data_fpne = data_path / "temp_evaluation_data.pkl" 
with open(data_fpne, 'rb') as file:
    data = pickle.load(file)


p_best = np.array([data['p_best']])


def calculate_output_for(ichunk):
    g_list = g_func(ichunk, p_best)
    g_dim = np.shape(g_list[0])[1]
    ochunk = []
    for i, g_vec in enumerate(g_list):
        if np.all(g_vec[0, :] >= 0.0):
            terms = [np.log(1.0 + np.exp(-g)) for g in g_vec[0, :]]
        else:
            terms = []
            for g in g_vec[0, :]:
                if g >= 0.0:
                    term = 0.693147180559945
                else:
                    term = np.log(1.0 + np.exp(-g))
                terms.append(term)
        score = -sum(terms)
        item = {'score': score, 'g_vec': g_vec, 'g_dim': g_dim}
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

    data['out'] = [item['score'] for item in outputs]
    data['g_dim'] = outputs[0]['g_dim']
    data['g_list'] = [item['g_vec'] for item in outputs]


    with open(data_fpne, 'wb') as file:
        pickle.dump(data, file)
