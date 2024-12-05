import jax.numpy as jnp
from jax import vmap, jit



""" insert case study specific functions for graph edges here"""


""" tablet press methods """
@jit
def data_IO_1(steady_state_outputs):
    # get mass flowrate of api and exipient
    return (steady_state_outputs[-2:].reshape(1, -1) * steady_state_outputs[-3
    ].reshape(1, -1)).reshape(-1,)


@jit
def data_IO_2(steady_state_outputs):
    # get porosity
    return steady_state_outputs[-1].squeeze()

vmap_data_IO_1 = vmap(vmap(data_IO_1, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)
vmap_data_IO_2 = vmap(vmap(data_IO_2, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)


""" batch reactor methods """
@jit
def data_transform(dynamic_profile):
    x = dynamic_profile[:-1].reshape(
        1, -1
    )
    return x.squeeze()

vmap_data_transform = vmap(vmap(data_transform, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)

@jit
def in_out_transform(x):
    return jnp.expand_dims(x[:,:-3], axis=1)

""" insert case study specific functions for constraints here"""
rl_dict = {(key,key+1): in_out_transform for key in range(12)}
CS_edge_holder = {'tablet_press': {(0,1): data_IO_1, (1,2): data_IO_2}, 'serial_mechanism_batch': {(0,1): data_transform}, 'constrained_rl': rl_dict}
vmap_CS_edge_holder = {'tablet_press': {(0,1): vmap_data_IO_1, (1,2): vmap_data_IO_2}, 'serial_mechanism_batch': {(0,1): vmap_data_transform}, 'constrained_rl': {key: vmap(vmap(value, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1) for key, value in rl_dict.items()}}

