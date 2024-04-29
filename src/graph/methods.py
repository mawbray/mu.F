import jax.numpy as jnp
from jax import vmap, jit



""" insert case study specific functions for graph edges here"""


""" tablet press methods """

@jax.jit
def data_IO_1(steady_state_outputs):
    # get mass flowrate of api and exipient
    return steady_state_outputs[-2:].reshape(1, -1) * steady_state_outputs[-3
    ].reshape(1, -1)


@jax.jit
def data_IO_2(steady_state_outputs):
    # get porosity
    return steady_state_outputs[-1].reshape(1, -1)

vmap_data_IO_1 = vmap(vmap(data_IO_1, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=1)
vmap_data_IO_2 = vmap(vmap(data_IO_2, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=1)



""" batch reactor methods """
@jax.jit
def data_transform(dynamic_profile):
    x = jnp.hstack([dynamic_profile[:-1].reshape(1, -1), jnp.zeros((1, 1))]).reshape(
        1, -1
    )
    return x

vmap_data_transform = vmap(vmap(data_transform, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=1)



""" insert case study specific functions for constraints here"""
CS_holder = {'tablet_press': {(0,1): data_IO_1, (1,2): data_IO_2}, 'CSTR': {(0,1): data_transform}}
vmap_CS_holder = {'tablet_press': {(0,1): vmap_data_IO_1, (1,2): vmap_data_IO_2}, 'CSTR': {(0,1): vmap_data_transform}}

