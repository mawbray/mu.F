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

""" convex estimator methods """
@jit
def data_transform_cvx(dynamic_profile):
    return dynamic_profile


vmap_data_transform_cvx = vmap(vmap(data_transform_cvx, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)

@jit 
def affine_cs34(dynamic_profile):
    return dynamic_profile[0]

@jit 
def affine_cs35(dynamic_profile):
    return dynamic_profile[1]

vmap_cs34 = vmap(vmap(affine_cs34, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)
vmap_cs35 = vmap(vmap(affine_cs35, in_axes=(0), out_axes=0), in_axes=(1), out_axes=1)


""" insert case study specific functions for constraints here"""
CS_edge_holder = {  'tablet_press': {(0,1): data_IO_1, (1,2): data_IO_2}, 'serial_mechanism_batch': {(0,1): data_transform}, 
                    'convex_estimator': {(0,5): data_transform_cvx, (1,5): affine_cs34, (2,5): affine_cs34, 
                                            (3,5): data_transform_cvx, (4,5): data_transform_cvx},
                    'convex_underestimator': {(0,5): data_transform_cvx, (1,5): data_transform_cvx, (2,5): data_transform_cvx, 
                                        (3,5): data_transform_cvx, (4,5): data_transform_cvx},
                    'affine_study': {(0,2): data_transform_cvx, (1,2): data_transform_cvx, (2,3): affine_cs34, (2,4): affine_cs35}}

vmap_CS_edge_holder = {'tablet_press': {(0,1): vmap_data_IO_1, (1,2): vmap_data_IO_2}, 'serial_mechanism_batch': {(0,1): vmap_data_transform},
                       'estimator': {(0,5): vmap_data_transform_cvx, (1,5): vmap_cs34, (2,5): vmap_cs34,
                                            (3,5): vmap_data_transform_cvx, (4,5): vmap_data_transform_cvx},
                       'convex_estimator': {(0,5): vmap_data_transform_cvx, (1,5): vmap_data_transform_cvx, (2,5): vmap_data_transform_cvx, 
                                            (3,5): vmap_data_transform_cvx, (4,5): vmap_data_transform_cvx},
                        'convex_underestimator': {(0,5): vmap_data_transform_cvx, (1,5): vmap_cs34, (2,5): vmap_cs34, 
                                            (3,5): vmap_data_transform_cvx, (4,5): vmap_data_transform_cvx},
                        'affine_study': {(0,2): vmap_data_transform_cvx, (1,2): vmap_data_transform_cvx, (2,3): vmap_cs34, (2,4): vmap_cs35}}

