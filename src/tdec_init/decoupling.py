import jax
from decoupling import Algorithm 
from decoupling.result import Decoupling
from decoupling.scaler import JacobianScaler

def cmtf_bsd(
    inputs,
    outputs,
    jacobians,
    rank: int,
    key: jax.Array,
    niters: int = 100,
    ninits: int = 5,
    splines_dof = None,
) -> Decoupling:

    scaler = JacobianScaler(jacobians)
    print(scaler.scaling_factors)
    #exit()
    jacobians, outputs = scaler.scale(jacobians, outputs)

    algorithm = Algorithm(rank, key, niters, ninits, splines_dof=splines_dof, use_smoothing=False, show_progress=True)

    print('computing the decoupling...', flush=True, end=' ')

    decoupling = algorithm.run(inputs, outputs, jacobians)

    print(f'done, best cpd error {algorithm.info.errors.min():.5f}', flush=True)

    return scaler.unscale(decoupling)
