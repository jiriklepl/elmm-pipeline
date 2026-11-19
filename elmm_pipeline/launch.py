"""The environment an application run starts in."""
import os

TRACE_VARIABLES = ("FORT_RUNTIME_TRACE", "FORT_OFFLOAD_TRACE", "CUDA_LAUNCH_BLOCKING")


def environment(threads, *, traced):
    """A fixed OpenMP budget, no inherited application or runtime switches, tracing only on request."""
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("ELMM_", "FORT_")) or name in TRACE_VARIABLES or name == "POISSON_SOLVER_KERNEL_CACHE_DIR":
            del env[name]
    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE", OMP_SCHEDULE="static",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if traced:
        env.update(FORT_RUNTIME_TRACE="1", FORT_OFFLOAD_TRACE="1")
    return env
