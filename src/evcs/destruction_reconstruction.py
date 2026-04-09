# This file is kept for backward compatibility.
# All functions have been moved to dedicated modules:
#
#   evcs.destroy       -> destroy_multi_u
#   evcs.greedy        -> reconstruct_multi_u_greedy, reconstruct_u_dict_fast
#   evcs.local_search  -> local_search_u_proxy
#   evcs.full_eval     -> covered_by_period, full_eval_from_U
#   evcs.proxy         -> evaluate_u_numpy_greedy, evaluate_u_numpy_greedy_jt, evaluate_u_numpy_greedy_binary
#   evcs.utils         -> _clone_u_matrix, _apply_u_matrix, _u_to_capacity_array
#   evcs.dr            -> DRLogger, default_parameters_binary, default_parameters_integer,
#                         run_one_policy_multi, run_DR_multi
#
# run_one_policy (single-period) was deleted — it was not called anywhere.

from evcs.destroy import destroy_multi_u
from evcs.greedy import reconstruct_multi_u_greedy, reconstruct_u_dict_fast
from evcs.local_search import local_search_u_proxy
from evcs.full_eval import covered_by_period, full_eval_from_U
from evcs.proxy import (
    evaluate_u_numpy_greedy,
    evaluate_u_numpy_greedy_jt,
    evaluate_u_numpy_greedy_binary,
)
from evcs.utils import _clone_u_matrix, _apply_u_matrix, _u_to_capacity_array
from evcs.dr import (
    DRLogger,
    default_parameters_binary,
    default_parameters_integer,
    run_one_policy_multi,
    run_DR_multi,
)

__all__ = [
    "destroy_multi_u",
    "reconstruct_multi_u_greedy",
    "reconstruct_u_dict_fast",
    "local_search_u_proxy",
    "covered_by_period",
    "full_eval_from_U",
    "evaluate_u_numpy_greedy",
    "evaluate_u_numpy_greedy_jt",
    "evaluate_u_numpy_greedy_binary",
    "_clone_u_matrix",
    "_apply_u_matrix",
    "_u_to_capacity_array",
    "DRLogger",
    "default_parameters_binary",
    "default_parameters_integer",
    "run_one_policy_multi",
    "run_DR_multi",
]
