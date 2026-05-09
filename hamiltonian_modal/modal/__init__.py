"""Modal decomposition package."""

from hamiltonian_modal.modal.basis_switching import (
    select_low_frequency_modes,
    select_modes,
    select_modes_by_frequency_range,
)
from hamiltonian_modal.modal.decoder import decode_momentum, decode_position, decode_velocity
from hamiltonian_modal.modal.decomposition import ModalBasis, compute_modal_basis, is_m_orthonormal
from hamiltonian_modal.modal.encoder import encode_momentum, encode_position, encode_velocity
from hamiltonian_modal.modal.stiffness import diagonal_stiffness, stiffness_from_gravity_hessian

__all__ = [
    # decomposition
    "ModalBasis",
    "compute_modal_basis",
    "is_m_orthonormal",
    # stiffness
    "diagonal_stiffness",
    "stiffness_from_gravity_hessian",
    # encoder
    "encode_position",
    "encode_velocity",
    "encode_momentum",
    # decoder
    "decode_position",
    "decode_velocity",
    "decode_momentum",
    # basis switching
    "select_modes",
    "select_low_frequency_modes",
    "select_modes_by_frequency_range",
]
