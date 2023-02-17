"""
Utilies for the gates.
"""


def construct_x_gate_args(device_param_lookup: dict, noise_scaling: float=1.0, phi: float=0.0):
    """ Takes the device parameters as lookup, and returns the corresponding arguments to sample an X gate. One can set
        a value 0 <= noise_scaling with which the noise is multiplied.
    """

    assert noise_scaling >= 0, "Negative noise values are not physical."
    noise_scaling = max(noise_scaling, 1e-15)

    return {
        "phi": phi,
        "p": device_param_lookup["p"][0] * noise_scaling,
        "T1": device_param_lookup["T1"][0] / noise_scaling,
        "T2": device_param_lookup["T2"][0] / noise_scaling
    }


def construct_cnot_gate_args(device_param_lookup: dict, noise_scaling: float=1.0, phi_ctr: float=0.0, phi_trg: float=0.0):
    """ Takes the device parameters as lookup, and returns the corresponding arguments to sample an X gate. One can set
        a value 0 <= noise_scaling with which the noise is multiplied.
    """

    assert noise_scaling >= 0, "Negative noise values are not physical."
    noise_scaling = max(noise_scaling, 1e-15)

    return {
        "phi_ctr": phi_ctr,
        "phi_trg": phi_trg,
        "t_cnot": device_param_lookup["t_cnot"][0][1],
        "p_cnot": device_param_lookup["p_cnot"][0][1] * noise_scaling,
        "p_single_ctr": device_param_lookup["p"][0] * noise_scaling,
        "p_single_trg": device_param_lookup["p"][1] * noise_scaling,
        "T1_ctr": device_param_lookup["T1"][0] / noise_scaling,
        "T2_ctr": device_param_lookup["T2"][0] / noise_scaling,
        "T1_trg": device_param_lookup["T1"][1] / noise_scaling,
        "T2_trg": device_param_lookup["T2"][1] / noise_scaling
    }
