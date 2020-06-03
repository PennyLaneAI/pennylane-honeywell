# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Honeywell Quantum Solutions device class
========================================

This module contains an abstract base class for constructing HQS devices for PennyLane.

"""
from pennylane import QubitDevice

from ._version import __version__

class HQSDevice(QubitDevice):
    r"""HQS device for PennyLane.

    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables
        api_key (str): The HQS API key. If not provided, the environment
            variable ``HQS_TOKEN`` is used.
        retry_delay (float): The time (in seconds) to wait between requests
            to the remote server when checking for completion of circuit
            execution.
    """
    # pylint: disable=too-many-instance-attributes
    name = "Honeywell Quantum Solutions PennyLane plugin"
    pennylane_requires = ">=0.9.0"
    version = __version__
    author = "Xanadu Inc."
    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True,
    }

    short_name = "hqs.base_device"
    _operation_map = {
        "RX": "X",
        "RY": "Y",
        "RZ": "Z",
        "BasisState": None,
        "PauliX": None,
        "PauliY": None,
        "PauliZ": None,
        "Hadamard": None,
    }

    BASE_HOSTNAME = ""
    TARGET_PATH = ""
    HTTP_METHOD = "POST"

    def __init__(self, wires, shots=1000, api_key=None, retry_delay=0.05):
        super().__init__(wires=wires, shots=shots, analytic=False)
        self.shots = shots
        self._retry_delay = retry_delay
        self._api_key = api_key
        self.set_api_configs()
        self.reset()

    def reset(self):
        """Reset the device."""
        pass

    def set_api_configs(self):
        """
        Set the configurations needed to connect to HQS API.
        """
        self._api_key = self._api_key or os.getenv("HQS_TOKEN")
        if not self._api_key:
            raise ValueError("No valid api key for HQS platform found.")
        self.header = {"User-Agent": "pennylane-hqs_v{}".format(__version__)}
        self.hostname = urllib.parse.urljoin("{}/".format(self.BASE_HOSTNAME), self.TARGET_PATH)

    @property
    def retry_delay(self):
        """
        The time (in seconds) to wait between requests
        to the remote server when checking for completion of circuit
        execution.

        """
        return self._retry_delay

    @retry_delay.setter
    def retry_delay(self, time):
        """Changes the devices's ``retry_delay`` property.

        Args:
            time (float): time (in seconds) to wait between calls to remote server

        Raises:
            DeviceError: if the retry delay is not a positive number
        """
        if time <= 0:
            raise DeviceError(
                "The specified retry delay needs to be positive. Got {}.".format(time)
            )

        self._retry_delay = float(time)

    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    def apply(self, operations, **kwargs):
        pass
