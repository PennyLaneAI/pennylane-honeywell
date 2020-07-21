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
import os
import json
import warnings
from time import sleep

import requests

import numpy as np

from pennylane import QubitDevice, DeviceError
from pennylane.operation import Sample

from ._version import __version__

OPENQASM_GATES = {
    "CNOT": "cx",
    "CZ": "cz",
    "U3": "u3",
    "U2": "u2",
    "U1": "u1",
    "Identity": "id",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "Hadamard": "h",
    "S": "s",
    "S.inv": "sdg",
    "T": "t",
    "T.inv": "tdg",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "SWAP": "swap",
    "Toffoli": "ccx",
    "CSWAP": "cswap",
    "PhaseShift": "u1",
}


class HQSDevice(QubitDevice):
    r"""Honeywell Quantum Services device for PennyLane.

    Args:
        wires (int): the number of wires to initialize the device with
        machine (str): name of the Honeywell machine to execute on
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

    short_name = "honeywell.hqs"
    _operation_map = {**OPENQASM_GATES}

    BASE_HOSTNAME = "https://qapi.honeywell.com/v1"
    TARGET_PATH = "job"
    TERMINAL_STATUSES = ["failed", "completed", "cancelled"]
    PRIORITY = "normal"
    LANGUAGE = "OPENQASM 2.0"
    DEFAULT_BACKEND = "HQS-LT-1.0-APIVAL"
    API_HEADER_KEY = "x-api-key"

    def __init__(self, wires, machine, shots=1000, api_key=None, retry_delay=2):
        super().__init__(wires=wires, shots=shots, analytic=False)
        self.machine = machine
        self.shots = shots
        self._retry_delay = retry_delay
        self._api_key = api_key
        self.set_api_configs()
        self.data = {
            "machine": self.machine,
            "language": self.LANGUAGE,
            "priority": self.PRIORITY,
            "count": self.shots,
            "options": None,
        }
        self.reset()

    def reset(self):
        """Reset the device."""
        self._results = None
        self._samples = None

    def set_api_configs(self):
        """
        Set the configurations needed to connect to HQS API.
        """
        self._api_key = self._api_key or os.getenv("HQS_TOKEN")
        if not self._api_key:
            raise ValueError("No valid api key for HQS platform found.")
        self.header = {
            "User-Agent": "pennylane-honeywell_v{}".format(__version__),
            self.API_HEADER_KEY: self._api_key,
        }
        self.hostname = "/".join([self.BASE_HOSTNAME, self.TARGET_PATH])

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

    @property
    def operations(self):
        """Get the supported set of operations.

        Returns:
            set[str]: the set of PennyLane operation names the device supports
        """
        return set(self._operation_map.keys())

    def execute(self, circuit, **kwargs):

        self.check_validity(circuit.operations, circuit.observables)

        self._circuit_hash = circuit.hash

        circuit_str = circuit.to_openqasm()

        body = {**self.data, "program": circuit_str}

        response = requests.post(self.hostname, json.dumps(body), headers=self.header)
        response.raise_for_status()

        job_data = response.json()

        job_id = job_data["job"]
        job_endpoint = "/".join([self.hostname, job_id])

        while job_data["status"] not in self.TERMINAL_STATUSES:
            sleep(self.retry_delay)
            response = requests.get(job_endpoint, headers=self.header)
            response.raise_for_status()

            job_data = response.json()

        if job_data["status"] == "failed":
            raise DeviceError("Job failed in remote backend.")
        if job_data["status"] == "cancelled":
            # possible to get partial results back for cancelled jobs
            try:
                num_results = len(job_data["results"]["c"])
                assert num_results > 0
                if num_results < self.shots:
                    warnings.warn(
                        "Partial results returned from cancelled remote job.", RuntimeWarning
                    )
            except:
                raise DeviceError("Job was cancelled without returning any results.")

        # pylint: disable=attribute-defined-outside-init
        self._results = job_data["results"]["c"]  # list of binary strings

        # generate computational basis samples
        self._samples = self.generate_samples()

        # compute the required statistics
        results = self.statistics(circuit.observables)

        # Ensures that a combination with sample does not put
        # expvals and vars in superfluous arrays
        all_sampled = all(obs.return_type is Sample for obs in circuit.observables)
        if circuit.is_sampled and not all_sampled:
            return self._asarray(results, dtype="object")  # pragma: no cover

        return self._asarray(results)

    def generate_samples(self):
        int_values = [int(x, 2) for x in self._results]
        samples_array = np.stack(np.unravel_index(int_values, [2] * self.num_wires)).T
        return samples_array

    def apply(self, operations, **kwargs):  # pragma: no cover
        """This method is not used in the HQSDevice class."""
