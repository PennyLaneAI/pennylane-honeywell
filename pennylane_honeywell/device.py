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
import getpass
import jwt
from time import sleep
from appdirs import user_config_dir

import requests

import numpy as np

import pennylane as qml
from pennylane import QubitDevice, DeviceError
from pennylane.operation import Sample
from pennylane_honeywell.credentials import Credentials

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

pennylane_honeywell_dir = user_config_dir("pennylane-honeywell", "Xanadu")

class RequestFailedError(Exception):
    """Raised when a request to the remote platform returns an error response."""

class HQSDevice(QubitDevice):
    r"""Honeywell Quantum Services device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of wires to initialize the device with,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        machine (str): name of the Honeywell machine to execute on
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables
        user (str): The HQS API key. If not provided, the environment
            variable ``HQS_TOKEN`` is used.
        retry_delay (float): The time (in seconds) to wait between requests
            to the remote server when checking for completion of circuit
            execution.
    """
    # pylint: disable=too-many-instance-attributes
    name = "Honeywell Quantum Solutions PennyLane plugin"
    pennylane_requires = ">=0.15.0"
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
    LANGUAGE = "OPENQASM 2.0"
    DEFAULT_BACKEND = "HQS-LT-1.0-APIVAL"
    API_HEADER_KEY = "x-api-key"

    def __init__(self, wires, machine, shots=1000, user=None, retry_delay=2):
        if shots is None:
            raise ValueError(
                "The honeywell.hqs device does not support analytic expectation values"
            )

        if shots < 1 or shots > 10000:
            raise ValueError(
                "Honeywell only supports shots to be between 1 and 10,000 when running a job."
            )

        super().__init__(wires=wires, shots=shots)
        self.machine = machine
        self.shots = shots
        self._retry_delay = retry_delay

        self._user = user
        self.set_api_configs()

        self.data = {
            "machine": self.machine,
            "language": self.LANGUAGE,
            "count": self.shots,
            "options": None,
        }
        self._access_token, self._refresh_token = self.load_tokens()

        self.reset()

    def reset(self):
        """Reset the device."""
        self._results = None
        self._samples = None

    def set_api_configs(self):
        """
        Set the configurations needed to connect to HQS API.
        """
        self._user = self._user or os.getenv("HQS_USER")
        if not self._user:
            raise ValueError("No user name for HQS platform found.")

        self.cred = Credentials(user_name=self._user)

        self.hostname = "/".join([self.BASE_HOSTNAME, self.TARGET_PATH])

    @staticmethod
    def token_is_expired(token):
        token_expiry_time = jwt.decode(token, verify=False, algorithms=["RS256"])["exp"]
        current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
        return token_expiry_time < current_time

    @staticmethod
    def save_tokens(access_token, refresh_token=None):
        config = qml.default_config

        config['honeywell']['global']['access_token'] = access_token

        if refresh_token is not None:
            config['honeywell']['global']['refresh_token'] = refresh_token

        directory, _ = os.path.split(qml.default_config._filepath)

        if not os.path.isdir(directory):
            os.mkdir(directory)

        with open(qml.default_config._filepath, "w") as f:
            toml.dump(config._config, f)

    @staticmethod
    def load_tokens():
        config = qml.default_config

        try:
            access_token = config['honeywell']['global']['access_token']
            refresh_token = config['honeywell']['global']['refresh_token']
            return access_token, refresh_token

        except KeyError:
            # There aren't any tokens from before
            return None, None

    def _login(self):
        header = self.get_job_retrieval_header()

        pwd = getpass.getpass(prompt="Enter your Honeywell account password: ")
        body = {"email": self._user,
                "password": pwd}

        r = requests.post('https://qapi.honeywell.com/v1/login', json = header.update(body))

        if response.status_code == 200:
            access_token = response.json()['id-token']
            refresh_token = response.json()['refresh-token']

            # Delete the user credential
            pwd = None
            return access_token, refresh_token

        raise RequestFailedError(
            f"Failed to get access token: {self._format_error_message(response)}"
        )

    @staticmethod
    def _format_error_message():
        body = response.json()
        status = body.get("status_code", "")
        code = body.get("code", ""),
        detail = body.get("detail", ""),
        meta = body.get("meta", ""),
        return f"{status} ({code}): {detail} ({meta})"

    def _refresh_access_token(self):
        # Refresh the access token using the refresh token
        body = {"refresh-token": self._refresh_token}

        response= requests.post('https://qapi.honeywell.com/v1/login', json = body)

        # Access tokens are also called id-tokens

        if response.status_code == 200:
            return response.json()['id-token']

        raise RequestFailedError(
            f"Failed to get access token: {self._format_error_message(response)}"
        )

    def get_valid_access_token(self):
        """Return an access token.

        This method will first try to use any stored tokens (access or refresh
        token) and otherwise ask for user credentials.

        1. Check the access token:
            i) if doesn't exist or it expired then check the refresh token
            ii) otherwise: use it

        2. Check the refresh token:
            i) if doesn't exist or it expired then ask user for credentials;
            ii) otherwise: request a new access token using the refresh token

        3. Request a new access token and refresh token using the user
        credentials

        Returns:
            str: access token to use for sending requests
        """
        if self._access_token is None or self.token_is_expired(self._access_token):

            if self._refresh_token is None or self.token_is_expired(self._refresh_token):

                # TODO: pull username from config file if exists
                self._access_token, self._refresh_token = self._login()
                save_tokens(self._access_token, refresh_token=self._refresh_token)

            else:

                # Refresh the access token using the refresh token
                headers = {"Content-Type": "application/json", "refresh-token": self._refresh_token}

                r = requests.post('https://qapi.honeywell.com/v1/login', json = headers)

                # Access tokens are also called id-tokens
                self._access_token = r.json()['id-token']
                save_tokens(self._access_token)

        return self._access_token

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

    def get_job_submission_header(self):
        """Create a header required for job submission.

        Returns:
            dict: the header required for job submission
        """
        access_token = self.get_valid_access_token()
        header = {
            "Content-Type": "application/json",
            "Authorization": access_token,
        }
        return header

    def get_job_retrieval_header(self):
        """Create a header required for job retrieval.

        Returns:
            dict: the header required for job retrieval
        """
        access_token = self.get_valid_access_token()
        header = {
            "Authorization": access_token,
        }
        return header

    def _submit_circuit(self, tape):
        """Submits a circuit for execution.

        Args:
            tape (QuantumTape): the circuit to submit

        Returns:
            dict: the header required for job retrieval
        """
        circuit_str = tape.to_openqasm()
        body = {**self.data, "program": circuit_str}

        header = self.get_job_submission_header()
        return requests.post(self.hostname, json.dumps(body), headers=header)

    def _query_results(self, job_data):
        """Queries the results for a specific job.

        Args:
            job_data (str): the response obtained after submitting a job

        Returns:
            str: the response with the job results
        """
        # Extract the job ID from the response
        job_id = job_data["job"]
        job_endpoint = "/".join([self.hostname, job_id])

        while job_data["status"] not in self.TERMINAL_STATUSES:
            sleep(self.retry_delay)
            header = self.get_job_retrieval_header()
            response = requests.get(job_endpoint, headers=header)
            response.raise_for_status()

            job_data = response.json()
        return job_data

    def execute(self, tape, **kwargs):

        self.check_validity(tape.operations, tape.observables)
        response = self._submit_circuit(tape)
        response.raise_for_status()

        job_data = response.json()

        job_data = self._query_results(job_data)

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
        results = self.statistics(tape.observables)

        # Ensures that a combination with sample does not put
        # expvals and vars in superfluous arrays
        all_sampled = all(obs.return_type is Sample for obs in tape.observables)
        if tape.is_sampled and not all_sampled:
            return self._asarray(results, dtype="object")  # pragma: no cover

        return self._asarray(results)

    def generate_samples(self):
        int_values = [int(x, 2) for x in self._results]
        samples_array = np.stack(np.unravel_index(int_values, [2] * self.num_wires)).T
        return samples_array

    def apply(self, operations, **kwargs):  # pragma: no cover
        """This method is not used in the HQSDevice class."""
