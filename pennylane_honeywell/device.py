# Copyright 2020-2021 Xanadu Quantum Technologies Inc.

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
import datetime
import getpass
import json
import os
import warnings
from time import sleep

import jwt
import numpy as np
import pennylane as qml
import requests
import toml
from pennylane import DeviceError, QubitDevice
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


class RequestFailedError(Exception):
    """Raised when a request to the remote platform returns an error response."""


class InvalidJWTError(Exception):
    """Raised when the returned JWT token is invalid."""


class ExpiredRefreshTokenError(Exception):
    """Raised when a refresh token used to get a new access token is
    expired."""


class HQSDevice(QubitDevice):
    r"""Honeywell Quantum Services device for PennyLane.

    Args:
        wires (int or Iterable[Number, str]]): Number of wires to initialize the device with,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``).
        machine (str): name of the Honeywell machine to execute on
        shots (int): number of circuit evaluations/random samples used
            to estimate expectation values of observables
        user_email (str): The user email used to authenticate to HQS. If not
            provided, the environment variable ``HQS_USER`` or the ``user_email``
            information from the PennyLane configuration file is used.
        access_token (str): The access token to use when authenticating to HQS.
            Note: an access token may have been saved to the config file
            automatically by the ``save_tokens`` method.
        refresh_token (str): The refresh token to use for obtaining a new
            access token when authenticating to HQS. Note: this argument may be parsed
            from the PennyLane configuration file. A refresh token may have
            been saved to the config file automatically by the ``save_tokens`` method.
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

    def __init__(
        self,
        wires,
        machine,
        shots=1000,
        user_email=None,
        access_token=None,
        refresh_token=None,
        retry_delay=2,
    ):
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

        self._user = user_email
        self.set_api_configs()

        self.data = {
            "machine": self.machine,
            "language": self.LANGUAGE,
            "count": self.shots,
            "options": None,
        }
        self._access_token = access_token
        self._refresh_token = refresh_token

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
            # Check the PennyLane config file too
            config = qml.default_config
            config.safe_get(config._config, *["honeywell", "global", "user_email"])

        self.hostname = "/".join([self.BASE_HOSTNAME, self.TARGET_PATH])

    @staticmethod
    def token_is_expired(token):
        """Check whether a given token is expired.

        Args:
            token (str): A token to check, could be an access token or a
                refresh token. The token is decoded using JWT to check its
                expiry date.

        Returns:
            bool: whether the token is expired or not
        """
        try:
            token_expiry_time = jwt.decode(
                token, options={"verify_signature": False}, algorithms=["RS256"]
            )["exp"]
        except jwt.DecodeError:
            # Some error happened: the token is invalid
            raise InvalidJWTError("Invalid JWT token received.")

        current_time = datetime.datetime.now(datetime.timezone.utc).timestamp()
        return token_expiry_time < current_time

    @staticmethod
    def save_tokens(access_token, refresh_token=None):
        """Save tokens to the PennyLane configuration file.

        Args:
            access_token (str): access token to save
            refresh_token (str): refresh token to save (if any)
        """
        config = qml.default_config

        config.safe_set(config._config, access_token, *["honeywell", "global", "access_token"])

        if refresh_token is not None:
            config.safe_set(
                config._config, refresh_token, *["honeywell", "global", "refresh_token"]
            )

        directory, _ = os.path.split(qml.default_config._filepath)

        if not directory:
            directory = config._user_config_dir
            config._filepath = os.path.join(directory, config._name)

        if not os.path.isdir(directory):
            os.mkdir(directory)

        with open(qml.default_config._filepath, "w") as file_to_write:
            toml.dump(config._config, file_to_write)

    def _login(self):
        """Login to the HQS service to obtain an access token and a refresh token.

        Returns:
            tuple: a valid access token and a refresh token

        Raises:
            ValueError: if no username was provided
            RequestFailedError: if the request failed
        """

        if not self._user:
            raise ValueError("No username for HQS platform found when trying to login.")

        pwd = getpass.getpass(prompt="Enter your Honeywell account password: ")
        body = {"Content-Type": "application/json", "email": self._user, "password": pwd}

        response = requests.post("https://qapi.honeywell.com/v1/login", json=body)

        if response.status_code == 200:
            response_json = response.json()
            access_token = response_json["id-token"]
            refresh_token = response_json["refresh-token"]

            # Delete the user credential
            del pwd
            return access_token, refresh_token

        raise RequestFailedError(
            f"Failed to get access token: {self._format_error_message(response)}"
        )

    @staticmethod
    def _format_error_message(response):
        """Formats an error message of an HTTP response.

        Args:
            response: the HTTP response to format

        Returns:
            str: a formatted version of the response
        """
        request_code = getattr(response, "status_code", "")
        reason = getattr(response, "reason", "")
        response_text = getattr(response, "text", "{}")
        text = json.loads(response_text)
        error_code = text.get("error", {}).get("code", "")
        error_reason = text.get("error", {}).get("text", "")
        return (
            f"Request {reason} with code {request_code}. \n"
            f"Error code {error_code} with reason: {error_reason}"
        )

    def _refresh_access_token(self):
        """Sends a request to refresh the access token using the stored refresh
        token.

        Returns:
            str: a valid access token

        Raises:
            RequestFailedError: if the request failed
        """
        # Refresh the access token using the refresh token
        body = {"Content-Type": "application/json", "refresh-token": self._refresh_token}

        response = requests.post("https://qapi.honeywell.com/v1/login", json=body)

        # Access tokens are also called id-tokens

        if response.status_code == 200:
            return response.json()["id-token"]

        if response.status_code in (400, 403):
            raise ExpiredRefreshTokenError("Invalid refresh token was used.")

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

        Raises:
            RequestFailedError: if the request failed
        """
        if self._access_token is None or self.token_is_expired(self._access_token):
            if self._refresh_token is not None:
                try:
                    self._access_token = self._refresh_access_token()
                    self.save_tokens(self._access_token)
                except ExpiredRefreshTokenError:
                    self._refresh_token = None

            if self._refresh_token is None:
                self._access_token, self._refresh_token = self._login()
                self.save_tokens(self._access_token, refresh_token=self._refresh_token)

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
        """Queries the results of a specific job.

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
