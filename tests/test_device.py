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
"""Tests for the HQSDevice class"""
import os
import pytest
import appdirs
import requests
import json
import datetime
import jwt
import toml
import getpass

import pennylane as qml
import numpy as np

import pennylane_honeywell
from pennylane_honeywell.device import HQSDevice, InvalidJWTError, RequestFailedError, ExpiredRefreshTokenError
from pennylane_honeywell import __version__

API_HEADER_KEY = "x-api-key"
BASE_HOSTNAME = "https://qapi.honeywell.com/v1"

DUMMY_MACHINE = "SOME_MACHINE_NAME"

DUMMY_EMAIL = "ABC123"

test_config = """\
[main]
shots = 1000

[default.gaussian]
hbar = 2

[honeywell.hqs]
shots = 99
user_email = "{}"
""".format(
    DUMMY_EMAIL
)


def get_example_tape_with_qasm():
    with qml.tape.QuantumTape() as tape:
        qml.RY(0.2, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliY(0))

    tape_openqasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg '
        "c[2];\nry(0.2) q[0];\ncx q[1],q[0];\nz q[1];\ns q[1];\nh q[1];\nmeasure q[0] -> "
        "c[0];\nmeasure q[1] -> c[1];\n"
    )

    return tape, tape_openqasm


MOCK_RESULTS = ["00", "00", "00", "00", "00", "00", "00", "00", "00", "00"]

REF_RESULTS_01 = ["01"] * 10
REF_RESULTS_10 = ["10"] * 10
REF_RESULTS_001 = ["001"] * 10
REF_RESULTS_010 = ["010"] * 10
REF_RESULTS_011 = ["011"] * 10
REF_RESULTS_100 = ["100"] * 10
REF_RESULTS_101 = ["101"] * 10
REF_RESULTS_110 = ["110"] * 10


class MockResponse:
    def __init__(self, num_calls=0):
        self.status_code = 200
        self.mock_post_response = {
            "job": "bf668869b6b74909a7e1fad2d7a0f932",
            "status": "queued",
        }
        self.mock_get_response = {
            "job": "bf668869b6b74909a7e1fad2d7a0f932",
            "name": "bf668869b6b74909a7e1fad2d7a0f932",
            "status": "completed",
            "cost": "19250",
            "start-date": "2020-06-05T16:30:40.458045",
            "end-date": "2020-06-05T16:30:40.458971",
            "submit-date": "2020-06-05T16:30:40.150961",
            "result-date": "2020-06-05T16:30:41.000031",
            "results": {"c": ["00"] * 10},
        }
        self.num_calls = num_calls

    def json(self):
        if self.num_calls == 0:
            self.num_calls = 1
            return self.mock_post_response
        else:
            return self.mock_get_response

    def raise_for_status(self):
        pass


MOCK_ACCESS_TOKEN = "123456789"
MOCK_REFRESH_TOKEN = "11111111"


class MockResponseWithTokens:
    def __init__(self, num_calls=0):
        self.status_code = 200
        self.mock_post_response = {
            "id-token": MOCK_ACCESS_TOKEN,
            "refresh-token": MOCK_REFRESH_TOKEN,
        }
        self.num_calls = num_calls

    def json(self):
        return self.mock_post_response


class MockResponseUnsuccessfulRequest:
    def __init__(self):

        self.status_code = "Not 200"
        self.mock_post_response = {
            "status_code": "Not 200",
            "code": "Not 200",
            "detail": "Mock error for login.",
            "meta": "Something went wrong.",
        }

    def json(self):
        return self.mock_post_response

now = datetime.datetime.now()


class TestHQSDevice:
    """Tests for the HQSDevice base class."""

    @pytest.mark.parametrize("num_wires", [1, 3])
    @pytest.mark.parametrize("shots", [1, 100])
    @pytest.mark.parametrize("retry_delay", [0.1, 1.0])
    def test_default_init(self, num_wires, shots, retry_delay):
        """Tests that the device is properly initialized."""

        dev = HQSDevice(
            num_wires, DUMMY_MACHINE, shots, user_email=DUMMY_EMAIL, retry_delay=retry_delay
        )

        assert dev.num_wires == num_wires
        assert dev.shots == shots
        assert dev.retry_delay == retry_delay
        assert dev.analytic == False
        assert dev.data == {
            "machine": DUMMY_MACHINE,
            "language": "OPENQASM 2.0",
            "count": shots,
            "options": None,
        }
        assert dev._results is None
        assert dev._samples is None
        assert dev.BASE_HOSTNAME == BASE_HOSTNAME
        assert dev._user == DUMMY_EMAIL

    def test_reset(self):
        """Tests that the ``reset`` method corretly resets data."""

        dev = HQSDevice(3, shots=10, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)

        dev._results = ["00"] * 10
        dev._samples = np.zeros((10, 3))
        dev.shots = 11

        dev.reset()

        assert dev._results is None
        assert dev._samples is None
        assert dev.shots == 11  # should not be reset

    def test_retry_delay(self):
        """Tests that the ``retry_delay`` property can be set manually."""

        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL, retry_delay=2.5)
        assert dev.retry_delay == 2.5

        dev.retry_delay = 1.0
        assert dev.retry_delay == 1.0

        with pytest.raises(qml.DeviceError, match="needs to be positive"):
            dev.retry_delay = -5

    def test_set_api_configs(self):
        """Tests that the ``set_api_configs`` method properly (re)sets the API configs."""

        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)
        new_user = "XYZ789"
        dev._user = new_user
        dev.BASE_HOSTNAME = "https://server.someaddress.com"
        dev.TARGET_PATH = "some/path"
        dev.set_api_configs()

        assert dev.hostname == "https://server.someaddress.com/some/path"
        assert dev._user == new_user

    def test_get_job_submission_header(self, monkeypatch):
        """Tests that the ``get_job_submission_header`` method properly returns
        the correct header."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)
        SOME_ACCESS_TOKEN = "XYZ789"
        monkeypatch.setattr(dev, "get_valid_access_token", lambda: SOME_ACCESS_TOKEN)

        expected = {
            "Content-Type": "application/json",
            "Authorization": SOME_ACCESS_TOKEN,
        }
        assert dev.get_job_submission_header() == expected

    def test_get_job_retrieval_header(self, monkeypatch):
        """Tests that the ``get_job_retrieval_header`` method properly returns
        the correct header."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)
        SOME_ACCESS_TOKEN = "XYZ789"
        monkeypatch.setattr(dev, "get_valid_access_token", lambda: SOME_ACCESS_TOKEN)

        expected = {
            "Authorization": SOME_ACCESS_TOKEN,
        }
        assert dev.get_job_retrieval_header() == expected

    def test_submit_circuit_method(self, monkeypatch):
        """Tests that the ``_submit_circuit`` method sends a request adhering
        to the Honeywell API specs."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)
        SOME_ACCESS_TOKEN = "XYZ789"
        monkeypatch.setattr(dev, "get_valid_access_token", lambda: SOME_ACCESS_TOKEN)

        call_history = []
        monkeypatch.setattr(
            requests,
            "post",
            lambda hostname, body, headers: call_history.append(tuple([hostname, body, headers])),
        )

        tape, tape_openqasm = get_example_tape_with_qasm()

        expected_data = {
            "machine": DUMMY_MACHINE,
            "language": dev.LANGUAGE,
            "count": dev.shots,
            "options": None,
        }

        expected_body = {**expected_data, "program": tape_openqasm}
        expected_header = {
            "Content-Type": "application/json",
            "Authorization": SOME_ACCESS_TOKEN,
        }
        dev._submit_circuit(tape)

        assert len(call_history) == 1
        hostname, body, headers = call_history[0]
        assert hostname == dev.hostname
        assert body == json.dumps(expected_body)
        assert headers == expected_header

    def test_login(self, monkeypatch):
        """Tests that an access token and a refresh token are returned if the
        _login method was successful."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)

        mock_response = MockResponseWithTokens()
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(getpass, "getpass", lambda *args, **kwargs: None)

        access_token, refresh_token = dev._login()
        assert access_token == MOCK_ACCESS_TOKEN
        assert refresh_token == MOCK_REFRESH_TOKEN

    def test_login_raises(self, monkeypatch):
        """Tests that an error is raised if the _login method was
        unsuccessful."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)

        mock_response = MockResponseUnsuccessfulRequest()
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(getpass, "getpass", lambda *args, **kwargs: None)

        with pytest.raises(RequestFailedError, match="Failed to get access token"):
            dev._login()

    def test_refresh_access_token(self, monkeypatch):
        """Tests that _refresh_access_token returns an access token for a
        successful request."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)

        mock_response = MockResponseWithTokens()
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

        access_token = dev._refresh_access_token()
        assert access_token == MOCK_ACCESS_TOKEN

    @pytest.mark.parametrize("code", [403, 400])
    def test_refresh_access_token_raises_for_expired(self, monkeypatch, code):
        """Tests that _refresh_access_token raises an error for an
        expired refresh token."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)

        class MockResponseForExpired:
            def __init__(self):

                self.status_code = code
                self.mock_post_response = {
                    "status_code": str(code),
                    "code": "Not 200",
                    "detail": "Mock error for refresh.",
                    "meta": "Something went wrong.",
                }

            def json(self):
                return self.mock_post_response

        mock_response = MockResponseForExpired()
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

        with pytest.raises(ExpiredRefreshTokenError, match="Invalid refresh token was used."):
            dev._refresh_access_token()

    def test_refresh_access_token_raises(self, monkeypatch):
        """Tests that _refresh_access_token raises an error for a
        unsuccessful request."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)

        mock_response = MockResponseUnsuccessfulRequest()
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)

        with pytest.raises(RequestFailedError, match="Failed to get access token"):
            dev._refresh_access_token()

    def test_get_valid_access_token_use_stored(self):
        """Test that the get_valid_access_token uses a stored token if it
        exists and it's not expired."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)
        valid_time = now.replace(now.year + 1)
        token = jwt.encode({"exp": valid_time}, "secret")
        dev._access_token = token
        assert dev.get_valid_access_token() == token

    @pytest.mark.parametrize("access_token_expiry", [0, None])
    @pytest.mark.parametrize("refresh_token_expiry", [0, None])
    def test_get_valid_access_token_new_tokens(
        self, access_token_expiry, refresh_token_expiry, monkeypatch
    ):
        """Test that the get_valid_access_token returns a new access and
        refresh tokens by logging in."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)

        if access_token_expiry:
            # Set the token to an outdated token
            dev._access_token = jwt.encode({"exp": access_token_expiry}, "secret")

        if refresh_token_expiry:
            # Set the token to an outdated token
            dev._refresh_token = jwt.encode({"exp": refresh_token_expiry}, "secret")

        some_token = 1234567
        some_refresh_token = 111111
        monkeypatch.setattr(dev, "_login", lambda *args, **kwargs: (some_token, some_refresh_token))
        monkeypatch.setattr(dev, "save_tokens", lambda *args, **kwargs: None)
        assert dev.get_valid_access_token() == some_token
        assert dev._refresh_token == some_refresh_token

    @pytest.mark.parametrize("access_token_expiry", [0, None])
    def test_get_valid_access_token_using_refresh_token(self, access_token_expiry, monkeypatch):
        """Test that the get_valid_access_token returns a new access token by
        refreshing using the refresh token."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)

        if access_token_expiry:
            # Set the token to an outdated token
            dev._access_token = jwt.encode({"exp": access_token_expiry}, "secret")

        # Set a refresh token with an expiry date in the future
        dev._refresh_token = jwt.encode({"exp": now.replace(now.year + 1)}, "secret")
        mock_response = MockResponseWithTokens()
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(dev, "save_tokens", lambda *args, **kwargs: None)

        assert dev.get_valid_access_token() == MOCK_ACCESS_TOKEN

    @pytest.mark.parametrize("access_token_expiry", [0, None])
    def test_get_valid_access_token_using_refresh_token_raises(
        self, access_token_expiry, monkeypatch
    ):
        """Test that the get_valid_access_token returns a new access token by
        refreshing using the refresh token."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)

        if access_token_expiry:
            # Set the token to an outdated token
            dev._access_token = jwt.encode({"exp": access_token_expiry}, "secret")

        # Set a refresh token with an expiry date in the future
        dev._refresh_token = jwt.encode({"exp": now.replace(now.year + 1)}, "secret")
        mock_response = MockResponseUnsuccessfulRequest()
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(dev, "save_tokens", lambda *args, **kwargs: None)

        with pytest.raises(RequestFailedError, match="Failed to get access token"):
            dev.get_valid_access_token()

    def test_query_results(self, monkeypatch):
        """Tests that the ``_query_results`` method sends a request adhering to
        the Honeywell API specs."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL, retry_delay=0.1)
        SOME_ACCESS_TOKEN = "XYZ789"
        monkeypatch.setattr(dev, "get_valid_access_token", lambda: SOME_ACCESS_TOKEN)

        # set num_calls=1 as the job was already submitted in cases when we get
        # the result
        mock_response = MockResponse(num_calls=1)

        call_history = []

        def wrapper(job_endpoint, headers):
            call_history.append(tuple([job_endpoint, headers]))
            return mock_response

        monkeypatch.setattr(requests, "get", wrapper)

        SOME_JOB_ID = "JOB123"
        mock_job_data = {"job": SOME_JOB_ID, "status": "not completed!"}
        res = dev._query_results(mock_job_data)

        expected_header = {
            "Authorization": SOME_ACCESS_TOKEN,
        }

        assert len(call_history) == 1
        job_endpoint, headers = call_history[0]
        assert job_endpoint == "/".join([dev.hostname, SOME_JOB_ID])
        assert headers == expected_header

    def test_query_results_expected_response(self, monkeypatch):
        """Tests that using the ``_query_results`` method an expected response
        is gathered."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL, retry_delay=0.01)
        SOME_ACCESS_TOKEN = "XYZ789"
        monkeypatch.setattr(dev, "get_valid_access_token", lambda: SOME_ACCESS_TOKEN)

        mock_response = MockResponse()
        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_response)

        SOME_JOB_ID = "JOB123"
        mock_job_data = {"job": SOME_JOB_ID, "status": "not completed!"}
        res = dev._query_results(mock_job_data)

        assert res == mock_response.mock_get_response

    def test_user_not_found_error(self, monkeypatch, tmpdir):
        """Tests that an error is thrown with the device is created without
        a valid API token."""

        monkeypatch.setenv("HQS_USER", "")
        monkeypatch.setenv("PENNYLANE_CONF", "")
        monkeypatch.setattr("os.curdir", tmpdir.join("folder_without_a_config_file"))

        monkeypatch.setattr(
            "pennylane.default_config", qml.Configuration("config.toml")
        )  # force loading of config
        with pytest.raises(ValueError, match="No username for HQS platform found"):
            HQSDevice(2, machine=DUMMY_MACHINE)._login()

    @pytest.mark.parametrize("token, expired", [(0, True), (now.replace(now.year + 1), False)])
    def test_token_is_expired(self, token, expired):
        """Tests that the token_is_expired method results in expected
        values."""
        token = jwt.encode({"exp": token}, "secret")

        assert HQSDevice(2, machine=DUMMY_MACHINE).token_is_expired(token) is expired

    def test_token_is_expired_raises(self):
        """Tests that the token_is_expired method raises an error for invalid
        JWT token."""
        with pytest.raises(InvalidJWTError, match="Invalid JWT token"):
            HQSDevice(2, machine=DUMMY_MACHINE).token_is_expired(Exception)

    @pytest.mark.parametrize("tokens", [[12345], [12345, 5432123]])
    @pytest.mark.parametrize("new_dir", [True, False])
    def test_save_tokens(self, monkeypatch, tmpdir, tokens, new_dir):
        """Tests that the save_tokens method correctly saves to the PennyLane
        configuration file."""
        mock_config = qml.Configuration("config.toml")

        if new_dir:
            # Case when the target directory doesn't exist
            filepath = tmpdir.join("new_dir").join("config.toml")
        else:
            filepath = tmpdir.join("config.toml")
        mock_config._filepath = filepath

        monkeypatch.setattr(qml, "default_config", mock_config)
        HQSDevice(2, machine=DUMMY_MACHINE).save_tokens(*tokens)

        with open(filepath) as f:
            configuration_file = toml.load(f)

        assert configuration_file["honeywell"]["global"]["access_token"] == tokens[0]

        if len(tokens) > 1:
            assert configuration_file["honeywell"]["global"]["refresh_token"] == tokens[1]

    @pytest.mark.parametrize("tokens", [[12345], [12345, 5432123]])
    @pytest.mark.parametrize("new_dir", [True, False])
    def test_save_tokens_no_config_found(self, monkeypatch, tmpdir, tokens, new_dir):
        """Tests that the save_tokens method correctly defaults to the user
        config directory when no configuration file exists."""
        config_file_name = "config.toml"
        mock_config = qml.Configuration(config_file_name)
        if new_dir:
            # Case when the target directory doesn't exist
            directory = tmpdir.join("new_dir")
        else:
            directory = tmpdir

        filepath = directory.join(config_file_name)
        mock_config._user_config_dir = directory

        # Only the filename is in the filepath: just like when no config file
        # was found
        mock_config._filepath = config_file_name

        monkeypatch.setattr(qml, "default_config", mock_config)

        HQSDevice(2, machine=DUMMY_MACHINE).save_tokens(*tokens)

        with open(filepath) as f:
            configuration_file = toml.load(f)

        assert configuration_file["honeywell"]["global"]["access_token"] == tokens[0]

        if len(tokens) > 1:
            assert configuration_file["honeywell"]["global"]["refresh_token"] == tokens[1]

    @pytest.mark.parametrize(
        "results, indices",
        [
            # TODO: confirm this encoding
            (["000"] * 10, [0, 0, 0]),
            (["001"] * 10, [0, 0, 1]),
            (["010"] * 10, [0, 1, 0]),
            (["011"] * 10, [0, 1, 1]),
            (["100"] * 10, [1, 0, 0]),
            (["101"] * 10, [1, 0, 1]),
            (["110"] * 10, [1, 1, 0]),
            (["111"] * 10, [1, 1, 1]),
        ],
    )
    def test_generate_samples(self, results, indices):
        """Tests that the generate_samples function of HQSDevice provides samples in
        the correct format expected by PennyLane."""
        dev = HQSDevice(3, machine=DUMMY_MACHINE, shots=10, user_email=DUMMY_EMAIL)
        dev._results = results
        res = dev.generate_samples()
        expected_array = np.stack([np.ravel(indices)] * 10)
        assert res.shape == (dev.shots, dev.num_wires)
        assert np.all(res == expected_array)


class TestHQSDeviceIntegration:
    """Integration tests of HQSDevice base class with PennyLane"""

    def test_invalid_op_exception(self):
        """Tests whether an exception is raised if the circuit is
        passed an unsupported operation."""
        dev = HQSDevice(2, machine=DUMMY_MACHINE, user_email=DUMMY_EMAIL)

        class DummyOp(qml.operation.Operation):
            num_params = 0
            num_wires = 1
            par_domain = None

        @qml.qnode(dev)
        def circuit():
            DummyOp(wires=[0])
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(qml.DeviceError, match="Gate DummyOp not supported"):
            circuit()

    @pytest.mark.parametrize("num_wires", [1, 3])
    @pytest.mark.parametrize("shots", [1, 200])
    def test_load_from_device_function(self, num_wires, shots):
        """Tests that the HQSDevice can be loaded from PennyLane `device` function."""

        dev = qml.device(
            "honeywell.hqs",
            wires=num_wires,
            machine=DUMMY_MACHINE,
            shots=shots,
            user_email=DUMMY_EMAIL,
        )

        assert dev.num_wires == num_wires
        assert dev.shots == shots
        assert dev.analytic == False
        assert dev.data == {
            "machine": DUMMY_MACHINE,
            "language": "OPENQASM 2.0",
            "count": shots,
            "options": None,
        }
        assert dev._results is None
        assert dev._samples is None
        assert dev.BASE_HOSTNAME == BASE_HOSTNAME
        assert dev._user == DUMMY_EMAIL

    def test_user_not_found_error_login(self, monkeypatch, tmpdir):
        """Tests that an error is thrown with the device if no user name was
        specified before a login."""

        monkeypatch.setenv("HQS_USER", "")
        monkeypatch.setenv("PENNYLANE_CONF", "")
        monkeypatch.setattr("os.curdir", tmpdir.join("folder_without_a_config_file"))

        monkeypatch.setattr(
            "pennylane.default_config", qml.Configuration("config.toml")
        )  # force loading of config
        with pytest.raises(ValueError, match="No username for HQS platform found"):
            qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE)._login()

    def test_device_gets_local_config(self, monkeypatch, tmpdir):
        """Tests that the device successfully reads a config from the local directory."""

        monkeypatch.setenv("PENNYLANE_CONF", "")
        monkeypatch.setenv("HQS_USER", "")

        tmpdir.join("config.toml").write(test_config)
        monkeypatch.setattr("os.curdir", tmpdir)
        monkeypatch.setattr(
            "pennylane.default_config", qml.Configuration("config.toml")
        )  # force loading of config

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE)

        assert dev.shots == 99
        assert dev._user == DUMMY_EMAIL

    def test_device_gets_email_default_config_directory(self, monkeypatch, tmpdir):
        """Tests that the device gets a user email that is stored in the default
        config directory."""
        monkeypatch.setenv("HQS_USER", "")
        monkeypatch.setenv("PENNYLANE_CONF", "")

        config_dir = tmpdir.mkdir("pennylane")  # fake default config directory
        config_dir.join("config.toml").write(test_config)
        monkeypatch.setenv(
            "XDG_CONFIG_HOME", os.path.expanduser(tmpdir)
        )  # HACK: only works on linux

        monkeypatch.setattr("os.curdir", tmpdir.join("folder_without_a_config_file"))

        c = qml.Configuration("config.toml")
        monkeypatch.setattr("pennylane.default_config", c)  # force loading of config

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE)

        assert dev._user == DUMMY_EMAIL

    def test_device_gets_email_pennylane_conf_env_var(self, monkeypatch, tmpdir):
        """Tests that the device gets an email via the PENNYLANE_CONF
        environment variable."""
        monkeypatch.setenv("HQS_USER", "")

        filepath = tmpdir.join("config.toml")
        filepath.write(test_config)
        monkeypatch.setenv("PENNYLANE_CONF", str(tmpdir))

        monkeypatch.setattr("os.curdir", tmpdir.join("folder_without_a_config_file"))
        monkeypatch.setattr(
            "pennylane.default_config", qml.Configuration("config.toml")
        )  # force loading of config

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE)

        assert dev._user == DUMMY_EMAIL

    def test_device_gets_email_hqs_token_env_var(self, monkeypatch):
        """Tests that the device gets an email that is stored in HQS_USER
        environment variable."""

        NEW_EMAIL = DUMMY_EMAIL + "XYZ987"
        monkeypatch.setenv("PENNYLANE_CONF", "")
        monkeypatch.setenv("HQS_USER", DUMMY_EMAIL + "XYZ987")

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE)

        assert dev._user == NEW_EMAIL

    def test_executes_with_online_api(self, monkeypatch):
        """Tests that a PennyLane QNode successfully executes with a
        mocked out online API."""

        dev = qml.device(
            "honeywell.hqs",
            wires=2,
            machine=DUMMY_MACHINE,
            shots=10,
            retry_delay=0.01,
            user_email=DUMMY_EMAIL,
        )

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(0.5, wires=0)
            return qml.expval(qml.PauliY(0))

        mock_response = MockResponse()
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(requests.Session, "request", lambda *args, **kwargs: mock_response)

        monkeypatch.setattr(dev, "get_valid_access_token", lambda *args, **kwargs: None)

        circuit(0.5, 1.2)
        assert dev._results == MOCK_RESULTS

    def test_exception_failed_job(self, monkeypatch):
        """Tests that an exception is raised if the job status is `failed`."""

        dev = qml.device(
            "honeywell.hqs",
            wires=2,
            machine=DUMMY_MACHINE,
            shots=10,
            retry_delay=0.01,
            user_email=DUMMY_EMAIL,
        )

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(0.5, wires=0)
            return qml.expval(qml.PauliY(0))

        mock_response = MockResponse()
        mock_response.mock_get_response["status"] = "failed"

        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(requests.Session, "request", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(dev, "get_valid_access_token", lambda *args, **kwargs: None)

        with pytest.raises(qml.DeviceError, match="Job failed in remote backend."):
            circuit(0.5, 1.2)

    def test_exception_cancelled_job_no_results(self, monkeypatch):
        """Tests that an exception is raised if the job status is `cancelled`
        and no partial results were returned."""

        dev = qml.device(
            "honeywell.hqs",
            wires=2,
            machine=DUMMY_MACHINE,
            shots=10,
            retry_delay=0.01,
            user_email=DUMMY_EMAIL,
        )

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(0.5, wires=0)
            return qml.expval(qml.PauliY(0))

        mock_response = MockResponse()
        mock_response.mock_get_response["status"] = "cancelled"
        mock_response.mock_get_response["results"] = None

        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(requests.Session, "request", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(dev, "get_valid_access_token", lambda *args, **kwargs: None)

        with pytest.raises(
            qml.DeviceError, match="Job was cancelled without returning any results"
        ):
            circuit(0.5, 1.2)

    def test_warning_cancelled_job_partial_results(self, monkeypatch):
        """Tests that a warning is given if the job status is `cancelled`
        and partial results were returned."""

        dev = qml.device(
            "honeywell.hqs",
            wires=2,
            machine=DUMMY_MACHINE,
            shots=10,
            retry_delay=0.01,
            user_email=DUMMY_EMAIL,
        )

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(0.5, wires=0)
            return qml.expval(qml.PauliY(0))

        mock_response = MockResponse()
        mock_response.mock_get_response["status"] = "cancelled"
        mock_response.mock_get_response["results"] = {"c": ["00"] * 3}

        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(requests.Session, "request", lambda *args, **kwargs: mock_response)

        monkeypatch.setattr(dev, "get_valid_access_token", lambda *args, **kwargs: None)

        with pytest.warns(
            RuntimeWarning, match="Partial results returned from cancelled remote job."
        ):
            circuit(0.5, 1.2)

        assert dev._results == ["00"] * 3

    @pytest.mark.parametrize(
        "wire_flip_idx, ref_result",
        [
            ([1, 0], REF_RESULTS_10),
            ([0, 1], REF_RESULTS_01),
            ([1, 0, 0], REF_RESULTS_100),
            ([1, 1, 0], REF_RESULTS_110),
            ([1, 0, 1], REF_RESULTS_101),
            ([0, 1, 0], REF_RESULTS_010),
            ([0, 0, 1], REF_RESULTS_001),
            ([0, 1, 1], REF_RESULTS_011),
        ],
    )
    def test_reference_results_correct_expval(self, wire_flip_idx, ref_result, monkeypatch):
        """Tests that a simple circuit with a known specific result from the platform leads to the proper
        expectation value in PennyLane."""
        num_wires = len(wire_flip_idx)
        dev = qml.device(
            "honeywell.hqs",
            wires=num_wires,
            machine=DUMMY_MACHINE,
            shots=10,
            retry_delay=0.01,
            user_email=DUMMY_EMAIL,
        )

        # bit flip circuit
        @qml.qnode(dev)
        def circuit():
            for w in wire_flip_idx:
                if w:
                    qml.PauliX(w)
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        mock_response = MockResponse()
        mock_response.mock_get_response["results"] = {"c": ref_result}
        monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(requests.Session, "request", lambda *args, **kwargs: mock_response)
        monkeypatch.setattr(dev, "get_valid_access_token", lambda *args, **kwargs: None)

        res = circuit()
        expected = (-1) ** np.array(wire_flip_idx)
        assert np.all(expected == res)

    def test_analytic_error(self):
        """Test that instantiating the device with `shots=None` results in an error"""
        with pytest.raises(ValueError, match="does not support analytic"):
            dev = qml.device("honeywell.hqs", wires=2, machine=None, shots=None)

    @pytest.mark.parametrize("shots", [-1, 100000])
    def test_incorrect_shots(self, shots):
        """Test that instantiating the device with incorrect number of shots
        results in an error"""
        with pytest.raises(ValueError, match="Honeywell only supports shots to be between"):
            dev = qml.device(
                "honeywell.hqs", wires=2, machine=None, user_email="someuser", shots=shots
            )

    @pytest.mark.skip(reason="no credentials are being specified for testing here")
    def test_connect(self):
        """Test running a circuit by connecting to HQS."""
        email = "<Enter email here>"
        dev = qml.device('honeywell.hqs', user_email=email, machine="HQS-LT-S1-APIVAL", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RY(0.3, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert circuit() == 1
