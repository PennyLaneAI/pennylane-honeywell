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
"""Tests for the HQSDevice class"""
import os
import pytest
import appdirs
import requests
import pytest

import pennylane as qml
import numpy as np

from pennylane_honeywell.device import HQSDevice
from pennylane_honeywell import __version__

API_HEADER_KEY = "x-api-key"
BASE_HOSTNAME = "https://qapi.honeywell.com/v1"

DUMMY_MACHINE = "SOME_MACHINE_NAME"

SOME_API_KEY = "ABC123"

test_config = """\
[main]
shots = 1000

[default.gaussian]
hbar = 2

[honeywell.hqs]
shots = 99
api_key = "{}"
""".format(
    SOME_API_KEY
)

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
    def __init__(self):
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
        self.num_calls = 0

    def json(self):
        if self.num_calls == 0:
            self.num_calls = 1
            return self.mock_post_response
        else:
            return self.mock_get_response

    def raise_for_status(self):
        pass

class TestHQSDevice:
    """Tests for the HQSDevice base class."""

    @pytest.mark.parametrize("num_wires", [1, 3])
    @pytest.mark.parametrize("shots", [1, 100])
    @pytest.mark.parametrize("retry_delay", [0.1, 1.0])
    def test_default_init(self, num_wires, shots, retry_delay):
        """Tests that the device is properly initialized."""

        dev = HQSDevice(num_wires, DUMMY_MACHINE, shots, SOME_API_KEY, retry_delay)

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
        dev.job_submission_header
        #assert API_HEADER_KEY in dev.header.keys()
        #assert dev.header[API_HEADER_KEY] == SOME_API_KEY

    def test_reset(self):
        """Tests that the ``reset`` method corretly resets data."""

        dev = HQSDevice(3, shots=10, machine=DUMMY_MACHINE, api_key=SOME_API_KEY)

        dev._results = ["00"] * 10
        dev._samples = np.zeros((10, 3))
        dev.shots = 11

        dev.reset()

        assert dev._results is None
        assert dev._samples is None
        assert dev.shots == 11  # should not be reset

    def test_retry_delay(self):
        """Tests that the ``retry_delay`` property can be set manually."""

        dev = HQSDevice(3, machine=DUMMY_MACHINE, api_key=SOME_API_KEY, retry_delay=2.5)
        assert dev.retry_delay == 2.5

        dev.retry_delay = 1.0
        assert dev.retry_delay == 1.0

        with pytest.raises(qml.DeviceError, match="needs to be positive"):
            dev.retry_delay = -5

    def test_set_api_configs(self):
        """Tests that the ``set_api_configs`` method properly (re)sets the API configs."""

        dev = HQSDevice(3, machine=DUMMY_MACHINE, api_key=SOME_API_KEY)
        new_api_key = "XYZ789"
        dev._api_key = new_api_key
        dev.BASE_HOSTNAME = "https://server.someaddress.com"
        dev.TARGET_PATH = "some/path"
        dev.set_api_configs()

        assert dev.header == {
            "x-api-key": new_api_key,
            "User-Agent": "pennylane-honeywell_v{}".format(__version__),
        }
        assert dev.hostname == "https://server.someaddress.com/some/path"

    def test_api_key_not_found_error(self, monkeypatch, tmpdir):
        """Tests that an error is thrown with the device is created without
        a valid API token."""

        monkeypatch.setenv("HQS_TOKEN", "")
        monkeypatch.setenv("PENNYLANE_CONF", "")
        monkeypatch.setattr("os.curdir", tmpdir.join("folder_without_a_config_file"))

        monkeypatch.setattr(
            "pennylane.default_config", qml.Configuration("config.toml")
        )  # force loading of config
        with pytest.raises(ValueError, match="No valid api key for HQS platform found"):
            dev = HQSDevice(2, machine=DUMMY_MACHINE)

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
        dev = HQSDevice(3, machine=DUMMY_MACHINE, shots=10, api_key=SOME_API_KEY)
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
        dev = HQSDevice(2, machine=DUMMY_MACHINE, api_key=SOME_API_KEY)

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

        dev = qml.device("honeywell.hqs", wires=num_wires, machine=DUMMY_MACHINE,
                         shots=shots, api_key=SOME_API_KEY)

        assert dev.num_wires == num_wires
        assert dev.shots == shots
        assert dev.analytic == False
        assert dev.data == {
            "machine": DUMMY_MACHINE,
            "language": "OPENQASM 2.0",
            "priority": "normal",
            "count": shots,
            "options": None,
        }
        assert dev._results is None
        assert dev._samples is None
        assert dev.BASE_HOSTNAME == BASE_HOSTNAME
        assert API_HEADER_KEY in dev.header.keys()
        assert dev.header[API_HEADER_KEY] == SOME_API_KEY

    def test_api_key_not_found_error(self, monkeypatch, tmpdir):
        """Tests that an error is thrown with the device is created without
        a valid API token."""

        monkeypatch.setenv("HQS_TOKEN", "")
        monkeypatch.setenv("PENNYLANE_CONF", "")
        monkeypatch.setattr("os.curdir", tmpdir.join("folder_without_a_config_file"))

        monkeypatch.setattr(
            "pennylane.default_config", qml.Configuration("config.toml")
        )  # force loading of config
        with pytest.raises(ValueError, match="No valid api key for HQS platform found"):
            dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE)

    def test_device_gets_local_config(self, monkeypatch, tmpdir):
        """Tests that the device successfully reads a config from the local directory."""

        monkeypatch.setenv("PENNYLANE_CONF", "")
        monkeypatch.setenv("HQS_TOKEN", "")

        tmpdir.join("config.toml").write(test_config)
        monkeypatch.setattr("os.curdir", tmpdir)
        monkeypatch.setattr(
            "pennylane.default_config", qml.Configuration("config.toml")
        )  # force loading of config

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE)

        assert dev.shots == 99
        assert API_HEADER_KEY in dev.header.keys()
        assert dev.header[API_HEADER_KEY] == SOME_API_KEY

    def test_device_gets_api_key_default_config_directory(self, monkeypatch, tmpdir):
        """Tests that the device gets an api key that is stored in the default
        config directory."""
        monkeypatch.setenv("HQS_TOKEN", "")
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

        assert API_HEADER_KEY in dev.header.keys()
        assert dev.header[API_HEADER_KEY] == SOME_API_KEY

    def test_device_gets_api_key_pennylane_conf_env_var(self, monkeypatch, tmpdir):
        """Tests that the device gets an api key via the PENNYLANE_CONF
        environment variable."""
        monkeypatch.setenv("HQS_TOKEN", "")

        filepath = tmpdir.join("config.toml")
        filepath.write(test_config)
        monkeypatch.setenv("PENNYLANE_CONF", str(tmpdir))

        monkeypatch.setattr("os.curdir", tmpdir.join("folder_without_a_config_file"))
        monkeypatch.setattr(
            "pennylane.default_config", qml.Configuration("config.toml")
        )  # force loading of config

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE)

        assert API_HEADER_KEY in dev.header.keys()
        assert dev.header[API_HEADER_KEY] == SOME_API_KEY

    def test_device_gets_api_key_hqs_token_env_var(self, monkeypatch):
        """Tests that the device gets an api key that is stored in HQS_TOKEN
        environment variable."""

        NEW_API_KEY = SOME_API_KEY + "XYZ987"
        monkeypatch.setenv("PENNYLANE_CONF", "")
        monkeypatch.setenv("HQS_TOKEN", SOME_API_KEY + "XYZ987")

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE)

        assert API_HEADER_KEY in dev.header.keys()
        assert dev.header[API_HEADER_KEY] == NEW_API_KEY

    def test_executes_with_online_api(self, monkeypatch):
        """Tests that a PennyLane QNode successfully executes with a
        mocked out online API."""

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE, shots=10, retry_delay=0.01, api_key=SOME_API_KEY)

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

        circuit(0.5, 1.2)
        assert dev._results == MOCK_RESULTS

    def test_exception_failed_job(self, monkeypatch):
        """Tests that an exception is raised if the job status is `failed`."""

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE, shots=10, retry_delay=0.01, api_key=SOME_API_KEY)

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

        with pytest.raises(qml.DeviceError, match="Job failed in remote backend."):
            circuit(0.5, 1.2)

    def test_exception_cancelled_job_no_results(self, monkeypatch):
        """Tests that an exception is raised if the job status is `cancelled`
        and no partial results were returned."""

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE, shots=10, retry_delay=0.01, api_key=SOME_API_KEY)

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

        with pytest.raises(
            qml.DeviceError, match="Job was cancelled without returning any results"
        ):
            circuit(0.5, 1.2)

    def test_warning_cancelled_job_partial_results(self, monkeypatch):
        """Tests that a warning is given if the job status is `cancelled`
        and partial results were returned."""

        dev = qml.device("honeywell.hqs", wires=2, machine=DUMMY_MACHINE, shots=10, retry_delay=0.01, api_key=SOME_API_KEY)

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
            "honeywell.hqs", wires=num_wires, machine=DUMMY_MACHINE, shots=10, retry_delay=0.01, api_key=SOME_API_KEY
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
            dev = qml.device("honeywell.hqs", wires=2, machine=None, user="someuser", shots=shots)
