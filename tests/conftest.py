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
import numpy as np
import pytest
import pennylane as qml

np.random.seed(42)

@pytest.fixture(params=[False, True])
def tape_mode(request, mocker):
    """Tests using this fixture will be run twice, once in tape mode and once without."""

    if request.param:
        # Several attributes and methods on the old QNode have a new location on the new QNode/tape.
        # Here, we dynamically mock so that the tests do not have to be modified to support both
        # tape and non-tape mode. Once tape mode is default, we can make the equivalent
        # changes directly in the tests.
        mocker.patch("pennylane.tape.QNode.ops", property(lambda self: self.qtape.operations + self.qtape.observables), create=True)
        mocker.patch("pennylane.tape.QNode.h", property(lambda self: self.diff_options["h"]), create=True)
        mocker.patch("pennylane.tape.QNode.order", property(lambda self: self.diff_options["order"]), create=True)
        mocker.patch("pennylane.tape.QNode.circuit", property(lambda self: self.qtape.graph), create=True)

        def patched_jacobian(self, args, **kwargs):
            method = kwargs.get("method", "best")

            if method == "A":
                method = "analytic"
            elif method == "F":
                method = "numeric"

            kwargs["method"] = method
            dev = kwargs["options"]["device"]

            return self.qtape.jacobian(dev, **kwargs)


        mocker.patch("pennylane.tape.QNode.jacobian", patched_jacobian, create=True)

    else:
        qml.disable_tape()

    yield

    qml.enable_tape()
