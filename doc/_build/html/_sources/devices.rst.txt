Honeywell Quantum Solutions Devices
===================================

The PennyLane-HQS plugin provides the ability for PennyLane to access
devices available via Honeywell Quantum Solutions' cloud hardware service.

.. raw::html
    <section id="dev">

Cloud ion-trap hardware
-----------------------

This PennyLane device connects you to ion-trap hardware available from
Honeywell Quantum Solutions.
Once the plugin has been installed, you can use this device
directly in PennyLane by specifying ``"hqs.dev"``, where ``"dev"`` is
the name of the online hardware device you wish to access:

.. code-block:: python

    import pennylane as qml

    dev = qml.device("hqs.dev", wires=2)

    @qml.qnode(dev)
    def circuit(w, x, y, z):
        qml.RX(w, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(0.5, wires=0)
        return qml.expval(qml.PauliZ(0))


Remote backend access
---------------------

The user will need to obtain access credentials for the Honeywell Quantum
Solutions platform in order to use these remote devices.
These credentials should be provided to PennyLane via a
`configuration file or environment variable <https://pennylane.readthedocs.io/en/stable/introduction/configuration.html>`_.
Specifically, the variable ``HQS_TOKEN`` must contain a valid access key for Honeywell's online platform.
