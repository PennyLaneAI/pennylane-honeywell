PennyLane-Honeywell Plugin
##########################

:Release: |release|

.. include:: ../README.rst
  :start-after:	header-start-inclusion-marker-do-not-remove
  :end-before: header-end-inclusion-marker-do-not-remove

Once the PennyLane-Honeywell plugin is installed, the Honeywell Quantum Solutions online devices can be
accessed straight away in PennyLane, without the need to import any additional packages.

Devices
=======

PennyLane-Honeywell provides Honeywell Quantum Solutions hardware devices for PennyLane:

.. devicegalleryitem::
    :name: 'honeywell.hqs'
    :description: Honeywell Quantum Solutions cloud ion-trap hardware.
    :link: devices.html#hqs

.. raw:: html

    <div style='clear:both'></div>
    </br>

Remote backend access
=====================


The user will need to obtain access credentials for the Honeywell Quantum
Solutions platform in order to use these remote devices.
These credentials should be provided to PennyLane via a `configuration file,
device argument or environment variable
<https://pennylane.readthedocs.io/en/stable/introduction/configuration.html>`_.
Specifically, the variable ``HQS_USER`` or the ``user_email`` device argument
must contain a valid email address for Honeywell's online platform. Upon a
first time log-in and after 30 days of a successful log-ing, a prompt will ask
for the password associated with the user email.


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   installation
   support

.. toctree::
   :maxdepth: 2
   :caption: Usage
   :hidden:

   devices

.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code/__init__
