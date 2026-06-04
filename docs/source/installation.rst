Installing ichor
----------------

To install ichor, simply do

.. code-block:: python

    python3 -m pip install -e ichor_core
    python3 -m pip install -e ichor_hpc
    python3 -m pip install -e ichor_cli

This will install all the packages in editable mode, so that any changes to the source code will
be available to the user directly.

+++++++++++++++++++++++++++++++++
Setting up ichor_config.yaml file
+++++++++++++++++++++++++++++++++

The **ichor_config.yaml** file is used to store configuration settings for the high performance computing (HPC)
clusters. This file is needed if you are using `ichor.hpc` and `ichor.cli` as these interface with the workload
manager on the HPC cluster. An example of the config file can be found in a separate page in the documentation,
as well as in the Github repo.

++++++++++++++++++++++++++++++
Setting up Python environments
++++++++++++++++++++++++++++++

Below is a more thorough explanation on how to set up ichor on a compute cluster such as CSF3 (used by the University of Manchester).


.. warning::

    You will need to make separate environments for CSF3 and CSF4.

    .. For CSF3 use ``source activate my_env`` to activate a CONDA environment in both the login node and when submitting jobs.
    .. Check out the guide here. Not sure why this is required.

    .. * `Anaconda CSF3 <https://ri.itservices.manchester.ac.uk/csf3/software/applications/anaconda-python/>`_

CSF3 and CSF4 differ in their Python module stacks. On CSF4, prefer the
non-Anaconda Python modules when available. For the active-learning daemon,
``python/3.11.3-gcccore-12.3.0`` is a suitable base because ARIADNE requires
Python >= 3.9. Run ``module avail python`` on the cluster if this module name
changes.

To load the recommended CSF4 Python module, use

.. code-block:: text

    module load python/3.11.3-gcccore-12.3.0
    module load python-bundle-pypi/2023.06-gcccore-12.3.0

The ``python-bundle-pypi`` module is useful while bootstrapping a venv because
it provides common packaging tools. Once the venv exists, use the venv's own
``python -m pip``.

If you are on CSF3 and only Anaconda provides a recent enough Python for your
project, you may still use Anaconda there. Avoid mixing Conda packages and pip
packages in the daemon environment unless you have a specific reason.

Check the active Python version with

.. code-block:: text

    python3 --version

On CSF3, you may need to do ``qrsh -l short`` as the network proxy is no longer available.
This goes into a submit (compute) node, you can access the internet and install packages as well as make new conda environments with different python versions.

.. warning::

    You will need to load the same Python module and activate the same venv again
    on whichever node installs packages. Create environments while on a node with
    internet access. After you have installed all the packages, you should be able
    to submit jobs using that venv.

Now you can make a ``venv`` environment which will use the Python version from
the loaded module. To make a venv, do

.. code-block:: text

    python3 -m venv ~/.venv/ichor

This creates a virtual environment in the ``~/.venv/ichor`` folder and all environment packages will be installed here.
To activate the venv environment, do ``source ~/.venv/env_name/bin/activate``.
To activate on GitBash, do ``. ~/.venv/ichor/Scripts/activate``.

.. note::

    You will not need an Anaconda module when using a venv made from a
    non-Anaconda Python module. You do still need to load the same Python module
    before activating the venv so the interpreter and runtime libraries match.

You should see ``(ichor)`` show up on the left side of the terminal, which indicates you are in the ``ichor`` environment. This is the
same for both venv and conda.

Make sure that you have at least python 3.7 in the current venv or conda environment and that setuptools and pip are all up to date.

To make sure you are using the latest versions of the packages, use

.. code-block:: text

    python3 -m pip install --upgrade pip setuptools

++++++++++++++++++++++++++++++
Downloading ichor
++++++++++++++++++++++++++++++

First, download the ichor source code to your home directory (again you need to be on the compute node to have internet access). It is recommended to download the code as a git repository,
so that you can pull changes from the github code when changes are made. You can follow the Github guides on how to clone.
If you download the code as a zip, you will not be able to pull from github and will have to download the code every time a change is made!

.. warning::

    You will need to use HTTPS to clone a repository to CSF3/CSF4 as SSH is not supported on the servers.
    Therefore, you will also need to create a Personal Access Token as Github no longer accepts direct password authentication on a server.
    Below are two guides how to clone a repository and create a personal access token

    * `Github Cloning a Repository <https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository>`_
    * `Github Personal Access Token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens>`_


To install each of the sub-packages, do

.. code-block:: python

    python3 -m pip install -e ichor_core
    python3 -m pip install -e ichor_hpc
    python3 -m pip install -e ichor_cli

Please install these in the given order, as there are dependencies between the packages.

The ``-e`` flag installs the package in editable mode,
meaning that changes in the ichor source code will be directly made in the installed package. As ichor is still work in progress, it makes it easier to make changes and then test the changes.

.. warning::

    You will need to have access to the relevant
    software on the computer cluster if submitting jobs with `ichor.hpc` or
    `ichor.cli`. Currently, the paths to programs are hard coded into the ichor code, so
    they will need to exist at the correct paths.

    Also, make sure that you have access to the right versions of the software
    on the right cluster.

.. note::

    You need to be connected to the internet to be able to download and install the relevant
    dependencies of ichor.

.. note::

    Note it is usually better to use venv.
    On CSF4, load a recent non-Anaconda Python module first, then create the
    venv from that interpreter. On CSF3, use whichever recent Python module is
    available to you; if that is Anaconda, use it only to create the venv and
    then install packages into the venv.
