Install
=======

requirements
------------
We support Python 3.8 on Linux Similar Python versions will probably work, but we don't regularly test using those versions. 

managing your Python environment
--------------------------------

We recommend installing `transcriptomic_clustering` into a managed Python environment. Having multiple isolated environments lets you install incompatible packages (or different versions of the same package!) simultaneously and prevents unexpected behavior by utilities that rely on the system Python installation.

Two popular tools for managing Python environments are `anaconda <https://anaconda.org/anaconda/anaconda-project>`_ and `venv <https://docs.python.org/3/library/venv.html>`_. The rest of this document assumes that you have created and activated an environment using one of these tools. Using anaconda, this looks like:

.. code-block:: bash

    conda create -y --name environment-name python=3.8
    conda activate environment-name

and using venv:

.. code-block:: bash

    python -m venv path/to/environment
    source path/to/environment/bin/activate

installing from github
----------------------

If you want to install a specific branch, tag, or commit of `transcriptomic_clustering`, you can do so using pip:

.. code-block:: bash

    pip install git+https://github.com/alleninstitute/transcriptomic_clustering@dev

The *dev* branch contains cutting-edge features that might not have been formally released yet. By installing this way, you can access those features.

installing for development
--------------------------

If you want to work on `transcriptomic_clustering`, you should first clone the repository, then install it in editable mode so that you can easily test your changes:

.. code-block:: bash

    git clone https://github.com/alleninstitute/transcriptomic_clustering
    cd transcriptomic_clustering

    pip install -r requirements.txt -U
    pip install -r test_requirements.txt -U

    pip install -e .



installing from pypy [coming soon!]
-----------------------------------
You can install the latest release from pypy by running:

.. code-block:: bash

    pip install transcriptomic_clustering
