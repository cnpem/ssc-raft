Usage
=====

The ``ssc-raft`` package is imported as ``sscRaft``. Algorithms can be executed using the following pattern:

.. code-block:: python

    import sscRaft

    data = ...
    params = {
        ...
    }

    output = sscRaft.function_name(data, params)


The examples above illustrate the general usage pattern of the package. 
For complete workflows, see the tutorials linked below. 
Detailed descriptions of algorithm-specific inputs, parameters, and outputs can be found in the :ref:`API reference <apipage>`.

.. - :ref:`io`
.. - :ref:`correction`
.. - :ref:`alignment`
.. - :ref:`phasefilter`
.. - :ref:`rings`
.. - :ref:`reconstruction`
.. - :ref:`forward`
.. - :ref:`pipelines`

.. toctree::
    :maxdepth: 1
    :caption: Contents

    tutorial/correction
    tutorial/phaseret
    tutorial/rings
    tutorial/alignment
    tutorial/excentric
    tutorial/reconstructions
    tutorial/forward
    tutorial/padding
    tutorial/pipelines




