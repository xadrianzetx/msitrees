Installation guide
==================

With pip
--------

.. code-block:: bash

    pip install msitrees

From source
-----------

.. code-block:: bash

    git clone https://github.com/xadrianzetx/msitrees.git
    cd msitrees
    python setup.py install

Windows builds require at least `MSVC2015 <https://www.microsoft.com/en-gb/download/details.aspx?id=48145>`_.
There is a known issue with Python 3.5 Windows build, where parallel jobs (e.g. in random forest) are taking longer to complete
than in other builds. Use newer python versions if possible to avoid this problem.