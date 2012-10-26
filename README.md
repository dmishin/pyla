Pure Python Linear Algebra
==========================

Features
--------

- Basic matrix and vector operations: multiplication, summation, scaling, transposition, inversion.
- Some more advanced matrix decompositions:
    - QR decomposition
    - SVD decomposition

Why use PYLA 
------------
PYLA is
- Lightweight. 
    It is much smaller than Numpy
- Standard pure Python. 
    It should work on every pyhton implementation, that is decent enough.
- Generic. 
    It is not limited by floats. Particularly, you can use GMPY's long floats with all matrix algorithms.

Performance
-----------

Performance is not a primary goal for PYLA. If you need performance, use of native libraries, such as Numpy, is highly recommended. As a a consequence, no special effort was done to increase performance.
However, the algorithms themselves are 

