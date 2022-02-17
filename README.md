# Linear Algebra Theory and Programming
The purpose of this repository is to experiment and apply different concepts and theories I've learned from studying linear algebra and implement them in python.

This repository implements the following:
- Basic matrix operations. multiplication, dot, cross, euclidean distance
- Finding matrix eigen values and eigen vectors
- Solving systems of linear equations using inverse (found with GJ Elimination, using cofactors or apply direct gauss-jordan elimination)
- Example of hillcipher encryption and decryption using matrices
- Using cross product to find planes containing set of three points


None of the implementations of the algorithms in this repository are meant to be fast or memory efficient. The aim to to keep them easy to understand and mirror the processes I have studied for manually applying the above theory.

# 3rd-Party Dependencies
I use numpy and sympy to solve systems of equations that fall outside of the scope of the linear algebra I am studying or are
difficult to do systematically. For example, sympy is used to solve the roots of a polynomial equations produced when finding eigen values or eigen vectors.

# Usage
You can refer to examples (all methods starting with `example_`) as examples for usage. This is code is not meant to be used in production.

1. Create a python venv `python3 -m venv venv`
2. Activate your venv `source ./venv/bin/activate`
3. Install the requirements (requirements.txt) into your venv. `pip install -r requirements.txt`
4. Run the examples by invoking the matrix.py script `python src/matrix.py` (The examples will execute and print to console)

# License
The license for all code contained in this repository is BSD 3
