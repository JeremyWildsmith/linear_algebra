## Jeremy Wildsmith 2022
## Feb 17

import math
import sympy as sym
from sympy import mod_inverse

class Matrix(object):
    def __init__(self, row, col, *, init_data=True):
        if init_data:
            self.data = [0] * (row * col)
        
        self.rows = row
        self.cols = col

    @staticmethod
    def create(src_data):
        all_data = []
        rows = 0
        cols = -1
        for r in src_data:
            rows += 1
            if cols < 0:
                cols = len(r)
            elif cols != len(r):
                raise ValueError()

            all_data += r

        res = Matrix(rows, cols)
        res.data = all_data

        return res

    @staticmethod
    def create_row(src_data):
        mat = Matrix(1, len(src_data), init_data=False)
        mat.data = list(src_data)

        return mat

    @staticmethod
    def create_column(src_data):
        return Matrix.create_row(src_data).transpose()

    @staticmethod
    def identity(dim):
        mat = Matrix(dim, dim)

        for i in range(dim):
            mat.data[i * dim + i] = 1
        
        return mat

    def rows(self):
        return self.rows
    
    def columns(self):
        return self.cols

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        
        for i in range(self.rows):
            for j in range(self.cols):
                result[(j, i)] = self[(i, j)]

        return result

    def add(self, other):
        if self.cols != other.cols or self.rows != other.rows:
            raise ValueError()
        
        r = Matrix(self.rows, self.cols, init_data=False)

        r.data = [self.data[i] + other.data[i] for i in range(len(self.data))]

        return r

    def sub(self, other):
        return self.add(other.scale(-1))        

    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError()

        product = Matrix(self.rows, other.cols)
        touched = []
        for i in range(product.cols * product.rows):
            dest_idx = ((int)(i / product.cols), i % product.cols)
            touched.append(dest_idx)
            calc = 0
            for im in range(product.rows):
                calc += self[(dest_idx[0], im)] * other[(im, dest_idx[1])]
            
            product[dest_idx] = calc
        
        return product

    def scale(self, scale):
        scaled = Matrix(self.rows, self.cols)

        for i in range(len(self.data)):
            scaled.data[i] = self.data[i] * scale

        return scaled

    def cross(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError()

        transpose = False
        
        if self.rows == 1:
            transpose = False
        elif self.cols == 1:
            transpose = True
        else:
            raise ValueError()

        working_a = self.transpose() if transpose else self
        working_b = other.transpose() if transpose else other

        working = Matrix(3, self.cols)

        working.writematrix(0, 0, Matrix.create_row([1 for i in range(self.cols)]))
        working.writematrix(1, 0, working_a)
        working.writematrix(2, 0, working_b)
        
        result = Matrix(1, self.cols)

        for i in range(self.cols):
            component = pow(-1, i) * determinant(minor(working, 0, i))
            result[(0, i)] = component

        if transpose:
            result = result.transpose()

        return result

    def dot(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError()
        
        return sum([self.data[i] * other.data[i] for i in range(len(self.data))])

    def submatrix(self, row, column, sub_dim, sub_dim_cols = -1):
        if sub_dim_cols < 0:
            sub_dim_cols = sub_dim

        dest_matrix = Matrix(sub_dim, sub_dim_cols)

        for x in range(sub_dim):
            for y in range(sub_dim_cols):
                dest_matrix[(x,y)] = self[(row + x, column + y)]

        return dest_matrix

    def writematrix(self, row, column, src_data):
        for x in range(src_data.rows):
            for y in range(src_data.cols):
                self[(row + x, column + y)] = src_data[(x, y)]

    def duplicate(self):
        m = Matrix(self.rows, self.cols)
        m.data = list(self.data)
        return m

    def is_zero(self):
        return not any([x > 0.0000001 for x in self.data])

    def __str__(self):
        str_data = []
        max_len = 0

        for i in range(len(self.data)):
            dat = self.data[i]
            try:
                dat = str(round(dat, 2))
            except TypeError as e:
                dat = str(dat)
            
            max_len = max(max_len, len(dat))
            str_data.append(dat)
        
        rows = []
        for i in range(self.rows):
            start = i * self.cols
            row_str = ", ".join([str(str_data[start + n]).rjust(max_len) for n in range(self.cols)])
            rows.append(row_str)

        return "\n".join(rows)

    def __getitem__(self, key):
        idx = key[0] * self.cols + key[1]
        return self.data[idx]

    def __setitem__(self, key, data):
        idx = key[0] * self.cols + key[1]
        self.data[idx] = data

def minor(matrix, row, col):
    minor = Matrix(matrix.rows - 1, matrix.cols - 1)

    for ii in range(matrix.rows):
        if ii == row:
            continue

        for jj in range(matrix.cols):
            if jj == col:
                continue
            
            i = (ii - 1) if ii > row else ii
            j = (jj - 1) if jj > col else jj

            minor[(i, j)] = matrix[(ii, jj)]

    return minor

def eucl_dist(matrix):
    if matrix.cols != 1 and matrix.rows != 1:
        raise ValueError()
    
    dist = math.sqrt(sum([x * x for x in matrix.data]))

    return dist

def solve_det(matrix, solutions):
    if matrix.cols != solutions.rows or solutions.cols != 1:
        raise ValueError()

    delta = determinant(matrix)

    res = []

    for c in range(matrix.cols):
        working = matrix.duplicate()
        working.writematrix(0, c, solutions)
        det = determinant(working)
        res += [det / delta]

    res_mat = Matrix(len(res), 1)
    res_mat.data = res

    return res_mat

def solve_direct_gj(matrix, solutions):
    if matrix.cols != solutions.rows or solutions.cols != 1:
        raise ValueError()
    
    working = Matrix(matrix.rows, matrix.cols + 1)
    working.writematrix(0, 0, matrix)
    working.writematrix(0, matrix.rows, solutions)

    gauss_jordan(working, matrix.cols, matrix.cols + 1)

    result = Matrix(solutions.rows, 1)

    r = working.submatrix(0, matrix.cols, matrix.rows, 1)
    result.writematrix(0, 0, working.submatrix(0, matrix.cols, matrix.rows, 1))

    return result

def gauss_jordan(working, pivot_cols, total_cols):
    print("Starting guass jordan elimination on matrix")
    print(working)

    for i in range(pivot_cols):
        pivot = (i, i)
        print("Pivot: " + str(pivot))

        src = working[pivot]

        print("Divide row " + str(i) + " by " + str(src))
        #divide entire row by pivot
        for ic in range(total_cols):
            working[(i, ic)] /= src
        
        print(working)

        print("Add row with other rows to clear column " + str(pivot[1]))
        for r in range(working.rows):
            if r == pivot[0]:
                continue

            val = working[(r, pivot[1])]

            for ic in range(total_cols):
                working[(r, ic)] -= val * working[(pivot[0], ic)]

        print(working)

def inverse_gj(matrix):
    if matrix.cols != matrix.rows:
        raise ValueError()

    working = Matrix(matrix.rows, matrix.cols * 2)
    working.writematrix(0, matrix.cols, Matrix.identity(matrix.rows))
    working.writematrix(0, 0, matrix)

    gauss_jordan(working, matrix.cols, matrix.cols * 2)
    
    inverse = Matrix(matrix.rows, matrix.rows)
    inverse.writematrix(0, 0, working.submatrix(0, matrix.cols, matrix.cols))

    return inverse

def cofactors(matrix):
    cofactors = Matrix(matrix.rows, matrix.cols)

    for key, val in minors(matrix).items():
        sign = pow(-1, key[0] + key[1])
        cofactors[key] = sign * determinant(val[1])

    return cofactors

def determinant(matrix):
    queue = []
    queue.append((1, matrix))
    calc = 0

    while queue:
        w = queue.pop()
        coefficient = w[0]
        mat = w[1]

        if mat.rows == 0 and mat.cols == 0:
            calc += coefficient
        elif mat.rows == 2 and mat.cols == 2:
            det = mat[(0,0)] * mat[(1,1)] - mat[(0,1)] * mat[(1,0)]
            det *= coefficient
            calc += det
        else:
            for key, value in minors(mat).items():
                if key[0] != 0:
                    continue

                sign = pow(-1, key[0] + key[1])
                sign *= value[0] * coefficient
                    
                queue.append((sign, value[1]))

    return calc

def normalize(matrix):
    if matrix.is_zero():
        raise ValueError()

    l = eucl_dist(matrix)

    r = Matrix(matrix.rows, matrix.cols, init_data=False)

    r.data = [matrix.data[i] / l for i in range(len(matrix.data))]

    return r

def minors(matrix):
    min_map = {}

    for i in range(matrix.rows):
        for j in range(matrix.cols):
            idx = (i, j)
            min_map[idx] = (matrix[idx], minor(matrix, i, j))
    
    return min_map

def inverse_cofactors(matrix):
    #inverse = 1/det(a) * adj(a)
    #adj(a) = transpose(C)
    ct = cofactors(matrix).transpose()
    return ct.scale(1/determinant(matrix))

def find_eigen_vectors(matrix):
    if matrix.rows != matrix.cols:
        raise ValueError()

    symlambda = sym.Symbol("λ")

    lambda_mat = Matrix.identity(matrix.rows).scale(symlambda)

    sublambda = matrix.sub(lambda_mat)

    print("Solving λ for determinant of below matrix being equal to zero")
    print(sublambda)
    print()
    solve_mat = determinant(sublambda)
    print(str(solve_mat) + " = 0")

    # Systematically solving polynomials is outside of scope of this project
    # Instead, I will use sympy to find lambdas after create the system necessary to solve (as seen above)
    print("Using sympy to solve system")
    solutions = sym.solveset(solve_mat, symlambda)

    col_sym_data = [sym.Symbol("c" + str(i), real=True) for i in range(matrix.rows)]

    var_matrix = Matrix(matrix.rows, 1, init_data=False)
    var_matrix.data = col_sym_data

    positive_symbol = sym.Symbol("p", positive=True)

    eigen_solutions = []

    for s in solutions:
        sysmat = matrix.sub(Matrix.identity(matrix.rows).scale(s)).multiply(var_matrix)
        sys = [
            sum(sysmat.submatrix(r, 0, 1, sysmat.cols).data)
            for r in range(sysmat.rows)
        ]

        #Sum of all variables should be greater than zero
        sys += [sum([abs(x) for x in col_sym_data]) - positive_symbol]

        vec = sym.solve(sys, col_sym_data)
        
        eigen_vector = Matrix(matrix.rows, 1)
        
        for i in range(matrix.rows):
            eigen_vector.data[i] = vec[0][i].subs(positive_symbol, 1)

        eigen_solutions.append((s, eigen_vector))

    return eigen_solutions

def example_find_normal_tang_comp():
    a = Matrix.create_row([3,6])
    wrt = Matrix.create_row([4,4])
    wrt_unit = normalize(wrt)
    parallel = wrt_unit.scale(a.dot(wrt_unit))
    normal = a.sub(parallel)

    print("Components of: ")
    print(a)

    print("With respect to a vector")
    print(wrt_unit)

    print("Parallel comonent:")
    print(parallel)
    print("Normal component:")
    print(normal)


def example_solve_system0():
    #x = 3, y=10, z = 4
    coefficients = Matrix.create([
        [2, 1, 1],
        [7, 2, 3],
        [3, 3, 3]
    ])

    sol = Matrix.create([
        [20],
        [53],
        [51]
    ])

    print("Solving System Ax=B for x")
    print("A:")
    print(coefficients)
    print()
    print("B:")
    print(sol)
    print()
    print("Finding inverse...")
    inverse = inverse_gj(coefficients)
    print()
    print("Found inverse:")
    print(inverse)
    print()
    print("Solved for x (inverse * A):")
    print(inverse.multiply(sol))

def example_solve_system1():
    #x = 3, y=10, z = 4
    coefficients = Matrix.create([
        [2, 1, 1],
        [7, 2, 3],
        [3, 3, 3]
    ])

    sol = Matrix.create([
        [20],
        [53],
        [51]
    ])

    print("Solving System Ax=B for x")
    print("A:")
    print(coefficients)
    print()
    print("B:")
    print(sol)
    print()

    res = solve_direct_gj(coefficients, sol)
    print()
    print("Solved for x (Direct GJ):")
    print(res)

def example_solve_system2():
    #x = 3, y=10, z = 4
    coefficients = Matrix.create([
        [2, 1, 1],
        [7, 2, 3],
        [3, 3, 3]
    ])

    sol = Matrix.create([
        [20],
        [53],
        [51]
    ])

    print("Solving System Ax=B for x")
    print("A:")
    print(coefficients)
    print()
    print("B:")
    print(sol)
    print()

    print("Solved for x:")
    print(solve_det(coefficients, sol))

def example_find_plane():
    #Find plane with three points
    a = Matrix.create_row([1,1,1])
    b = Matrix.create_row([3,4,7])
    c = Matrix.create_row([0,8,1])

    #Create a line between AB and AC
    ab = b.sub(a)
    ac = c.sub(a)

    print("Points are:")
    print("A: (" + str(a) + ")")
    print("B: (" + str(b) + ")")
    print("C: (" + str(c) + ")")
    print()

    print("Vector AB: (" + str(ab) + ")")
    print("Vector AC: (" + str(ac) + ")")

    normal = ab.cross(ac)

    print("Plane Normal Vector: (" + str(normal) + ")")

    #if equation multiplied by a point
    #PP' DOT N = 0
    #If a given point P' minus a point on the plane P, dotted with our normal is zero, we know the point is on the plane
    #P' minus a point P will create a vector between P and P', if this vector is orthogonal to the normal, then it is on the plane
    def on_plane(p):
        return p.sub(a).dot(normal) == 0

    print("A on plane: " + str(on_plane(a)))
    print("B on plane: " + str(on_plane(b)))
    print("C on plane: " + str(on_plane(c)))
    print("(0,0,0) on plane: " + str(on_plane(Matrix.create_row([0,0,0]))))

    px = sym.Symbol("x")
    py = sym.Symbol("y")
    pz = sym.Symbol("z")

    print("Plane equation is:")
    plane_eq = Matrix.create_row([px, py, pz]).sub(a).dot(normal)
    print(str(plane_eq) + " = 0")

def example_find_eigen():
    print("""
    This example demonstrates finding eigen-values and eigen vectors.
    """)

    mat = Matrix.create([
        [-1, 0],
        [0, 1],
    ])

    print("Finding eigenvalues / vectors for the matrix:")
    print(mat)
    print()

    vectors = find_eigen_vectors(mat)

    #Verify answers are correct
    for v in vectors:
        print("Found eigenvalue/vector:")
        print("    - eigenvalue: " + str(v[0]))
        print("    - eignevector: ")
        print(str(v[1]))
        print()
        expected = v[1].scale(v[0])
        r = mat.multiply(v[1])
        passed = all([r.data[i] == expected.data[i] for i in range(len(r.data))])
        print("Confirmed is valid" if passed else "FAILED")
        print()

def example_find_angle():
    print("""
    This example demonstrates dot product between two vectors to find the angle between them
    """)

    # Finding angle between two vectors
    a = Matrix.create_row([1,1,1])
    b = Matrix.create_row([3,4,7])
    c = Matrix.create_row([0,8,1])

    print("Points are:")
    print("A: (" + str(a) + ")")
    print("B: (" + str(b) + ")")
    print("C: (" + str(c) + ")")
    print()

    ab = b.sub(a)
    ac = c.sub(a)

    print("Vector AB: (" + str(ab) + ")")
    print("Vector AC: (" + str(ac) + ")")

    # a dot b = |a||b|cos theta
    #theta = inverse_cos( (a dot b) / ( len_a * len_b) )
    ab_dot_ac = ab.dot(ac)
    print("AB dot AC: " + str(ab_dot_ac))
    angle = math.acos(ab_dot_ac / (eucl_dist(ab) * eucl_dist(ac))) / math.pi * 180.0

    if angle > 180:
        angle = 180 - angle

    print("Angle between AB and AC is " + str(round(angle, 2)) + " degrees")


def example_hillcipher():

    A_CODE = "A".encode("utf-8")[0]
    def encode_mod27(c):
        delta = 1 + (c - A_CODE)

        if delta < 0 or delta > 27:
            return 0
        
        return int(delta)

    def decode_mod27(c):
        d = int(c) % 27
        if d == 0:
            return '-'
        
        return chr(A_CODE + d - 1)

    def encrypt(key, message):
        digraph_rows = key.rows
        
        message += " " * (len(message) % digraph_rows)

        upper_message = message.upper().encode("utf-8")
        
        digraph_columns = int(len(message) / digraph_rows)
        digraph = Matrix(digraph_rows, digraph_columns)

        for i in range(digraph_columns):
            dg = Matrix(digraph_rows, 1, init_data=False)
            idx = i * digraph_rows
            dg.data = [encode_mod27(x) for x in upper_message[idx:(idx + digraph_rows)]]
            digraph.writematrix(0, i, dg)

        cipher_mat = key.multiply(digraph)
        print(key)
        
        for i in range(len(cipher_mat.data)):
            cipher_mat.data[i] %= 27

        ciphertext = ""
        for d in cipher_mat.transpose().data:
            ciphertext += decode_mod27(d)

        return ciphertext

    def decrypt(key, message):
        digraph_rows = key.rows

        decrypt_key_adj = cofactors(key).transpose()

        det_inverse = mod_inverse(determinant(key) % 27, 27)

        decrypt_key = decrypt_key_adj.scale(det_inverse)

        for i in range(len(decrypt_key.data)):
            decrypt_key.data[i] %= 27

        data_message = [encode_mod27(x) for x in message.encode("utf-8")]
        
        if len(data_message) % digraph_rows != 0:
            raise ValueError()
        
        digraph_columns = int(len(message) / digraph_rows)

        digraph = Matrix(digraph_columns, digraph_rows, init_data=False)
        digraph.data = [int(x) for x in data_message]
        digraph = digraph.transpose()

        plain_mat = decrypt_key.multiply(digraph)

        plaintext = ""

        for d in plain_mat.transpose().data:
            plaintext += decode_mod27(d)

        return plaintext

    print("Example 2x2 key")
    # Example with a 2x2 matrix
    test_key_2x2 = Matrix.create([
        [3, 5],
        [1, 6]
    ])

    if determinant(test_key_2x2) % 3 == 0:
        raise ValueError()

    ciphertext = encrypt(test_key_2x2, "PLEASE USE THE OTHER DOOR")
    plaintext = decrypt(test_key_2x2, ciphertext)

    print("Encrypted, cipher-text was: " + ciphertext)
    print("Decrypted, plain-text after decrypt was: " + plaintext)


    print("Example 4x4 key")
    # Example with a 2x2 matrix
    test_key_4x4 = Matrix.create([
        [3, 5, 1, 8],
        [1, 7, 3, 1],
        [9, 6, 2, 1],
        [2, 1, 3, 1]
    ])

    if determinant(test_key_4x4) % 3 == 0:
        raise ValueError()

    ciphertext = encrypt(test_key_4x4, "PYTHON IS GREAT")
    plaintext = decrypt(test_key_4x4, ciphertext)

    print("Encrypted, cipher-text was: " + ciphertext)
    print("Decrypted, plain-text after decrypt was: " + plaintext)

examples = {
    "Finding Eigen Values and Vectors": example_find_eigen,
    "Example shows using dot product to calculate angle between vectors": example_find_angle,
    "Using dot products for find parallel and normal components to a given vector": example_find_normal_tang_comp,
    "Uses cross product to find a plane containg three different points": example_find_plane,
    "Example of solving a linear system by using the inverse found with Gauss-Jordan Elimination": example_solve_system0,
    "Example solves a linear system using direct guass jordan elimination": example_solve_system1,
    "Example solves a linear system using camer's rule": example_solve_system2,
    "Example shows hillcipher encryption and decryption using matrices and inverse matrices": example_hillcipher
}

def main():
    for name, proc in examples.items():
        print("EX: " + name)
        print("-" * 40)
        proc()
        print()
        print()
        print()

if __name__ == "__main__":
    main()