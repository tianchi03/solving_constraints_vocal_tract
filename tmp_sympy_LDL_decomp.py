def LDL_sparse(matrix):
        """Algorithm for numeric LDL factorization, exploiting sparse structure.
        """
        Lrowstruc = matrix.row_structure_symbolic_cholesky()
        L = matrix.eye(matrix.rows)
        D = matrix.zeros(matrix.rows, matrix.cols)

        for i in range(len(Lrowstruc)):
            for j in Lrowstruc[i]:
                if i != j:
                    L[i, j] = matrix[i, j]
                    summ = 0
                    for p1 in Lrowstruc[i]:
                        if p1 < j:
                            for p2 in Lrowstruc[j]:
                                if p2 < j:
                                    if p1 == p2: #cancel possible ici
                                        summ += L[i, p1]*L[j, p1]*D[p1, p1]
                                else:
                                    break
                        else:
                            break
                    L[i, j] -= summ #ici
                    L[i, j] /= D[j, j] #ici
                else: # i == j
                    D[i, i] = self[i, i].cancel() ### cancel rajouté
                    summ = 0
                    for k in Lrowstruc[i]:
                        if k < i:
                            summ += (L[i, k]**2*D[k, k]).cancel() ### cancelrajouté
                        else:
                            break
                    D[i, i] -= summ
                    D[i,i] = D[i,i].cancel() #rajouté

    return L, D

def LDL_solve(matrix, rhs):
        # for speed reasons, this is not uncommented, but if you are
        # having difficulties, try uncommenting to make sure that the
        # input matrix is symmetric

        #assert matrix.is_symmetric()
        L, D = LDL_sparse(matrix)
        Z = lower_triangular_solve(L, rhs)
        Y = diagonal_solve(D, Z)
        return upper_triangular_solve(L, Y)).T

def lower_triangular_solve(matrix, rhs):
    """Fast algorithm for solving a lower-triangular system,
    exploiting the sparsity of the given matrix.
    """
    rows = [[] for i in range(matrix.rows)]
    for i, j, v in matrix.row_list():
        if i > j:
            rows[i].append((j, v))
    X = rhs.as_mutable().copy()
    for j in range(rhs.cols): #on peut éventuellement rajouter des cancel ici
        for i in range(rhs.rows):
            for u, v in rows[i]:
                X[i, j] -= v*X[u, j]
            X[i, j] /= matrix[i, i]
    return matrix._new(X)

def diagonal_solve(matrix, rhs):
    """Helper function of function diagonal_solve, without the error
    checks, to be used privately.
    """
    return matrix._new(
        rhs.rows, rhs.cols, lambda i, j: rhs[i, j] / matrix[i, i])


