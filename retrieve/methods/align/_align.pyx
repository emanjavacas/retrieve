
def sw_alignment(
        int s1len,
        int s2len,
        # penalties
        float e_gap,
        float o_gap,
        float t_gap,
        object scores):

    # pointers
    cdef int i, j, row, col, move
    cdef float mscore, vscore, hscore, best_score
    cdef int VGAP = 1
    cdef int MATCH = 2
    cdef int HGAP = 3
    cdef int END = 4
    cdef int GAP_SYM = -1

    # output and intermediate computations
    cdef list a1 = []
    cdef list a2 = []
    cdef list smatrix = [[0.0 for i in range(s2len + 1)] for j in range(s1len + 1)]
    cdef list tmatrix = [[4 for i in range(s2len + 1)] for j in range(s1len + 1)]

    # compute scores and traceback matrix
    best_score = 0.0
    for i in range(1, s1len + 1):
        for j in range(1, s2len + 1):

            # match
            mscore = smatrix[i-1][j-1] + scores[i-1,j-1]

            # vertical
            if i == s1len:
                vscore = smatrix[i-1][j] + t_gap
            elif tmatrix[i-1][j] == VGAP:
                vscore = smatrix[i-1][j] + e_gap
            else:
                vscore = smatrix[i-1][j] + o_gap

            # horizontal
            if j == s2len:
                hscore = smatrix[i][j-1] + t_gap
            elif tmatrix[i][j-1] == HGAP:
                hscore = smatrix[i][j-1] + e_gap
            else:
                hscore = smatrix[i][j-1] + o_gap

            if mscore >= vscore and mscore >= hscore and mscore >= 0.0:
                smatrix[i][j] = mscore
                tmatrix[i][j] = MATCH
            elif vscore >= hscore and vscore >= 0.0:
                smatrix[i][j] = vscore
                tmatrix[i][j] = VGAP
            elif hscore >= 0.0:
                smatrix[i][j] = hscore
                tmatrix[i][j] = HGAP
            else:
                smatrix[i][j] = 0.0
                tmatrix[i][j] = END

            if smatrix[i][j] >= best_score:
                best_score = smatrix[i][j]
                row, col = i, j

    best_score = smatrix[row][col]
    move = tmatrix[row][col]

    # traceback
    while move != END:
        if move == MATCH:
            a1.append(row - 1)
            a2.append(col - 1)
            row -= 1
            col -= 1
        elif move == VGAP:
            a1.append(row - 1)
            a2.append(GAP_SYM)
            row -= 1
        else:
            a1.append(GAP_SYM)
            a2.append(col - 1)
            col -= 1
        move = tmatrix[row][col]

    return a1[::-1], a2[::-1], best_score
