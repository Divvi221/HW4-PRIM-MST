import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'
    assert mst.shape == adj_mat.shape #assert that mst adjacency matrix has same dimensions as the input graph adjacency matrix
    #assertion for num edges in mst which should be equal to N-1
    num_edges = 0
    for i in range(len(mst)):
        for j in range(i+1,len(mst[i])):
            if mst[i][j] > 0:
                num_edges += 1
    assert num_edges == (len(adj_mat) - 1)
    assert num_edges >= 0



def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    mat = np.array([[0, 2, 0, 3],
                    [2, 0, 1, 0],
                    [0, 1, 0, 4],
                    [3, 0, 4, 0]])
    x = Graph(mat)
    mst1 = Graph.construct_mst(x)
    total = 0
    for i in range(x.mst.shape[0]):
        for j in range(i+1):
            total += x.mst[i, j]
    assert total == 6 #expected weight = 6
    dif = mat - x.mst
    for elem in dif:
        for j in elem:
            print(j)
            assert j in mat #ensure that all elements in the mst are from the og matrix
