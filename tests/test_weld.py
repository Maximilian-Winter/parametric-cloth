import numpy as np

from parametric_cloth.simulation.weld import weld_vertices


def test_simple_pair_merges_to_midpoint():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    new_v, new_f, old_to_new = weld_vertices(verts, faces, [(0, 3)])
    assert len(new_v) == 3
    assert np.allclose(new_v[old_to_new[0]], [0, 0, 0.5])
    assert old_to_new[0] == old_to_new[3]


def test_transitive_three_way_merge():
    verts = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0], [9, 9, 9]], float)
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    new_v, new_f, old_to_new = weld_vertices(verts, faces, [(0, 1), (1, 2)])
    assert len(new_v) == 2
    assert old_to_new[0] == old_to_new[1] == old_to_new[2]
    assert np.allclose(new_v[old_to_new[0]], [2, 0, 0])


def test_degenerate_faces_are_dropped():
    verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    faces = np.array([[0, 1, 2]])
    _, new_f, _ = weld_vertices(verts, faces, [(0, 1), (1, 2)])
    assert len(new_f) == 0


def test_no_pairs_leaves_mesh_unchanged():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2]])
    new_v, new_f, old_to_new = weld_vertices(verts, faces, [])
    assert np.allclose(new_v, verts)
    assert np.array_equal(new_f, faces)
    assert list(old_to_new) == [0, 1, 2]


def test_unrelated_vertices_stay_separate():
    verts = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5], [6, 6, 6]], float)
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    new_v, _, old_to_new = weld_vertices(verts, faces, [(0, 1)])
    assert len(new_v) == 3
    assert old_to_new[2] != old_to_new[3]
