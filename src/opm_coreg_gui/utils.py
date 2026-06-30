"""Utility functions for rigid registration and VTK visualization."""

import numpy as np
import vtk

# TODO: Write appropriate comments wherever necessary.


def compute_rigid_transform(src_pts, dst_pts):
    """Compute the rigid transformation that best aligns src_pts to dst_pts.

    Args:
        src_pts: List of source points (Nx3).
        dst_pts: List of destination points (Nx3).

    Returns:
        T: 4x4 homogeneous transformation matrix.
        rms: Root mean square error of the alignment.
        errs: Per-point errors after transformation.

    Raises:
        TypeError: if inputs cannot be converted to numeric arrays.
        ValueError: if shapes are incompatible, contain non-finite values, or do not provide
                    a sufficient / well-conditioned point configuration to determine a unique
                    rigid transform.
    """
    # Input checks
    try:
        src = np.asarray(src_pts, dtype=float)
    except Exception as exc:
        raise TypeError("src_pts must be array-like of numeric values") from exc
    try:
        dst = np.asarray(dst_pts, dtype=float)
    except Exception as exc:
        raise TypeError("dst_pts must be array-like of numeric values") from exc

    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src_pts must have shape (N,3); got {src.shape}")
    if dst.ndim != 2 or dst.shape[1] != 3:
        raise ValueError(f"dst_pts must have shape (N,3); got {dst.shape}")
    if src.shape != dst.shape:
        raise ValueError(
            f"src_pts and dst_pts must have the same shape; got {src.shape} vs {dst.shape}"
        )

    n_points = src.shape[0]
    if n_points < 3:
        raise ValueError(
            f"At least 3 points are required to compute a rigid transform; got {n_points}"
        )

    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise ValueError("Input points must not contain NaN or infinite values")

    # Central algorithm
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)

    # Check conditioning of the cross-covariance: ensure we have enough non-negligible singular values
    tol = np.max(S) * max(src.shape) * np.finfo(S.dtype).eps
    n_nonzero = int((S > tol).sum())
    if n_nonzero < 2:
        # less than 2 meaningful singular values indicates degenerate configuration (e.g. identical points)
        raise ValueError(
            "Point configuration is degenerate (rank-deficient). "
            "Provide at least three non-colinear points to determine a unique rigid transform."
        )

    # Get rotation matrix and translation vector
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_mean - R @ src_mean
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    errs = np.linalg.norm(((R @ src.T).T + t) - dst, axis=1)
    rms = np.sqrt(np.mean(errs**2))
    return T, rms, errs


def validate_transform(T, ortho_tol=1e-3, det_tol=1e-3, finite_tol=1e-6):
    """Validate a 4x4 rigid transform matrix.

    Checks:
    - shape is (4,4)
    - last row is [0,0,0,1]
    - rotation part is orthonormal (R.T @ R ~= I)
    - rotation det ~= 1 (no reflection/scale)
    - all elements finite

    Returns (bool, message).
    """
    T = np.asarray(T)
    if T.shape != (4, 4):
        return False, f"Expected shape (4,4), got {T.shape}"
    if not np.all(np.isfinite(T)):
        return False, "Non-finite values in transform"
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=finite_tol):
        return False, f"Bottom row must be [0,0,0,1], got {T[3, :]}"
    R = T[:3, :3]
    RtR = R.T @ R
    if not np.allclose(RtR, np.eye(3), atol=ortho_tol):
        return (
            False,
            f"Rotation not orthonormal (max err={np.max(np.abs(RtR - np.eye(3))):.4g})",
        )
    det = np.linalg.det(R)
    if not np.isfinite(det):
        return False, "Rotation determinant not finite"
    if not np.isclose(det, 1.0, atol=det_tol):
        return False, f"Rotation determinant {det:.4g} differs from 1"
    return True, "OK"


def vtk_matrix_from_numpy(T):
    """Convert a 4x4 numpy array to a vtkMatrix4x4.

    Args:
        T: 4x4 numpy array representing a homogeneous transformation.

    Returns:
        M: vtkMatrix4x4 with the same values as T.
    """
    M = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            M.SetElement(i, j, float(T[i, j]))
    return M


def load_mesh_to_polydata(path):
    """Load a mesh file (PLY or STL) and return it as vtkPolyData.

    Args:
        path: Path to the mesh file.

    Returns:
        pd: vtkPolyData containing the mesh geometry.
    """
    if path.lower().endswith(".ply"):
        r = vtk.vtkPLYReader()
    else:
        r = vtk.vtkSTLReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def actor_from_polydata(pd, color=(0.85, 0.85, 0.85), opacity=1.0):
    """Create a vtkActor from vtkPolyData with specified color and opacity.

    Args:
        pd: vtkPolyData containing the geometry to visualize.
        color: Tuple of (R, G, B) values for the actor's color.
        opacity: Float value between 0 and 1 for the actor's opacity.

    Returns:
        a: vtkActor ready to be added to a vtkRenderer.
    """
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    a = vtk.vtkActor()
    a.SetMapper(mapper)
    a.GetProperty().SetColor(*color)
    a.GetProperty().SetOpacity(opacity)
    return a


def sphere_actor(pos, color=(1, 0, 0), radius=2.5):
    """Create a vtkActor representing a sphere at the given position.

    Args:
        pos: Tuple of (x, y, z) coordinates for the sphere's center.
        color: Tuple of (R, G, B) values for the sphere's color.
        radius: Float value for the sphere's radius.

    Returns:
        a: vtkActor representing the sphere.
    """
    s = vtk.vtkSphereSource()
    s.SetCenter(pos)
    s.SetRadius(radius)
    s.SetPhiResolution(16)
    s.SetThetaResolution(16)
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(s.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*color)
    a.GetProperty().SetAmbient(0.3)
    a.GetProperty().SetDiffuse(0.7)
    return a


def create_text_actor(
    text, position, renderer, color=(1, 1, 1), font_size=14, offset=(5, 5, 5)
):
    """Create a VTK text label that always faces the camera.

    Args:
        text: Text to be displayed in the window.
        position: Tuple of (x, y, z) coordinates for the text location.
        renderer: renderer for the window in which text actors will be drawn
        color: Tuple of (R, G, B) values for the text color.
        font_size: font size of the text label.
        offset: Tuple of (offsetx, offsety, offstez) to offset the text label from the sphere centre.

    Returns:
        actor: vtkActor representing the text label.
    """
    # Create text source
    text_source = vtk.vtkVectorText()
    text_source.SetText(str(text))

    # Create mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(text_source.GetOutputPort())

    actor = vtk.vtkFollower()
    actor.SetMapper(mapper)
    actor.SetCamera(renderer.GetActiveCamera())

    # Position with offset
    position = (
        position[0] + offset[0],
        position[1] + offset[1],
        position[2] + offset[2],
    )
    actor.SetPosition(position[0], position[1], position[2])
    actor.SetScale(font_size)

    # Set text color
    actor.GetProperty().SetColor(color)

    # Optionally add a white background for readability
    actor.GetProperty().SetOpacity(1.0)

    return actor


def arrow_actor(start, vec, scale=1.0, color=(1, 0, 0)):
    """Create a vtkActor representing an arrow from start in the direction of vec.

    Args:
        start: Tuple of (x, y, z) coordinates for the arrow's base.
        vec: Tuple of (dx, dy, dz) representing the arrow's direction and length.
        scale: Float value to scale the arrow's length.
        color: Tuple of (R, G, B) values for the arrow's color.

    Returns:
        a: vtkActor representing the arrow, or None if vec is too small.
    """
    length = np.linalg.norm(vec)
    if length < 1e-9:
        return None
    direction = vec / length
    arrow = vtk.vtkArrowSource()
    x_axis = np.array([1.0, 0.0, 0.0])
    v = np.cross(x_axis, direction)
    s = np.linalg.norm(v)
    c = np.dot(x_axis, direction)
    R = np.eye(3)
    if s != 0:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
    mat = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            mat.SetElement(i, j, float(R[i, j] * length * scale))
        mat.SetElement(i, 3, float(start[i]))
    mat.SetElement(3, 0, 0)
    mat.SetElement(3, 1, 0)
    mat.SetElement(3, 2, 0)
    mat.SetElement(3, 3, 1)
    t = vtk.vtkTransform()
    t.SetMatrix(mat)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(arrow.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(mapper)
    a.SetUserTransform(t)
    a.GetProperty().SetColor(*color)
    return a


def save_transform(T, out_base):
    """Save a 4x4 transformation matrix to both .npy and .txt files.

    Args:
        T: 4x4 numpy array representing the transformation matrix.
        out_base (Path): Base path (without extension) for the output files.
    """
    np.save(str(out_base) + ".npy", T)
    np.savetxt(str(out_base) + ".txt", T, fmt="%.8f")
