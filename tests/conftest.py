import numpy as np
import pytest


class DummyGUI:
    def __init__(self):
        self.info_messages = []

    def update_info(self, msg):
        self.info_messages.append(msg)


@pytest.fixture
def gui():
    """Base GUI fixture with empty state."""
    return DummyGUI()


@pytest.fixture
def gui_with_valid_aux_points(gui):
    gui.aux_alignment_points_from_source = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
    ]
    gui.aux_alignment_points_from_target = [
        np.array([1, 1, 1]),
        np.array([2, 1, 1]),
        np.array([1, 2, 1]),
    ]
    return gui


@pytest.fixture
def gui_with_valid_sensor_points(gui):
    gui.sensor_alignment_points_from_source = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
    ]
    gui.sensor_alignment_points_from_target = [
        np.array([1, 2, 3]),
        np.array([2, 2, 3]),
        np.array([1, 3, 3]),
    ]
    return gui


@pytest.fixture
def gui_with_valid_head_points(gui):
    gui.head_alignment_points_from_source = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
    ]
    gui.head_alignment_points_from_target = [
        np.array([0, 0, 1]),
        np.array([1, 0, 1]),
        np.array([0, 1, 1]),
    ]
    return gui


@pytest.fixture
def valid_transform():
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]
    return T


@pytest.fixture
def invalid_transform():
    T = np.eye(4)
    T[0, 0] = 2  # breaks orthonormality
    return T


@pytest.fixture
def gui_with_transforms(gui, valid_transform):
    gui.T_aux_to_scan = valid_transform.copy()
    gui.T_dev_to_scan = valid_transform.copy()
    gui.T_head_to_scan = valid_transform.copy()
    return gui


@pytest.fixture
def gui_with_fiducials(gui):
    gui.marked_fiducials_on_head_scan = {
        "NAS": np.array([0, 0, 0]),
        "LPA": np.array([1, 0, 0]),
        "RPA": np.array([0, 1, 0]),
    }
    gui.mri_fiducials = {
        "NAS": np.array([1, 1, 1]),
        "LPA": np.array([2, 1, 1]),
        "RPA": np.array([1, 2, 1]),
    }
    return gui


@pytest.fixture
def mock_compute_transform_success(monkeypatch, valid_transform):
    def _mock(src, dst):
        n = len(src)
        return valid_transform.copy(), 0.0, np.zeros(n)

    monkeypatch.setattr("src.compute_helpers.compute_rigid_transform", _mock)
    return _mock


@pytest.fixture
def mock_compute_transform_failure(monkeypatch):
    def _mock(src, dst):
        raise ValueError("Mocked transform failure")

    monkeypatch.setattr("src.compute_helpers.compute_rigid_transform", _mock)
    return _mock


@pytest.fixture
def mock_validate_success(monkeypatch):
    def _mock(T):
        return True, "OK"

    monkeypatch.setattr("src.compute_helpers.validate_transform", _mock)
    return _mock


@pytest.fixture
def mock_validate_failure(monkeypatch):
    def _mock(T):
        return False, "Invalid transform"

    monkeypatch.setattr("src.compute_helpers.validate_transform", _mock)
    return _mock


@pytest.fixture
def mock_landmarks_success(monkeypatch):
    def _mock(landmarks, marked_points):
        return np.asarray(marked_points)  # passthrough

    monkeypatch.setattr(
        "src.compute_helpers.obtain_true_landmarks_from_marked_points", _mock
    )
    return _mock


@pytest.fixture
def mock_landmarks_failure(monkeypatch):
    def _mock(*args, **kwargs):
        raise ValueError("Landmark extraction failed")

    monkeypatch.setattr(
        "src.compute_helpers.obtain_true_landmarks_from_marked_points", _mock
    )
    return _mock


def assert_no_transform_set(gui, attr_name):
    assert not hasattr(gui, attr_name) or getattr(gui, attr_name) is None


def assert_transform_set(gui, attr_name):
    assert hasattr(gui, attr_name)
    T = getattr(gui, attr_name)
    assert isinstance(T, np.ndarray)
    assert T.shape == (4, 4)


def assert_success(result, gui, attr_name):
    assert result is True
    assert_transform_set(gui, attr_name)


def assert_failure(result, gui, attr_name):
    assert result is False
    assert_no_transform_set(gui, attr_name)


@pytest.fixture
def gui_with_aux_landmarks(gui_with_valid_aux_points):
    gui_with_valid_aux_points.aux_dev_landmarks = {
        "A": np.array([0, 0, 0]),
        "B": np.array([1, 0, 0]),
        "C": np.array([0, 1, 0]),
    }
    return gui_with_valid_aux_points


@pytest.fixture
def gui_with_helmet_landmarks(gui_with_valid_sensor_points):
    gui_with_valid_sensor_points.helmet_landmark_positions = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
    ]
    return gui_with_valid_sensor_points


class FakeVTKPoints:
    def __init__(self, n_points=0, point=(0.0, 0.0, 0.0)):
        self.n_points = n_points
        self.point = point

    def GetNumberOfPoints(self):
        return self.n_points

    def GetPoint(self, i):
        return self.point


class FakePolyData:
    def __init__(self, n_points=0):
        self.points = FakeVTKPoints(n_points=n_points)

    def GetPoints(self):
        return self.points

    def GetPoint(self, idx):
        return (0.0, 0.0, 0.0)


class FakeIdArray:
    def __init__(self, n=0):
        self._n = n

    def InsertNextValue(self, val):
        self._n += 1

    def GetNumberOfTuples(self):
        return self._n


class FakeMesh:
    def __init__(self, n_cells=1000):
        self.n_cells = n_cells

    def clip_surface(self, sphere):
        return FakeMesh(n_cells=self.n_cells)

    def clip(self, normal=None, origin=None, invert=False):
        return FakeMesh(n_cells=self.n_cells)


@pytest.fixture
def mock_pyvista(monkeypatch):
    import types

    fake_pv = types.SimpleNamespace()

    def fake_wrap(polydata):
        return FakeMesh(n_cells=1000)

    def fake_sphere(*args, **kwargs):
        return object()

    fake_pv.wrap = fake_wrap
    fake_pv.Sphere = fake_sphere

    monkeypatch.setattr("src.compute_helpers.pv", fake_pv)


@pytest.fixture
def mock_pyvista_zero_cells(monkeypatch):
    import types

    fake_pv = types.SimpleNamespace()

    def fake_wrap(polydata):
        return FakeMesh(n_cells=0)  # triggers early return

    def fake_sphere(*args, **kwargs):
        return object()

    fake_pv.wrap = fake_wrap
    fake_pv.Sphere = fake_sphere

    monkeypatch.setattr("src.compute_helpers.pv", fake_pv)


@pytest.fixture
def mock_pyvista_small_cells(monkeypatch):
    import types

    fake_pv = types.SimpleNamespace()

    def fake_wrap(polydata):
        return FakeMesh(n_cells=10)  # below threshold

    def fake_sphere(*args, **kwargs):
        return object()

    fake_pv.wrap = fake_wrap
    fake_pv.Sphere = fake_sphere

    monkeypatch.setattr("src.compute_helpers.pv", fake_pv)


class FakeDelaunay:
    def __init__(self, points):
        pass

    def find_simplex(self, p):
        return 1  # always inside


@pytest.fixture
def mock_delaunay_failure(monkeypatch):
    def _fail(*args, **kwargs):
        raise Exception("Delaunay failed")

    monkeypatch.setattr("src.compute_helpers.Delaunay", _fail)


@pytest.fixture
def mock_delaunay_success(monkeypatch):
    monkeypatch.setattr("src.compute_helpers.Delaunay", FakeDelaunay)


@pytest.fixture
def mock_vtk_minimal(monkeypatch):
    import types

    vtk_mock = types.SimpleNamespace()

    vtk_mock.vtkIdTypeArray = lambda: FakeIdArray(0)

    vtk_mock.vtkSelectionNode = lambda: types.SimpleNamespace(
        SetFieldType=lambda *a: None,
        SetContentType=lambda *a: None,
        SetSelectionList=lambda *a: None,
    )

    vtk_mock.vtkSelection = lambda: types.SimpleNamespace(AddNode=lambda *a: None)

    vtk_mock.vtkExtractSelection = lambda: types.SimpleNamespace(
        SetInputData=lambda *a: None,
        Update=lambda: None,
        GetOutputPort=lambda: None,
    )

    vtk_mock.vtkGeometryFilter = lambda: types.SimpleNamespace(
        SetInputConnection=lambda *a: None,
        Update=lambda: None,
        GetOutput=lambda: None,
    )

    monkeypatch.setattr("src.compute_helpers.vtk", vtk_mock)


@pytest.fixture
def fake_polydata_empty():
    return FakePolyData(n_points=0)


@pytest.fixture
def fake_polydata_small():
    return FakePolyData(n_points=10)


@pytest.fixture
def fake_polydata_large():
    return FakePolyData(n_points=200)


@pytest.fixture
def valid_fiducials():
    return {
        "NAS": np.array([0, 0, 0]),
        "LPA": np.array([1, 0, 0]),
        "RPA": np.array([0, 1, 0]),
        "NOSE": np.array([0, 0.5, 0]),
        "CHIN": np.array([0, -0.5, 0]),
    }


@pytest.fixture
def incomplete_fiducials():
    return {"NAS": np.array([0, 0, 0])}


@pytest.fixture
def mock_extract_face_roi_success(monkeypatch):
    def _mock(*args, **kwargs):
        return object()  # dummy ROI

    monkeypatch.setattr("src.compute_helpers.extract_face_roi", _mock)


@pytest.fixture
def mock_extract_face_roi_failure(monkeypatch):
    def _mock(*args, **kwargs):
        return None

    monkeypatch.setattr("src.compute_helpers.extract_face_roi", _mock)


@pytest.fixture
def mock_extract_convex_success(monkeypatch):
    monkeypatch.setattr(
        "src.compute_helpers.extract_convex_hull_roi", lambda *a, **k: object()
    )


@pytest.fixture
def mock_extract_convex_failure(monkeypatch):
    monkeypatch.setattr(
        "src.compute_helpers.extract_convex_hull_roi", lambda *a, **k: None
    )


@pytest.fixture
def gui_with_head_polydata(gui, fake_polydata_large, valid_fiducials):
    gui.head_polydata = fake_polydata_large
    gui.marked_fiducials_on_head_scan = valid_fiducials
    return gui


@pytest.fixture
def gui_with_irregular_points(gui, fake_polydata_large):
    gui.head_polydata = fake_polydata_large
    gui.head_alignment_points_from_source = [
        np.array([0, 0, 0]),
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]
    return gui


# ==========================================================
# GUI fixtures for ICP tests
# ==========================================================


@pytest.fixture
def gui_empty(gui):
    """All attributes missing / None."""
    gui.T_head_to_mri = None
    gui.mri_polydata = None
    gui.face_roi_in_head_frame = None
    gui.irregular_roi_in_head_frame = None
    gui.T_head_to_scan = None
    gui.master_scan_polydata = None
    return gui


@pytest.fixture
def gui_with_mri_only(gui):
    gui.T_head_to_mri = None
    gui.mri_polydata = object()
    return gui


@pytest.fixture
def gui_with_transform_only(gui):
    gui.T_head_to_mri = np.eye(4)
    gui.mri_polydata = None
    return gui


@pytest.fixture
def gui_with_no_roi(gui):
    gui.T_head_to_mri = np.eye(4)
    gui.mri_polydata = object()
    gui.face_roi_in_head_frame = None
    gui.irregular_roi_in_head_frame = None
    return gui


@pytest.fixture
def gui_master_scan_missing(gui):
    gui.T_head_to_scan = None
    gui.master_scan_polydata = None
    gui.irregular_roi_in_head_frame = None
    return gui


@pytest.fixture
def gui_master_with_transform_only(gui):
    gui.T_head_to_scan = np.eye(4)
    gui.master_scan_polydata = None
    return gui


@pytest.fixture
def gui_master_no_roi(gui):
    gui.T_head_to_scan = np.eye(4)
    gui.master_scan_polydata = object()
    gui.irregular_roi_in_head_frame = None
    return gui


# ==========================================================
# Minimal safe mocks so deep pipeline doesn't crash
# (only used when needed)
# ==========================================================


@pytest.fixture
def mock_vtk_safe(monkeypatch):
    """Minimal VTK mocks to avoid crashes if execution reaches deep code."""

    import types

    vtk_mock = types.SimpleNamespace()

    vtk_mock.vtkTransform = lambda: types.SimpleNamespace(SetMatrix=lambda *a: None)

    vtk_mock.vtkTransformPolyDataFilter = lambda: types.SimpleNamespace(
        SetTransform=lambda *a: None,
        SetInputData=lambda *a: None,
        Update=lambda: None,
        GetOutput=lambda: object(),
    )

    vtk_mock.vtkIterativeClosestPointTransform = lambda: types.SimpleNamespace(
        SetSource=lambda *a: None,
        SetTarget=lambda *a: None,
        GetLandmarkTransform=lambda: types.SimpleNamespace(
            SetModeToRigidBody=lambda: None
        ),
        SetMaximumNumberOfIterations=lambda *a: None,
        StartByMatchingCentroidsOff=lambda: None,
        Modified=lambda: None,
        Update=lambda: None,
        GetMatrix=lambda: types.SimpleNamespace(
            GetElement=lambda r, c: 1.0 if r == c else 0.0
        ),
    )

    monkeypatch.setattr("src.compute_helpers.vtk", vtk_mock)


@pytest.fixture
def mock_surface_rms(monkeypatch):
    monkeypatch.setattr(
        "src.compute_helpers.compute_surface_rms",
        lambda *a, **k: 1.0,
    )


@pytest.fixture
def mock_vtk_matrix(monkeypatch):
    monkeypatch.setattr(
        "src.compute_helpers.vtk_matrix_from_numpy",
        lambda x: None,
    )
