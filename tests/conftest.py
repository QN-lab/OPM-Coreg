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

    monkeypatch.setattr("src.compute_helpers.obtain_true_landmarks_from_marked_points", _mock)
    return _mock


@pytest.fixture
def mock_landmarks_failure(monkeypatch):
    def _mock(*args, **kwargs):
        raise ValueError("Landmark extraction failed")

    monkeypatch.setattr("src.compute_helpers.obtain_true_landmarks_from_marked_points", _mock)
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
