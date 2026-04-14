from pathlib import Path
import os


class ProjectPaths:
    def __init__(self):
        # Priority: env var > marker-based > fallback
        self.root_dir = Path(
            os.environ.get("OPM_GUI_PROJECT_ROOT", self._find_root_with_marker())
        )

    @staticmethod
    def _find_root_with_marker():
        """Find root by looking for common markers."""
        current = Path(__file__).parent
        for parent in [current] + list(current.parents):
            if (parent / ".project_root").exists():  # your custom marker
                return parent
            if (parent / ".git").exists():
                return parent
        return current.parent  # fallback

    def get_results_dir(self):
        """Create and get results directory."""
        results_dir = self.root_dir / "results"
        results_dir.mkdir(exist_ok=True)
        return results_dir

    def get_logs_dir(self):
        """Create and get logs directory."""
        logs_dir = self.root_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        return logs_dir
