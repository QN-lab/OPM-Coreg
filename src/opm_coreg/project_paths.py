from pathlib import Path
import os


class ProjectPaths:
    def __init__(self):
        # Priority: env var > Path.home()/Documents/opm-coreg-gui > Path.home()/opm-coreg-gui > empty string
        env_root = os.environ.get("OPM_COREG_GUI_ROOT_DIR", "")
        if env_root:
            self.root_dir = Path(env_root)
        else:
            print("OPM_COREG_GUI_ROOT_DIR environment variable not set. Using Home/Documents/opm-coreg-gui or Home/opm-coreg-gui as fallback.")
            if (Path.home() / "Documents").exists():
                self.root_dir = Path.home() / "Documents" / "opm-coreg-gui"
            else:
                self.root_dir = Path.home() / "opm-coreg-gui"
            try:
                self.root_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                # Raise an exception and close the application if the directory cannot be created
                raise Exception(
                    f"Failed to create root directory at {self.root_dir}. Please check permissions and available disk space. Error: {e}"
                )

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
