import os
import joblib
import gcsfs
import logging
from typing import Any

logger = logging.getLogger(__name__)

def is_gcs_path(path: str) -> bool:
    """Check if a path is a Google Cloud Storage path."""
    return path.startswith("gs://")

def ensure_dir(dir_path: str):
    """Ensure the directory exists, bypassing for GCS paths."""
    if not is_gcs_path(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def exists(path: str) -> bool:
    """Check if a file exists, handles both local and GCS paths."""
    if is_gcs_path(path):
        fs = gcsfs.GCSFileSystem()
        return fs.exists(path)
    else:
        return os.path.exists(path)

def dirname(path: str) -> str:
    """Safely get the directory name from a path."""
    if is_gcs_path(path):
        return "/".join(path.split("/")[:-1])
    else:
        return os.path.dirname(path)

def join_path(base_path: str, *paths: str) -> str:
    """Safely join paths for both local OS and GCS urls."""
    if is_gcs_path(base_path):
        # Remove trailing slashes and join with /
        result = base_path.rstrip('/')
        for p in paths:
            result = f"{result}/{p.lstrip('/')}"
        return result
    else:
        return os.path.join(base_path, *paths)

def save_artifact(obj: Any, path: str):
    """
    Save an object using joblib. Handles both local and GCS paths.
    
    Args:
        obj: The python object to save (e.g., model, scaler, encoders)
        path: Local or gs:// path
    """
    if is_gcs_path(path):
        logger.info(f"Saving artifact to GCS: {path}")
        fs = gcsfs.GCSFileSystem()
        # gcsfs will automatically create parent directories in the bucket
        with fs.open(path, 'wb') as f:
            joblib.dump(obj, f)
    else:
        logger.info(f"Saving artifact locally: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(obj, path)

def load_artifact(path: str) -> Any:
    """
    Load an object using joblib. Handles both local and GCS paths.
    
    Args:
        path: Local or gs:// path
        
    Returns:
        The loaded python object
    """
    if is_gcs_path(path):
        logger.info(f"Loading artifact from GCS: {path}")
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, 'rb') as f:
            return joblib.load(f)
    else:
        logger.info(f"Loading artifact locally: {path}")
        return joblib.load(path)
