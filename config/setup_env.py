import sys, os


def setup():
    """Return (PROJECT_ROOT, DATA_DIR, DEVICE) for the current environment."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir     = os.path.join(project_root, 'pack', 'processed')

    for d in ['config', 'src']:
        p = os.path.join(project_root, d)
        if p not in sys.path:
            sys.path.insert(0, p)

    device = _detect_device()
    return project_root, data_dir, device


def _detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'
