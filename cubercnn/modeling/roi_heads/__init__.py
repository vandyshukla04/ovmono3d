from .roi_heads import *

# roi_heads_gdino depends on groundingdino, which has a CUDA-kernel build
# dependency. For WildBox training/eval we use TEST.ORACLE2D=True and never
# invoke GroundingDINO, so skip the import if the package isn't installed.
try:
    from .roi_heads_gdino import *
except ImportError as _e:
    import warnings
    warnings.warn(
        f"GroundingDINO-based ROI head not available ({_e}); "
        f"oracle-2D and open-vocab 2D paths will fail if used, but "
        f"standard training/eval will still work."
    )
