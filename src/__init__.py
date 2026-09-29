"""Customer segmentation with K-means (Prodigy InfoTech ML Task 02).

The package is deliberately small and explicit:

* :mod:`src.config`        - paths, feature names, units and constants
* :mod:`src.data`          - loading, validating and cleaning the CSV
* :mod:`src.clustering`    - K-means evaluation, k selection, cluster summaries
* :mod:`src.train`         - training entry point that writes every artifact
* :mod:`src.predict`       - loading artifacts and assigning a profile
* :mod:`src.visualization` - all matplotlib figures
"""

__all__ = ["clustering", "config", "data", "predict", "train", "visualization"]
