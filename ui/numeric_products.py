"""Download descriptors and session-confined paths for numeric products."""
from pathlib import Path

PRODUCT_SUFFIXES = frozenset({'.png', '.svg', '.pdf', '.csv', '.json'})


def product_path(engine, artifact_id):
    artifact = engine.numeric_artifacts.get(artifact_id)
    if artifact is None or not artifact.path or artifact.artifact_type not in ('figure','table','log'):
        raise ValueError('unknown numeric product')
    path = Path(artifact.path).resolve()
    path.relative_to(engine._numeric_output_root().resolve())
    if path.suffix not in PRODUCT_SUFFIXES or not path.is_file():
        raise ValueError('unavailable numeric product')
    return path


def numeric_downloads(engine, artifact_ids):
    downloads = []
    for aid in artifact_ids:
        try:
            path = product_path(engine, aid)
        except (ValueError, OSError):
            continue
        downloads.append({'artifact_id':aid, 'name':path.name})
    return downloads
