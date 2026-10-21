"""The results gallery (settings.RESULTS_GALLERY_DIR, on the results drive) shown inside the PiGallery2 media folder as one junction.

Publishing hardlinks a render into the gallery, and a hardlink cannot cross drives: so new results are published into the junction path
(settings.gallery_results_mount()), which writes them on the results drive next to their renders, while everything already in the
gallery stays where it is.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import settings


class GalleryMountError(RuntimeError):
    """The mount path exists but is not a junction to the results gallery."""


def _create_junction(target: str, link: str) -> None:
    import _winapi

    _winapi.CreateJunction(target, link)


def ensure_results_mount(gallery_dir: Path = settings.GALLERY_DIR, target: Path = settings.RESULTS_GALLERY_DIR,
                         name: str = settings.GALLERY_RESULTS_MOUNT, create=_create_junction) -> Path:
    """Return the mount path, creating the results gallery folder and the junction to it when missing."""
    target, mount = Path(target), Path(gallery_dir) / name
    target.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(mount):
        if not os.path.isjunction(mount):
            raise GalleryMountError(f"{mount} exists and is not a junction: refusing to publish into it")
        if os.path.normcase(os.path.realpath(mount)) != os.path.normcase(os.path.realpath(target)):
            raise GalleryMountError(f"{mount} is a junction to {os.path.realpath(mount)}, not to {target}")
        return mount
    create(str(target), str(mount))
    return mount
