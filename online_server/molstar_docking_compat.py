import os
import re
import shutil
import hashlib
from pathlib import Path

import streamlit.components.v1 as components

try:
    import streamlit_molstar.docking as _orig_docking
except Exception as _e:
    raise RuntimeError(f"streamlit_molstar.docking import failed: {_e}")


_COMPAT_STATUS = {}
_PATCH_VERSION = "20260304v3"


def _get_file_type(file_path: str) -> str:
    return os.path.splitext(file_path)[1][1:].lower()


def _hash_main_chunks(js_dir: Path) -> str:
    h = hashlib.sha256()
    mains = sorted(js_dir.glob("main.*.chunk.js"))
    for f in mains:
        h.update(f.name.encode("utf-8"))
        h.update(f.read_bytes())
    return h.hexdigest()


def _prepare_patched_build_dir() -> tuple[Path, str]:
    src_build = Path(_orig_docking.__file__).resolve().parent / "frontend" / "build"
    if not src_build.exists():
        raise RuntimeError(f"Molstar docking frontend build not found: {src_build}")

    src_js_dir = src_build / "static" / "js"
    src_hash = _hash_main_chunks(src_js_dir)
    cache_tag = f"{_PATCH_VERSION}_{src_hash[:10]}"

    dst_root = Path(__file__).resolve().parent / ".molstar_docking_compat"
    dst_build = dst_root / f"build_{cache_tag}"
    dst_js_dir = dst_build / "static" / "js"
    dst_root.mkdir(parents=True, exist_ok=True)

    if not dst_build.exists():
        shutil.copytree(src_build, dst_build)

    create_marker = (
        "allowMajorPerformanceCaveat:!0,preferWebgl1:!0,"
        "disableAntialiasing:!0,enableWboit:!1,enableDpoit:!1,codexCompat:!0"
    )
    strict_opts = (
        "{layoutIsExpanded:!0,layoutShowControls:!0,viewportShowControls:!0,"
        "viewportShowAnimation:!1,allowMajorPerformanceCaveat:!0,preferWebgl1:!0,"
        "disableAntialiasing:!0,enableWboit:!1,enableDpoit:!1,codexCompat:!0}"
    )

    style_old = 'style:{position:"absolute",width:"100%",height:this.props.height,overflow:"hidden"}'
    style_new = (
        'style:{position:"relative",width:"100%",height:this.props.height,'
        'minHeight:this.props.height,overflow:"hidden"}'
    )

    patched_files = []
    already_patched = []
    for js_file in sorted(dst_js_dir.glob("main.*.chunk.js")):
        try:
            text = js_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Force more permissive WebGL init:
        # 1) allow major performance caveat (software/blacklist fallback)
        # 2) prefer WebGL1 (improves compatibility on older/quirky drivers)
        # 3) disable antialiasing and advanced OIT paths to reduce context init pressure
        # 4) avoid absolute root container in iframe to reduce mount false-negatives
        new_text, n = re.subn(
            r"Se\.create\(this\.parentRef,\{[^{}]*layoutIsExpanded:!1[^{}]*viewportShowAnimation:!1[^{}]*\}\)",
            f"Se.create(this.parentRef,{strict_opts})",
            text,
            count=1,
        )

        style_hits = 0
        if style_old in new_text:
            new_text = new_text.replace(style_old, style_new)
            style_hits = 1

        # Keep Molstar control tree visible by default so interaction visibility toggles are accessible.
        control_hits = 0
        if "layoutShowControls:!1" in new_text:
            new_text = new_text.replace("layoutShowControls:!1", "layoutShowControls:!0")
            control_hits += 1
        if "layoutIsExpanded:!1" in new_text:
            new_text = new_text.replace("layoutIsExpanded:!1", "layoutIsExpanded:!0")
            control_hits += 1

        if (n > 0) or (style_hits > 0) or (control_hits > 0):
            js_file.write_text(new_text, encoding="utf-8")
            patched_files.append(js_file.name)
        elif create_marker in text:
            already_patched.append(js_file.name)

    if not patched_files and not already_patched:
        raise RuntimeError(
            "Failed to patch Molstar docking frontend. Expected main.*.chunk.js signature was not found."
        )

    _COMPAT_STATUS.update(
        {
            "patch_version": _PATCH_VERSION,
            "source_build_dir": str(src_build),
            "compat_build_dir": str(dst_build),
            "source_hash": src_hash,
            "cache_tag": cache_tag,
            "patched_files": patched_files,
            "already_patched_files": already_patched,
            "component_name": f"molstar_component_docking_compat_{cache_tag}",
            "create_marker_found": bool(patched_files or already_patched),
        }
    )
    return dst_build, cache_tag


_build_dir, _cache_tag = _prepare_patched_build_dir()
_component_func_docking_compat = components.declare_component(
    f"molstar_component_docking_compat_{_cache_tag}", path=str(_build_dir)
)


def st_molstar_docking(
    receptor_file_path,
    ligand_file_path,
    *,
    gt_ligand_file_path=None,
    gt_ligand_file_paths=None,
    options=None,
    height="240px",
    key=None,
):
    with open(receptor_file_path) as f:
        receptor_file_content = f.read()
        receptor_file_format = _get_file_type(receptor_file_path)
    with open(ligand_file_path) as f:
        ligand_file_content = f.read()
        ligand_file_format = _get_file_type(ligand_file_path)

    if gt_ligand_file_path:
        with open(gt_ligand_file_path) as f:
            gt_ligand_file_content = f.read()
            gt_ligand_file_format = _get_file_type(gt_ligand_file_path)
    else:
        gt_ligand_file_content = None
        gt_ligand_file_format = None

    gt_ligand_file_contents = []
    gt_ligand_file_formats = []
    if gt_ligand_file_paths:
        for p in gt_ligand_file_paths:
            with open(p) as f:
                tmp_file_content = f.read()
                tmp_file_format = _get_file_type(p)
                gt_ligand_file_contents.append(tmp_file_content)
                gt_ligand_file_formats.append(tmp_file_format)

    params = {
        "scene": "docking",
        "height": height,
        "receptorFile": {
            "data": "<placeholder>",
            "format": receptor_file_format,
        },
        "receptorFile_data": receptor_file_content,
        "ligandFile": {
            "data": "<placeholder>",
            "format": ligand_file_format,
        },
        "ligandFile_data": ligand_file_content,
    }
    if gt_ligand_file_content:
        params.update(
            {
                "gtLigandFile": {
                    "data": "<placeholder>",
                    "format": gt_ligand_file_format,
                },
                "gtLigandFile_data": gt_ligand_file_content,
            }
        )
    if gt_ligand_file_contents and gt_ligand_file_formats and len(gt_ligand_file_contents) == len(gt_ligand_file_formats):
        gt_ligand_files = []
        for i in range(len(gt_ligand_file_contents)):
            gt_ligand_files.append(
                {
                    "file": {
                        "data": gt_ligand_file_contents[i],
                        "format": gt_ligand_file_formats[i],
                    },
                }
            )
        params.update({"gtLigandFiles": gt_ligand_files})
    if options:
        params.update({"options": options})

    _component_func_docking_compat(key=key, default=None, **params)


def get_compat_status():
    return dict(_COMPAT_STATUS)
