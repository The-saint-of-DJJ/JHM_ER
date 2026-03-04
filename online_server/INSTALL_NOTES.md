# Install Notes

## 1) Core install (pip)

```bash
conda activate JHM_ER
pip install -r online_server/requirements.txt
```

## 2) Optional chemistry CLI tools (recommended via conda-forge)

`plip` and `openbabel` are not included in `requirements.txt` on purpose.
Installing them via pip may trigger source builds and fail on SWIG/OpenBabel headers.

```bash
conda activate JHM_ER
conda install -c conda-forge openbabel plip
```

## 3) Why this split

- `openbabel` (pip) often tries to compile native extensions.
- This project can run core prediction/docking UI without them.
- If missing, the app will show warnings and disable related optional analysis paths.
