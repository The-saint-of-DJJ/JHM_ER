# ==== 🎯 完整综合分析平台：预测 + 对接 + 毒性 + AI对话 + 可视化 ====
"""
集成功能：
1. 化合物输入与靶点/毒性预测
2. 分子对接（可选择受体结构）
3. 毒性注释（PubChem查询）
4. AI对话窗口（DeepSeek毒性解释）
5. 3D相互作用可视化
"""

import time, json, subprocess, shutil, os, re, html, math, hashlib, random
from dataclasses import dataclass
import numpy as np, pandas as pd
import torch, torch.nn as nn
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
import requests
import urllib3
from urllib.parse import quote
# RDKit drawing
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Geometry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========== 配置 ==========
def _detect_base_dir():
    # Keep paths relative to project root; try CWD and this file's location.
    here = Path(__file__).resolve().parent
    candidates = [
        Path("."),
        Path(".."),
        here,
        here.parent,
    ]
    for cand in candidates:
        try:
            if (cand / "Output").exists():
                return cand.resolve()
        except Exception:
            continue
    # Fallback to repository root near this file if available.
    if (here.parent / "Output").exists():
        return here.parent.resolve()
    return Path(".")

def _pick_existing_path(*paths: Path):
    for p in paths:
        try:
            if p and p.exists():
                return p
        except Exception:
            continue
    return None

def _cmd_exists(cmd: str) -> bool:
    if not cmd:
        return False
    try:
        p = Path(cmd)
        if p.exists():
            return True
    except Exception:
        pass
    return shutil.which(cmd) is not None

def _find_plip_cmd() -> str:
    env_cmd = os.getenv("PLIP_CMD")
    if env_cmd:
        try:
            if Path(env_cmd).exists():
                return env_cmd
        except Exception:
            pass
    if shutil.which("plip"):
        return "plip"
    # Probe common conda env locations
    home = Path.home()
    for base in [home / "miniconda3" / "envs", home / "anaconda3" / "envs"]:
        try:
            if not base.exists():
                continue
            for p in base.glob("*/bin/plip"):
                if p.exists():
                    return str(p)
        except Exception:
            continue
    return "plip"


class Config:
    # Use relative paths from the current working directory
    BASE_DIR = _detect_base_dir()
    OUTPUT_DIR = BASE_DIR / 'Output'
    RECEPTOR_DIR = BASE_DIR / 'Structures' / 'Receptors'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # External tools: prefer PATH/env vars; avoid hard-coded absolute paths.
    TOOLS_DIR = _pick_existing_path(BASE_DIR / "Tools", BASE_DIR / "tools") or (BASE_DIR / "Tools")

    _VINA_WORKDIR = _pick_existing_path(BASE_DIR / "vina_1.2.7_linux_x86_64")
    _VINA_LOCAL = _pick_existing_path(
        TOOLS_DIR / "Docking" / "vina",
        TOOLS_DIR / "vina",
    )
    VINA = os.getenv("VINA_CMD") or (str(_VINA_WORKDIR) if _VINA_WORKDIR else (str(_VINA_LOCAL) if _VINA_LOCAL else "vina"))

    _ADFR_BIN_ABS = _pick_existing_path(Path("/home/xuchengjie/Program/ADFRsuite-1.0/bin"))
    _ADFR_BIN_LOCAL = _pick_existing_path(
        TOOLS_DIR / "ADFRsuite-1.0" / "bin",
        TOOLS_DIR / "ADFRsuite" / "bin",
    )
    _adfr_bin = _ADFR_BIN_ABS or _ADFR_BIN_LOCAL
    _prep_rec = (_adfr_bin / "prepare_receptor") if _adfr_bin else None
    _prep_lig = (_adfr_bin / "prepare_ligand") if _adfr_bin else None
    PREP_REC = os.getenv("PREP_REC_CMD") or (str(_prep_rec) if _prep_rec and _prep_rec.exists() else "prepare_receptor")
    PREP_LIG = os.getenv("PREP_LIG_CMD") or (str(_prep_lig) if _prep_lig and _prep_lig.exists() else "prepare_ligand")
    # New main models (trained in notebook)
    MODEL_DIR = OUTPUT_DIR / 'notebook_Target_Prediction_ER' / 'models'
    MODEL_TARGET = MODEL_DIR / 'main_target_classifier.joblib'
    MODEL_REG = MODEL_DIR / 'main_ic50_regressor.joblib'
    # Feature flags must match training-time config
    # Must match training-time feature flags (model expects 2066 dims = 2048 + 10 + 8)
    USE_PHYCHEM_FEATURES = True
    USE_DOCKING_FEATURES = False
    USE_3D_DESCRIPTORS = True
    DOCK_IMPUTE_VALUE = 0.0
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    # Default docking box (can be overridden in UI)
    CENTER_X = 0.0
    CENTER_Y = 0.0
    CENTER_Z = 0.0
    SIZE_X = 20.0
    SIZE_Y = 20.0
    SIZE_Z = 20.0
    EXHAUSTIVENESS = 8
    PLIP_CMD = _find_plip_cmd()


def check_docking_tools(lang: str = "中文"):
    """Check external docking dependencies (vina + preparation tools)."""
    is_zh = _is_zh_lang(lang)
    missing = []

    # 1) Vina is mandatory
    if not _cmd_exists(Config.VINA):
        missing.append(f"vina (VINA_CMD={Config.VINA})")

    # 2) For ligand/receptor prep, we can fallback to obabel
    obabel_ok = shutil.which("obabel") is not None

    rec_ok = _cmd_exists(Config.PREP_REC) or obabel_ok
    lig_ok = _cmd_exists(Config.PREP_LIG) or obabel_ok

    if not rec_ok:
        missing.append(
            f"prepare_receptor (PREP_REC_CMD={Config.PREP_REC}) {'或' if is_zh else 'or'} obabel"
        )
    if not lig_ok:
        missing.append(
            f"prepare_ligand (PREP_LIG_CMD={Config.PREP_LIG}) {'或' if is_zh else 'or'} obabel"
        )

    if missing:
        if is_zh:
            msg = "Docking 依赖缺失: " + "；".join(missing) + "。\n"
            msg += (
                "请安装/配置 AutoDock Vina 与 (ADFRsuite 或 OpenBabel)。也可设置环境变量: "
                "VINA_CMD, PREP_REC_CMD, PREP_LIG_CMD（推荐使用相对路径，例如 Tools/vina）。"
            )
        else:
            msg = "Docking dependencies missing: " + "; ".join(missing) + ".\n"
            msg += (
                "Install/configure AutoDock Vina and (ADFRsuite or OpenBabel). You can also set environment "
                "variables: VINA_CMD, PREP_REC_CMD, PREP_LIG_CMD (recommended to use relative paths, e.g., Tools/vina)."
            )
        return False, msg

    return True, "OK"

# ========== 分子特征计算 (恢复) ==========
from rdkit.Chem import rdMolDescriptors

# === Toxic functional group alerts (order aligns with DESC_NAMES) ===
TOXIC_ALERTS = [
    ("alert_nitro_aromatic", "[c][N+](=O)[O-]", "芳香硝基"),
    ("alert_nitro_aliphatic", "[CX4][N+](=O)[O-]", "脂肪族硝基"),
    ("alert_azide", "N=[N+]=N", "叠氮"),
    ("alert_aniline", "cN", "苯胺"),
    ("alert_anilide", "Nc1ccc(cc1)C=O", "苯胺酰胺"),
    ("alert_isocyanate", "N=C=O", "异氰酸酯"),
    ("alert_michael_acceptor", "[CH2]=[CH]-C(=O)[#6]", "Michael 受体"),
    ("alert_epoxide", "C1OC1", "环氧"),
    ("alert_organohalide", "[CX4]([F,Cl,Br,I])([F,Cl,Br,I])[F,Cl,Br,I]", "多卤代"),
    ("alert_quinone", "O=C1C=CC(=O)C=C1", "醌"),
    ("alert_thioamide", "NC(=S)", "硫酰胺"),
    ("alert_hydrazine", "NN", "肼/肼类"),
]

def simple_atom_features(atom):
    return np.array([
        atom.GetAtomicNum() / 100.0, atom.GetDegree() / 5.0,
        atom.GetFormalCharge() / 5.0, atom.GetTotalValence() / 6.0,
        atom.GetTotalNumHs(includeNeighbors=True) / 4.0,
        float(atom.GetIsAromatic()), float(atom.IsInRing()),
        atom.GetMass() / 200.0,
        float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D),
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D2),
        1.0
    ], dtype=np.float32)

def mol_to_simple_graph(mol, max_nodes=200):
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0 or n_atoms > max_nodes:
        return None
    node_feats = np.vstack([simple_atom_features(atom) for atom in mol.GetAtoms()]).astype(np.float32)
    edge_types = np.zeros((n_atoms, n_atoms), dtype=np.uint8)
    for idx in range(n_atoms):
        edge_types[idx, idx] = 0
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 4
    }
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_id = bond_type_map.get(bond.GetBondType(), 1)
        edge_types[i, j] = edge_id
        edge_types[j, i] = edge_id
    return {'node_feats': node_feats, 'edge_types': edge_types, 'n_nodes': n_atoms}

def compute_descriptors(mol):
    if mol is None:
        return np.zeros(24, dtype=np.float32)
    try:
        phys = [
            Descriptors.ExactMolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol),
            rdMolDescriptors.CalcLabuteASA(mol),
            Descriptors.HeavyAtomCount(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol)
        ]
        
        toxic_vals = []
        for _name, smarts, _label in TOXIC_ALERTS:
            patt = Chem.MolFromSmarts(smarts)
            if patt:
                toxic_vals.append(float(len(mol.GetSubstructMatches(patt))))
            else:
                toxic_vals.append(0.0)

        return np.array(phys + toxic_vals, dtype=np.float32)

    except Exception:
        return np.zeros(24, dtype=np.float32)

def compute_fingerprint(mol, n_bits=2048, radius=2):
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((1, n_bits), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr[0])
        return arr[0].astype(np.float32)
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def detect_toxic_alerts(mol):
    """Return matched toxic functional groups with counts."""
    if mol is None:
        return []
    alerts = []
    for name, smarts, label in TOXIC_ALERTS:
        patt = Chem.MolFromSmarts(smarts)
        if not patt:
            continue
        n = len(mol.GetSubstructMatches(patt))
        if n > 0:
            alerts.append({
                'name': name,
                'label': label,
                'count': int(n),
                'smarts': smarts,
            })
    return alerts


ER_TARGETS = {"esr1", "esr2", "gper1"}

# Endocrine-disruption structural motifs (non-exhaustive).
# Format: (key, zh_label, en_label, smarts, score)
EDC_MOTIF_RULES = [
    (
        "phthalate_like",
        "邻苯二甲酸酯样",
        "Phthalate-like",
        "O=C(O[#6])c1ccccc1C(=O)O[#6]",
        2,
    ),
    (
        "paraben_like",
        "对羟基苯甲酸酯样",
        "Paraben-like",
        "[#6][OX2][C](=O)c1ccc([OX2H])cc1",
        1,
    ),
    (
        "benzophenone_like",
        "二苯甲酮样",
        "Benzophenone-like",
        "O=C(c1ccccc1)c1ccccc1",
        1,
    ),
    (
        "halogenated_phenol",
        "卤代酚样",
        "Halogenated phenol-like",
        "c1cc([F,Cl,Br,I])ccc1O",
        1,
    ),
    (
        "stilbene_like",
        "二苯乙烯样",
        "Stilbene-like",
        "c1ccc(/C=C/c2ccccc2)cc1",
        2,
    ),
]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def assess_toxicity_risk(smiles, pred_results, toxicity_results, ac50_info, lang: str = "中文"):
    """
    综合风险判定（用于替代仅靠结构警示的 Pass/Risk 逻辑）：
    1) 结构警示
    2) ER 相关靶点预测概率
    3) AC50 强度
    4) 内分泌干扰结构特征（如双酚样多酚结构）
    """
    is_zh = str(lang).strip() == "中文" or str(lang).lower().startswith("zh")
    score = 0
    basis = []
    flags = {
        "structural_alert": False,
        "er_activity_signal": False,
        "potency_signal": False,
        "edc_motif_signal": False,
    }

    tox_rows = toxicity_results if isinstance(toxicity_results, list) else []
    pred_rows = pred_results if isinstance(pred_results, list) else []
    ac50 = ac50_info if isinstance(ac50_info, dict) else {}

    # 1) 结构警示
    if tox_rows:
        flags["structural_alert"] = True
        score += 3
        labels = []
        for a in tox_rows:
            if not isinstance(a, dict):
                continue
            lb = str(a.get("Label", "") or a.get("Alert", "")).strip()
            if lb:
                labels.append(lb)
        labels = sorted(set(labels))
        if labels:
            show_labels = ", ".join(labels[:4])
            if len(labels) > 4:
                show_labels += " ..."
            basis.append(
                f"检测到结构警示: {show_labels}" if is_zh else f"Structural alerts detected: {show_labels}"
            )
        else:
            basis.append("检测到结构警示" if is_zh else "Structural alerts detected")

    # 2) ER 相关靶点信号：累积 ESR1/ESR2/GPER1 概率（避免仅看 Top1）
    er_prob_sum = 0.0
    top_target = ""
    top_prob = 0.0
    for i, row in enumerate(pred_rows):
        if not isinstance(row, dict):
            continue
        raw = str(row.get("Raw Target", row.get("Target Name", "")) or "").strip().lower()
        p = _safe_float(row.get("Probability", 0.0)) or 0.0
        if i == 0:
            top_target = raw
            top_prob = p
        if raw in ER_TARGETS:
            er_prob_sum += p

    if er_prob_sum >= 0.60:
        flags["er_activity_signal"] = True
        score += 3
        basis.append(
            f"ER相关活性概率高 (ΣP={er_prob_sum:.2f})"
            if is_zh else f"High ER-activity probability signal (sum P={er_prob_sum:.2f})"
        )
    elif er_prob_sum >= 0.40:
        flags["er_activity_signal"] = True
        score += 2
        basis.append(
            f"ER相关活性概率中等 (ΣP={er_prob_sum:.2f})"
            if is_zh else f"Moderate ER-activity probability signal (sum P={er_prob_sum:.2f})"
        )
    elif er_prob_sum >= 0.25:
        flags["er_activity_signal"] = True
        score += 1
        basis.append(
            f"存在ER相关活性迹象 (ΣP={er_prob_sum:.2f})"
            if is_zh else f"Weak ER-activity signal present (sum P={er_prob_sum:.2f})"
        )

    if top_target and top_target != "decoy":
        basis.append(
            f"Top1 靶点: {top_target.upper()} (P={top_prob:.2f})"
            if is_zh else f"Top-1 target: {top_target.upper()} (P={top_prob:.2f})"
        )

    # 3) AC50 强度信号
    ac50_nm = _safe_float(ac50.get("AC50_nM", ac50.get("IC50_nM")))
    if ac50_nm is not None and ac50_nm > 0:
        if ac50_nm <= 1e4:
            flags["potency_signal"] = True
            score += 3
            basis.append(
                f"预测活性较强 (AC50≈{ac50_nm:.3g} nM)"
                if is_zh else f"Predicted potent activity (AC50≈{ac50_nm:.3g} nM)"
            )
        elif ac50_nm <= 1e5:
            flags["potency_signal"] = True
            score += 2
            basis.append(
                f"预测活性中等 (AC50≈{ac50_nm:.3g} nM)"
                if is_zh else f"Predicted moderate activity (AC50≈{ac50_nm:.3g} nM)"
            )
        elif ac50_nm <= 1e6:
            flags["potency_signal"] = True
            score += 1
            basis.append(
                f"预测存在弱活性 (AC50≈{ac50_nm:.3g} nM)"
                if is_zh else f"Predicted weak activity (AC50≈{ac50_nm:.3g} nM)"
            )

    # 4) 内分泌干扰结构线索（解决 BPA 无结构警示却被判安全的问题）
    phenol_count = 0
    aromatic_rings = 0
    logp_val = None
    edc_hits = []
    edc_score = 0
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None:
            phenol_pat = Chem.MolFromSmarts("c[OX2H]")
            if phenol_pat is not None:
                phenol_count = len(mol.GetSubstructMatches(phenol_pat))
            aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
            logp_val = _safe_float(Descriptors.MolLogP(mol))

            # BPA/BPF/BPS-like broad signal
            if phenol_count >= 2 and aromatic_rings >= 2:
                edc_score = max(edc_score, 2)
                edc_hits.append("双酚/多酚样" if is_zh else "Bisphenol/polyphenol-like")

            # Additional EDC motifs
            for _k, zh_label, en_label, smarts, s in EDC_MOTIF_RULES:
                patt = Chem.MolFromSmarts(smarts)
                if patt is None:
                    continue
                if mol.HasSubstructMatch(patt):
                    edc_score = max(edc_score, int(s))
                    edc_hits.append(zh_label if is_zh else en_label)

            # Weak generic endocrine-like signal: phenolic aromatic + lipophilicity window
            if phenol_count >= 1 and aromatic_rings >= 1 and logp_val is not None and 2.0 <= logp_val <= 7.0:
                edc_score = max(edc_score, 1)
                edc_hits.append("酚性芳香+疏水性窗口" if is_zh else "Phenolic aromatic + lipophilicity window")
    except Exception:
        pass

    if edc_score > 0:
        flags["edc_motif_signal"] = True
        score += int(edc_score)
        if edc_hits:
            uniq = []
            for x in edc_hits:
                if x not in uniq:
                    uniq.append(x)
            show = ", ".join(uniq[:4]) + (" ..." if len(uniq) > 4 else "")
            basis.append(
                f"检测到内分泌干扰结构线索: {show}"
                if is_zh else f"Endocrine-disruption structural clues detected: {show}"
            )
        else:
            basis.append(
                "检测到内分泌干扰结构线索"
                if is_zh else "Endocrine-disruption structural clues detected"
            )

    is_risk = score >= 2
    if score >= 5:
        level = "high"
        status_text = "高风险 (High Risk)" if is_zh else "High Risk"
    elif is_risk:
        level = "concern"
        status_text = "潜在风险 (Potential Risk)" if is_zh else "Potential Risk"
    else:
        level = "pass"
        status_text = "通过 (Pass)" if is_zh else "Pass"
        if not basis:
            basis.append("未见明显风险信号" if is_zh else "No strong risk signals detected")

    return {
        "is_risk": bool(is_risk),
        "risk_level": level,
        "status_text": status_text,
        "score": int(score),
        "basis": basis,
        "flags": flags,
        "er_prob_sum": float(er_prob_sum),
        "top_target": top_target,
        "top_probability": float(top_prob),
        "ac50_nM": ac50_nm,
        "phenol_count": int(phenol_count),
        "aromatic_rings": int(aromatic_rings),
        "logp": logp_val,
        "edc_hits": edc_hits,
    }

# === New main-model featurization ===
def smiles_to_mol(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except Exception:
        return None


def build_morgan_matrix(smiles_list, radius: int = 2, n_bits: int = 2048):
    X_bits = np.zeros((len(smiles_list), n_bits), dtype=np.int8)
    valid_mask = np.zeros((len(smiles_list),), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            X_bits[i] = arr
            valid_mask[i] = True
        except Exception:
            valid_mask[i] = False
    return X_bits, valid_mask


def featurize_smiles(smiles_list):
    X_bits, valid_mask = build_morgan_matrix(smiles_list)
    X_new = {
        'bits': X_bits,
        'bool': X_bits.astype(bool),
        'float': X_bits.astype(np.float32),
        'float_cls': X_bits.astype(np.float32),
        'float_reg': X_bits.astype(np.float32),
    }

    # Extra features must match training-time ordering in the notebook.
    if Config.USE_PHYCHEM_FEATURES:
        phys_rows = []
        for smi in smiles_list:
            mol = smiles_to_mol(smi)
            if mol is None:
                phys_rows.append([0.0] * 10)
                continue
            phys_rows.append([
                float(Descriptors.MolWt(mol)),
                float(rdMolDescriptors.CalcTPSA(mol)),
                float(Descriptors.MolLogP(mol)),
                float(rdMolDescriptors.CalcNumHBD(mol)),
                float(rdMolDescriptors.CalcNumHBA(mol)),
                float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
                float(Descriptors.RingCount(mol)),
                float(rdMolDescriptors.CalcNumAromaticRings(mol)),
                float(mol.GetNumHeavyAtoms()),
                float(Descriptors.FractionCSP3(mol)),
            ])
        X_phys = np.array(phys_rows, dtype=np.float32)
        X_new['float_cls'] = np.hstack([X_new['float_cls'], X_phys]).astype(np.float32)
        X_new['float_reg'] = np.hstack([X_new['float_reg'], X_phys]).astype(np.float32)

    if Config.USE_DOCKING_FEATURES:
        impute = float(Config.DOCK_IMPUTE_VALUE)
        dock = np.full((len(smiles_list), 1), impute, dtype=np.float32)
        miss = np.ones((len(smiles_list), 1), dtype=np.float32)
        X_dock = np.hstack([dock, miss]).astype(np.float32)
        X_new['float_reg'] = np.hstack([X_new['float_reg'], X_dock]).astype(np.float32)

    if Config.USE_3D_DESCRIPTORS:
        SHAPE_COLS = [
            'Asphericity',
            'Eccentricity',
            'InertialShapeFactor',
            'NPR1',
            'NPR2',
            'RadiusOfGyration',
            'SpherocityIndex',
        ]
        shape = np.zeros((len(smiles_list), len(SHAPE_COLS)), dtype=np.float32)
        miss = np.zeros((len(smiles_list), 1), dtype=np.float32)

        for i, smi in enumerate(smiles_list):
            mol0 = smiles_to_mol(smi)
            if mol0 is None:
                miss[i, 0] = 1.0
                continue
            mol = Chem.AddHs(mol0)
            params = AllChem.ETKDGv3()
            params.randomSeed = 1337
            if AllChem.EmbedMolecule(mol, params=params) != 0:
                miss[i, 0] = 1.0
                continue
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
                except Exception:
                    pass
            try:
                shape[i, :] = np.array([
                    float(rdMolDescriptors.CalcAsphericity(mol)),
                    float(rdMolDescriptors.CalcEccentricity(mol)),
                    float(rdMolDescriptors.CalcInertialShapeFactor(mol)),
                    float(rdMolDescriptors.CalcNPR1(mol)),
                    float(rdMolDescriptors.CalcNPR2(mol)),
                    float(rdMolDescriptors.CalcRadiusOfGyration(mol)),
                    float(rdMolDescriptors.CalcSpherocityIndex(mol)),
                ], dtype=np.float32)
            except Exception:
                miss[i, 0] = 1.0

        X_shape = np.hstack([shape, miss]).astype(np.float32)
        X_new['float_cls'] = np.hstack([X_new['float_cls'], X_shape]).astype(np.float32)
        X_new['float_reg'] = np.hstack([X_new['float_reg'], X_shape]).astype(np.float32)

    return X_new, valid_mask


def _align_proba(proba: np.ndarray, fitted_classes: np.ndarray, global_classes: np.ndarray) -> np.ndarray:
    idx = {c: i for i, c in enumerate(fitted_classes)}
    out = np.zeros((proba.shape[0], len(global_classes)), dtype=np.float64)
    for j, c in enumerate(global_classes):
        if c in idx:
            out[:, j] = proba[:, idx[c]]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return out / row_sum


def _predict_proba_from_artifact(art: dict, X_new: dict) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(art, dict):
        raise ValueError("Classifier artifact must be a dict.")

    if art.get('type') == 'ensemble':
        X_map = {name: X_new[kind] for name, kind in art['x_kind'].items()}
        proba = art['model'].predict_proba(X_map)
        classes_ = np.array(getattr(art['model'], 'classes_', art.get('classes_', [])))
        return proba, classes_

    # single
    name = next(iter(art['x_kind'].keys()))
    X_in = X_new[art['x_kind'][name]]
    raw = art['model'].predict_proba(X_in)
    fitted = np.array(getattr(art['model'], 'classes_', art.get('classes_', [])))
    classes_ = np.array(art.get('classes_', fitted))
    proba = _align_proba(raw, fitted, classes_)
    return proba, classes_


@dataclass
class WeightedEnsembleClassifier:
    estimators: dict
    weights: dict
    classes_: np.ndarray

    def predict_proba(self, X_map: dict[str, np.ndarray]) -> np.ndarray:
        proba = None
        for name, est in self.estimators.items():
            p = _align_proba(est.predict_proba(X_map[name]), est.classes_, self.classes_)
            w = float(self.weights[name])
            proba = p * w if proba is None else proba + p * w
        return proba


@dataclass
class WeightedEnsembleRegressor:
    estimators: dict
    weights: dict
    bias: float = 0.0

    def predict(self, X_map: dict[str, np.ndarray]) -> np.ndarray:
        pred = np.full((len(next(iter(X_map.values()))),), float(self.bias), dtype=np.float64)
        for name, est in self.estimators.items():
            p = est.predict(X_map[name])
            w = float(self.weights[name])
            pred = pred + p * w
        return pred


def _register_pickle_classes():
    import sys
    import types
    # Ensure the project root is on sys.path and create a stable package alias
    # so joblib can import classes saved under "online_server.*".
    try:
        here = Path(__file__).resolve()
        project_root = here.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        if "online_server" not in sys.modules:
            pkg = types.ModuleType("online_server")
            pkg.__path__ = [str(here.parent)]
            sys.modules["online_server"] = pkg
        mod = sys.modules.get(__name__)
        if mod is not None:
            sys.modules.setdefault("online_server.backend", mod)
        pkg = sys.modules.get("online_server")
        if pkg is not None:
            setattr(pkg, "WeightedEnsembleClassifier", WeightedEnsembleClassifier)
            setattr(pkg, "WeightedEnsembleRegressor", WeightedEnsembleRegressor)
    except Exception:
        pass

    main_mod = sys.modules.get('__main__')
    if main_mod is not None:
        main_mod.WeightedEnsembleClassifier = WeightedEnsembleClassifier
        main_mod.WeightedEnsembleRegressor = WeightedEnsembleRegressor

def prepare_graph_inputs(mol, max_nodes=200):
    graph = mol_to_simple_graph(mol, max_nodes=max_nodes)
    if graph is None:
        return None, "Invalid molecule"
    node_feats = np.zeros((max_nodes, graph['node_feats'].shape[1]), dtype=np.float32)
    node_feats[:graph['n_nodes']] = graph['node_feats']
    edge_types = np.zeros((max_nodes, max_nodes), dtype=np.uint8)
    edge_types[:graph['n_nodes'], :graph['n_nodes']] = graph['edge_types']
    node_mask = np.zeros(max_nodes, dtype=bool)
    node_mask[:graph['n_nodes']] = True
    desc = compute_descriptors(mol)
    fp = compute_fingerprint(mol)
    return {
        'node_feats': torch.from_numpy(node_feats).unsqueeze(0).to(Config.DEVICE),
        'edge_types': torch.from_numpy(edge_types).unsqueeze(0).to(Config.DEVICE).long(),
        'node_mask': torch.from_numpy(node_mask).unsqueeze(0).to(Config.DEVICE),
        'desc': torch.from_numpy(desc).unsqueeze(0).to(Config.DEVICE),
        'fp': torch.from_numpy(fp).unsqueeze(0).to(Config.DEVICE)
    }, None

# ========== 补充缺失的模型定义 (自动生成) ==========
import torch
import torch.nn as nn
import numpy as np

# 1. 常量定义
DESC_NAMES = ['ExactMolWt', 'MolLogP', 'TPSA', 'NumHBD', 'NumHBA', 'NumRotB', 'RingCount', 'FractionCSP3', 'LabuteASA', 'HeavyAtomCount', 'NumAromaticRings', 'NumAliphaticRings', 'alert_nitro_aromatic', 'alert_nitro_aliphatic', 'alert_azide', 'alert_aniline', 'alert_anilide', 'alert_isocyanate', 'alert_michael_acceptor', 'alert_epoxide', 'alert_organohalide', 'alert_quinone', 'alert_thioamide', 'alert_hydrazine']
FINGERPRINT_BITS = 2048
GRAPH_DIM = 384
GT_LAYERS = 6
GT_HEADS = 8
DROPOUT = 0.1
DROPEDGE_P = 0.1
EDGE_VOCAB_SIZE = 6 

# === Two-step classifier featurization (ECFP + simple descriptors + optional MACCS) ===
descriptor_fns_simple = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
]

def calc_descriptors_simple(mol):
    return np.array([fn(mol) for fn in descriptor_fns_simple], dtype=np.float32)

def morgan_fp_bits(mol, radius: int = 2, n_bits: int = 2048):
    arr = np.zeros((n_bits,), dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def maccs_fp_bits(mol):
    bv = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((bv.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def build_feature_vector_two_step(smiles: str, use_maccs: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        scaffold = None
    if scaffold is None:
        return None
    parts = [morgan_fp_bits(mol), calc_descriptors_simple(mol)]
    if use_maccs:
        parts.append(maccs_fp_bits(mol))
    return np.concatenate(parts).astype(np.float32)

# 2. Graph Transformer Definitions
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, dim, heads, dropout, edge_vocab):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.store_attn = False
        self.last_attn = None
        self.edge_encoder = nn.Embedding(edge_vocab, heads)
        with torch.no_grad():
            self.edge_encoder.weight.zero_()
            if edge_vocab > 1:
                self.edge_encoder.weight[1:].normal_(mean=0.0, std=0.02)

    def forward(self, x, attn_mask, edge_types, node_mask):
        B, N, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, self.heads, self.dim_head).transpose(1, 2) for t in qkv]
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if edge_types is not None:
            edge_bias = self.edge_encoder(edge_types)
            edge_bias = edge_bias.permute(0, 3, 1, 2)
            attn_logits = attn_logits + edge_bias
        mask = None
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        row_valid = node_mask.unsqueeze(1).unsqueeze(-1)
        attn_logits = torch.where(row_valid, attn_logits, torch.zeros_like(attn_logits))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = attn * row_valid.float()
        attn = self.dropout(attn)
        if self.store_attn:
            self.last_attn = attn.detach().to('cpu')
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.dim)
        out = self.to_out(out)
        out = out * node_mask.unsqueeze(-1).float()
        return out

class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads, dropout, edge_vocab):
        super().__init__()
        self.attn = MultiHeadGraphAttention(dim, heads, dropout, edge_vocab)
        self.ffn = FeedForward(dim, dim * 4, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, attn_mask, edge_types, node_mask):
        attn_out = self.attn(self.norm1(x), attn_mask, edge_types, node_mask)
        x = x + self.dropout(attn_out)
        ff_out = self.ffn(self.norm2(x))
        x = x + ff_out
        x = x * node_mask.unsqueeze(-1).float()
        return x

class GraphTransformerModel(nn.Module):
    def __init__(self, node_feat_dim, num_targets, desc_dim=len(DESC_NAMES), fp_dim=FINGERPRINT_BITS,
                 graph_dim=None, gt_layers=None, gt_heads=None, dropout=None, dropedge_p=None):
        super().__init__()
        self.graph_dim = int(graph_dim if graph_dim is not None else GRAPH_DIM)
        self.gt_layers = int(gt_layers if gt_layers is not None else GT_LAYERS)
        self.gt_heads = int(gt_heads if gt_heads is not None else GT_HEADS)
        self.dropout = float(dropout if dropout is not None else DROPOUT)
        self.desc_dim = desc_dim
        self.fp_dim = fp_dim
        self.dropedge_p = DROPEDGE_P if dropedge_p is None else dropedge_p
        half_dim = max(self.graph_dim // 2, 32)
        self.input_proj = nn.Linear(node_feat_dim, self.graph_dim)
        self.desc_mlp = nn.Sequential(nn.Linear(desc_dim, half_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(half_dim, half_dim))
        self.fp_mlp = nn.Sequential(nn.Linear(fp_dim, half_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(half_dim, half_dim))
        self.layers = nn.ModuleList([GraphTransformerLayer(self.graph_dim, self.gt_heads, self.dropout, EDGE_VOCAB_SIZE) for _ in range(self.gt_layers)])
        self.norm = nn.LayerNorm(self.graph_dim)
        self.readout = nn.Sequential(nn.Linear(self.graph_dim + half_dim + half_dim, self.graph_dim), nn.GELU(), nn.Dropout(self.dropout), nn.Linear(self.graph_dim, num_targets))

    def forward(self, node_feats, node_mask, edge_types, global_desc=None, fingerprints=None):
        x = self.input_proj(node_feats)
        x = x * node_mask.unsqueeze(-1).float()
        if edge_types is not None:
            B, N = edge_types.shape[0], edge_types.shape[1]
            eye = torch.eye(N, dtype=torch.long, device=edge_types.device).unsqueeze(0).expand(B, -1, -1)
            edge_types = torch.where(eye.bool(), edge_types.clamp(min=EDGE_VOCAB_SIZE-1), edge_types)
        attn_mask = (edge_types > 0)
        node_pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        attn_mask = attn_mask & node_pair_mask
        for layer in self.layers:
            x = layer(x, attn_mask, edge_types, node_mask)
        x = self.norm(x)
        mask = node_mask.unsqueeze(-1).float()
        graph_emb = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        batch_size = x.size(0)
        desc_emb = self.desc_mlp(global_desc) if global_desc is not None else torch.zeros(batch_size, self.graph_dim // 2, device=x.device)
        fp_emb = self.fp_mlp(fingerprints) if fingerprints is not None else torch.zeros(batch_size, self.graph_dim // 2, device=x.device)
        combo = torch.cat([graph_emb, desc_emb, fp_emb], dim=-1)
        return self.readout(combo)

class TabMixer(nn.Module):
    def __init__(self, input_dim, hidden=512, depth=4, dropout=0.15):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.LayerNorm(dim)))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, x):
        for layer in self.layers: x = x + layer(x)
        return self.head(x).squeeze(-1)

# ========== 智能模型加载器 ==========
def _merge_model_kwargs(defaults, meta):
    kwargs = defaults.copy()
    info_sources = []
    if isinstance(meta, dict):
        info_sources.append(meta)
        cfg = meta.get('config') if isinstance(meta.get('config'), dict) else None
        if cfg:
            info_sources.append(cfg)
    for src in info_sources:
        for key in ['node_feat_dim', 'desc_dim', 'fp_dim', 'graph_dim', 'gt_layers',
                    'gt_heads', 'dropout', 'dropedge_p', 'input_dim', 'hidden', 'depth']:
            if key in src:
                kwargs[key] = src[key]
        if 'targets' in src and 'num_targets' in kwargs and isinstance(src['targets'], (list, tuple)):
            kwargs['num_targets'] = len(src['targets'])
        if 'num_targets' in src:
            kwargs['num_targets'] = src['num_targets']
    return kwargs

def smart_load_model(path, model_class, default_kwargs, device):
    obj = torch.load(path, map_location=device)
    meta = obj if isinstance(obj, dict) else {}
    if isinstance(obj, torch.nn.Module):
        return obj.eval(), 'complete_model', meta
    if isinstance(obj, dict):
        state_dict = None
        for key in ['model_state_dict', 'state_dict', 'model', 'net', 'best_model']:
            if key in obj:
                candidate = obj[key]
                state_dict = candidate.state_dict() if isinstance(candidate, torch.nn.Module) else candidate
                break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
            state_dict = obj
        if state_dict is not None:
            kwargs = _merge_model_kwargs(default_kwargs, meta)
            model = model_class(**kwargs).to(device)
            model.load_state_dict(state_dict, strict=False)
            return model.eval(), 'state_dict', meta
    return None, 'unknown_format', meta

# ========== 预测引擎 ==========
class PredictionEngine:
    def __init__(self):
        self.models = {}
        self.loaded = False
        self.target_labels = []
        self.cls_art = None
        self.reg_art = None

    def load_models(self):
        if self.loaded:
            return True, "Models already loaded"
        try:
            _register_pickle_classes()
            import joblib

            if not Config.MODEL_TARGET.exists():
                return False, f"❌ 缺少主靶点模型: {Config.MODEL_TARGET}"
            if not Config.MODEL_REG.exists():
                return False, "❌ 缺少主AC50模型文件，请检查服务器模型目录配置。"

            self.cls_art = joblib.load(Config.MODEL_TARGET)
            self.reg_art = joblib.load(Config.MODEL_REG)

            if isinstance(self.cls_art, dict):
                if self.cls_art.get('type') == 'postprocess_esr12':
                    classes = self.cls_art.get('classes') or self.cls_art.get('classes_')
                    if not classes and isinstance(self.cls_art.get('base'), dict):
                        base = self.cls_art['base']
                        if base.get('type') == 'ensemble':
                            classes = list(getattr(base['model'], 'classes_', []))
                        else:
                            classes = list(base.get('classes_', [])) or list(getattr(base.get('model', None), 'classes_', []))
                    self.target_labels = list(classes or [])
                elif self.cls_art.get('type') == 'ensemble':
                    self.target_labels = list(getattr(self.cls_art['model'], 'classes_', []))
                else:
                    self.target_labels = list(self.cls_art.get('classes_', []))

            self.loaded = True
            return True, "✓ 模型加载成功"
        except Exception as e:
            import traceback
            return False, f"Error: {str(e)}\n{traceback.format_exc()[:500]}"

    def predict_target(self, mol):
        if not self.loaded:
            success, msg = self.load_models()
            if not success:
                return None, f"Models not loaded: {msg}"
        smi = Chem.MolToSmiles(mol) if mol is not None else None
        if smi is None or not smi:
            return None, "Invalid molecule"

        X_new, valid = featurize_smiles([smi])
        if not bool(valid[0]):
            return None, "Featurization failed (invalid SMILES)"

        art = self.cls_art
        try:
            if art.get('type') == 'postprocess_esr12':
                base = art.get('base', {})
                base_proba, base_classes = _predict_proba_from_artifact(base, X_new)
                classes_ = np.array(art.get('classes') or art.get('classes_') or base_classes)
                if list(base_classes) != list(classes_):
                    base_proba = _align_proba(base_proba, base_classes, classes_)
                # Optional ESR1/ESR2 probability scaling (from notebook tuning)
                scale = art.get('scale') or {}
                if scale:
                    base_proba = base_proba.copy()
                    idx_map = {c: i for i, c in enumerate(classes_)}
                    for cls, fac in scale.items():
                        if cls in idx_map:
                            base_proba[:, idx_map[cls]] *= float(fac)
                    row_sum = base_proba.sum(axis=1, keepdims=True)
                    row_sum[row_sum == 0] = 1.0
                    base_proba = base_proba / row_sum

                base_pred = classes_[np.argmax(base_proba, axis=1)]

                labels = art.get('labels', {})
                decoy_label = labels.get('decoy', 'Decoy')
                gper1_label = labels.get('gper1', 'GPER1')
                esr1_label = labels.get('esr1', 'ESR1')
                esr2_label = labels.get('esr2', 'ESR2')

                idx_decoy = list(classes_).index(decoy_label) if decoy_label in classes_ else None
                idx_gper1 = list(classes_).index(gper1_label) if gper1_label in classes_ else None

                pred = base_pred.copy()
                post_proba = base_proba.copy()

                if idx_decoy is not None and idx_gper1 is not None and art.get('esr12_model') is not None:
                    t_decoy = float(art.get('thresholds', {}).get('decoy', 0.0))
                    t_gper1 = float(art.get('thresholds', {}).get('gper1', 0.0))
                    t_esr2 = float(art.get('thresholds', {}).get('esr2', 0.5))
                    p_decoy = base_proba[:, idx_decoy]
                    p_gper1 = base_proba[:, idx_gper1]

                    esr12_kind = art.get('esr12_kind', 'float_cls')
                    if esr12_kind not in X_new:
                        esr12_kind = 'float_cls' if 'float_cls' in X_new else list(X_new.keys())[0]
                    esr12_prob = art['esr12_model'].predict_proba(X_new[esr12_kind])[:, 1]

                    keep_decoy = (base_pred == decoy_label) & (p_decoy >= t_decoy)
                    keep_gper1 = (base_pred == gper1_label) & (p_gper1 >= t_gper1)
                    override = ~(keep_decoy | keep_gper1)
                    pred[override] = np.where(esr12_prob[override] >= t_esr2, esr2_label, esr1_label)

                    if esr1_label in classes_ and esr2_label in classes_:
                        idx_esr1 = list(classes_).index(esr1_label)
                        idx_esr2 = list(classes_).index(esr2_label)
                        post_proba[override, :] = 0.0
                        post_proba[override, idx_esr1] = 1.0 - esr12_prob[override]
                        post_proba[override, idx_esr2] = esr12_prob[override]

                probs = {str(c): float(p) for c, p in zip(classes_, post_proba[0])}
                return {'target': str(pred[0]), 'probs': probs, 'unknown': False}, None

            proba, classes_ = _predict_proba_from_artifact(art, X_new)
            pred = classes_[int(np.argmax(proba, axis=1)[0])]
            probs = {str(c): float(p) for c, p in zip(classes_, proba[0])}
            return {'target': str(pred), 'probs': probs, 'unknown': False}, None
        except Exception as e:
            import traceback
            return None, f"Prediction error: {str(e)}\n{traceback.format_exc()[:500]}"

    def _predict_with_reg_artifact(self, art, X_new, idx):
        if art.get('type') == 'ensemble':
            X_map = {name: X_new[kind][idx] for name, kind in art['x_kind'].items()}
            return art['model'].predict(X_map)
        name = next(iter(art['x_kind'].keys()))
        X_in = X_new[art['x_kind'][name]][idx]
        return art['model'].predict(X_in)

    def predict_ac50_info(self, mol):
        if not self.loaded:
            success, msg = self.load_models()
            if not success:
                return None, f"Models not loaded: {msg}"

        smi = Chem.MolToSmiles(mol) if mol is not None else None
        if smi is None or not smi:
            return None, "Invalid molecule"

        X_new, valid = featurize_smiles([smi])
        if not bool(valid[0]):
            return None, "Featurization failed (invalid SMILES)"

        reg_art = self.reg_art
        try:
            if reg_art.get('type') == 'per_target':
                pred_t, err = self.predict_target(mol)
                if err:
                    return None, err
                target_label = pred_t.get('target')
                if target_label not in reg_art['artifacts']:
                    return None, f"No regressor for target: {target_label}"
                pred = self._predict_with_reg_artifact(reg_art['artifacts'][target_label], X_new, [0])
            else:
                pred = self._predict_with_reg_artifact(reg_art, X_new, [0])

            val = float(pred[0])
            ac50_nM = float(10 ** (9.0 - val))
            ac50_uM = float(ac50_nM / 1000.0)
            return {'pAC50': val, 'AC50_nM': ac50_nM, 'AC50_uM': ac50_uM}, None
        except Exception as e:
            import traceback
            return None, f"AC50 error: {str(e)}\n{traceback.format_exc()[:500]}"

    # Backward-compatible alias (returns AC50 in μM)
    def predict_ac50(self, mol):
        info, err = self.predict_ac50_info(mol)
        if err:
            return None, err
        return info['AC50_uM'], None

    # Legacy helper: keep for backward compatibility, but avoid using in the web UI.
    def predict_ic50(self, mol):
        info, err = self.predict_ac50_info(mol)
        if err:
            return None, err
        # Map to legacy key names.
        return {
            "pIC50": info.get("pAC50"),
            "IC50_nM": info.get("AC50_nM"),
            "IC50_uM": info.get("AC50_uM"),
        }, None

# Shared prediction engine instance for the web app
_PRED_ENGINE = None

def _get_pred_engine():
    global _PRED_ENGINE
    if _PRED_ENGINE is None:
        _PRED_ENGINE = PredictionEngine()
    return _PRED_ENGINE

def target_prediction(smiles: str):
    """Run target prediction + toxicity alerts for a single SMILES."""
    try:
        if not smiles or not str(smiles).strip():
            return {"error": "Empty SMILES"}, []
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}, []

        engine = _get_pred_engine()
        pred, err = engine.predict_target(mol)
        if err:
            return {"error": err}, []

        probs = pred.get("probs", {}) or {}
        items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        pred_results = []
        for i, (name, prob) in enumerate(items):
            raw_name = str(name)
            display_name = raw_name
            is_decoy = raw_name.lower() == "decoy"
            if is_decoy:
                display_name = "非雌激素受体靶向性"
            pred_results.append({
                "Rank": i + 1,
                "Target Name": display_name,
                "Raw Target": raw_name,
                "Is Decoy": bool(is_decoy),
                "Probability": float(prob),
            })

        # Optional AC50 prediction (skip for Decoy)
        ac50_info = None
        raw_top = str(pred.get("target", ""))
        if raw_top.lower() != "decoy":
            ac50_info, ac50_err = engine.predict_ac50_info(mol)
            if ac50_err:
                ac50_info = {"error": ac50_err}
            else:
                ac50_info = dict(ac50_info)
                ac50_info["target"] = raw_top
        else:
            ac50_info = {"note": "Decoy: no ER target"}

        alerts = detect_toxic_alerts(mol)
        toxicity_results = [
            {
                "Alert": a.get("name", ""),
                "Label": a.get("label", ""),
                "Count": int(a.get("count", 0)),
                "SMARTS": a.get("smarts", ""),
            }
            for a in alerts
        ]

        return pred_results, toxicity_results, ac50_info
    except Exception as e:
        return {"error": f"{str(e)}"}, []

# ========== 对接功能 ==========
def _run_cmd(cmd, cwd=None, label="command", log_file=None):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if log_file and cwd:
        log_path = Path(cwd) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8", errors="ignore") as fh:
            if proc.stdout:
                fh.write("STDOUT:\n" + proc.stdout + "\n")
            if proc.stderr:
                fh.write("STDERR:\n" + proc.stderr + "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit={proc.returncode})")
    return proc

def _prepare_ligand_pdbqt(lig_pdb: Path, tmp_dir: Path):
    try:
        _run_cmd([Config.PREP_LIG, '-l', lig_pdb.name, '-o', 'ligand.pdbqt'], cwd=tmp_dir, label="prepare_ligand")
        return tmp_dir / "ligand.pdbqt"
    except Exception:
        obabel_bin = shutil.which("obabel")
        if obabel_bin:
            _run_cmd([obabel_bin, lig_pdb.name, '-O', 'ligand.pdbqt', '--partialcharge', 'gasteiger'], 
                    cwd=tmp_dir, label="obabel")
            return tmp_dir / "ligand.pdbqt"
        raise

def _prepare_receptor_pdbqt(receptor_path: Path, tmp_dir: Path):
    rec_copy = tmp_dir / receptor_path.name
    shutil.copyfile(receptor_path, rec_copy)
    out_path = tmp_dir / "receptor.pdbqt"
    try:
        _run_cmd(
            [Config.PREP_REC, "-r", rec_copy.name, "-o", out_path.name, "-A", "checkhydrogens"],
            cwd=tmp_dir,
            label="prepare_receptor",
        )
        if out_path.exists():
            return out_path
    except Exception:
        pass

    # Fallback: OpenBabel PDB -> PDBQT (no ADFRsuite required).
    obabel_bin = shutil.which("obabel")
    if obabel_bin:
        _run_cmd([obabel_bin, rec_copy.name, "-O", out_path.name], cwd=tmp_dir, label="obabel_receptor")
        return out_path

    raise RuntimeError("prepare_receptor 不可用，且未找到 obabel，无法生成 receptor.pdbqt")

def _compute_box(receptor_path: Path):
    coords = []
    with open(receptor_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except Exception:
                    continue
    if not coords:
        raise RuntimeError("Failed to parse receptor coordinates")
    coords = np.array(coords)
    center = coords.mean(axis=0)
    size = np.clip((coords.max(axis=0) - coords.min(axis=0)) * 1.1, 20.0, None)
    return center, size

def get_receptor_box(receptor_name: str):
    """Compute docking box from full receptor coordinates."""
    try:
        receptor_path = Config.RECEPTOR_DIR / receptor_name
        if not receptor_path.exists():
            return None, None, f"Receptor not found: {receptor_name}"
        center, size = _compute_box(receptor_path)
        return center, size, None
    except Exception as e:
        return None, None, str(e)

def _rdkit_to_3d_pdb(mol) -> Path:
    tmp_dir = Config.OUTPUT_DIR / "Docking_Temp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_pdb = tmp_dir / "ligand_rdkit.pdb"
    work_mol = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(work_mol, params=params) != 0:
        raise ValueError("RDKit Embed 失败")
    try:
        AllChem.MMFFOptimizeMolecule(work_mol, maxIters=200)
    except Exception:
        AllChem.UFFOptimizeMolecule(work_mol, maxIters=200)
    Chem.MolToPDBFile(work_mol, str(tmp_pdb))
    return tmp_pdb

def _obabel_from_smiles(smiles: str, out_pdb: Path):
    obabel_bin = shutil.which("obabel")
    if not obabel_bin:
        raise RuntimeError("未找到 obabel")
    _run_cmd([obabel_bin, "-:", smiles, "-O", str(out_pdb), "--gen3d"], label="obabel_gen3d")
    if not out_pdb.exists():
        raise RuntimeError("obabel gen3d 失败")
    return out_pdb

def dock_molecule(mol, receptor_name):
    receptor_path = Config.RECEPTOR_DIR / receptor_name
    if not receptor_path.exists():
        return None, f"Receptor {receptor_name} not found"
    try:
        tmp_dir = Config.OUTPUT_DIR / "Docking_Temp"
        tmp_dir.mkdir(exist_ok=True)
        try:
            lig_pdb = _rdkit_to_3d_pdb(mol)
        except Exception:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
            lig_pdb = tmp_dir / "ligand_rdkit.pdb"
            _obabel_from_smiles(smiles, lig_pdb)
        lig_pdbqt = _prepare_ligand_pdbqt(lig_pdb, tmp_dir)
        rec_pdbqt = _prepare_receptor_pdbqt(receptor_path, tmp_dir)
        center, size = _compute_box(receptor_path)
        cmd = [Config.VINA, '--receptor', rec_pdbqt.name, '--ligand', lig_pdbqt.name,
               '--out', 'out.pdbqt', '--center_x', f"{center[0]:.3f}",
               '--center_y', f"{center[1]:.3f}", '--center_z', f"{center[2]:.3f}",
               '--size_x', f"{size[0]:.3f}", '--size_y', f"{size[1]:.3f}",
               '--size_z', f"{size[2]:.3f}", '--exhaustiveness', '8']
        _run_cmd(cmd, cwd=tmp_dir, label="vina", log_file="vina.log")
        score = 0.0
        with open(tmp_dir / "out.pdbqt") as f:
            for line in f:
                if 'REMARK VINA RESULT' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        score = float(parts[3])
                    break
        return {
            'score': score,
            'receptor': str(receptor_path),
            'ligand': str(tmp_dir / "out.pdbqt"),
            'receptor_name': receptor_name
        }, None
    except Exception as e:
        import traceback
        return None, f"Docking error: {str(e)}"

# --- Streamlit docking helpers (match app.py interface) ---
_DOCKING_STATE = {
    "receptor_name": None,
    "receptor_path": None,
    "ligand_pdbqt": None,
    "receptor_pdbqt": None,
    "tmp_dir": None,
}

def docking_preparation(receptor_name: str, smiles: str) -> bool:
    """Prepare receptor/ligand PDBQT for later docking."""
    try:
        receptor_path = Config.RECEPTOR_DIR / receptor_name
        if not receptor_path.exists():
            return False
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return False
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_dir = Config.OUTPUT_DIR / "Docking_Temp"
        tmp_dir.mkdir(exist_ok=True)
        # Clean previous outputs that may confuse the UI
        for fname in ["ligand.pdbqt", "receptor.pdbqt", "out.pdbqt", "vina.log", "docked_ligand.pdb"]:
            p = tmp_dir / fname
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        try:
            lig_pdb = _rdkit_to_3d_pdb(mol)
        except Exception:
            smiles_clean = Chem.MolToSmiles(Chem.RemoveHs(mol))
            lig_pdb = tmp_dir / "ligand_rdkit.pdb"
            _obabel_from_smiles(smiles_clean, lig_pdb)
        lig_pdbqt = _prepare_ligand_pdbqt(lig_pdb, tmp_dir)
        rec_pdbqt = _prepare_receptor_pdbqt(receptor_path, tmp_dir)
        # persist smiles for downstream PLIP use
        try:
            smi_path = Config.OUTPUT_DIR / "docked_ligand.smiles"
            smi_path.write_text((smiles or "").strip() + "\n", encoding="utf-8")
        except Exception:
            pass
        _DOCKING_STATE.update({
            "receptor_name": receptor_name,
            "receptor_path": receptor_path,
            "ligand_pdbqt": lig_pdbqt,
            "receptor_pdbqt": rec_pdbqt,
            "tmp_dir": tmp_dir,
        })
        return True
    except Exception:
        _DOCKING_STATE.update({
            "receptor_name": None,
            "receptor_path": None,
            "ligand_pdbqt": None,
            "receptor_pdbqt": None,
            "tmp_dir": None,
        })
        return False

def _parse_vina_scores(pdbqt_path: Path):
    scores = []
    try:
        with open(pdbqt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "REMARK VINA RESULT" in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        affinity = float(parts[3])
                        rmsd_lb = float(parts[4])
                        rmsd_ub = float(parts[5])
                        scores.append({
                            "rank": len(scores) + 1,
                            "affinity": affinity,
                            "rmsd_lb": rmsd_lb,
                            "rmsd_ub": rmsd_ub,
                        })
    except Exception:
        pass
    return scores

def _convert_pdbqt_to_pdb(pdbqt_path: Path, out_pdb: Path) -> bool:
    obabel_bin = shutil.which("obabel")
    if not obabel_bin:
        return False
    # 强制加上 -f 1 -l 1 确保我们只将得分最高（Top 1）的对接构象输出，供 Mol* 准确分析
    _run_cmd([obabel_bin, str(pdbqt_path), "-O", str(out_pdb), "-f", "1", "-l", "1"], label="obabel_pdb")
    return out_pdb.exists()

def _convert_pdbqt_pose_to_pdb(pdbqt_path: Path, out_pdb: Path) -> bool:
    obabel_bin = shutil.which("obabel")
    if not obabel_bin:
        return False
    _run_cmd([obabel_bin, str(pdbqt_path), "-O", str(out_pdb)], label="obabel_pdb_pose")
    return out_pdb.exists()

def _convert_pdb_to_format(in_pdb: Path, out_path: Path) -> bool:
    obabel_bin = shutil.which("obabel")
    if not obabel_bin:
        return False
    _run_cmd([obabel_bin, str(in_pdb), "-O", str(out_path)], label="obabel_convert")
    return out_path.exists()

def _split_vina_pose_blocks(pdbqt_path: Path):
    try:
        lines = pdbqt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    blocks = []
    cur = []
    has_model = False
    for line in lines:
        if line.startswith("MODEL"):
            if cur:
                blocks.append(cur)
                cur = []
            has_model = True
            cur.append(line)
            continue
        cur.append(line)
        if line.startswith("ENDMDL"):
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    if not has_model:
        return [lines] if lines else []
    return blocks

def prepare_pose_complexes(receptor_path: Path, out_pdbqt: Path, out_dir: Path, convert_format: str = "mol2"):
    results = []
    try:
        blocks = _split_vina_pose_blocks(out_pdbqt)
        if not blocks:
            return results
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, block in enumerate(blocks, 1):
            pose_dir = out_dir / f"pose_{idx:03d}"
            pose_dir.mkdir(parents=True, exist_ok=True)
            pose_pdbqt = pose_dir / "ligand.pdbqt"
            pose_pdb = pose_dir / "ligand.pdb"
            complex_pdb = pose_dir / "complex.pdb"
            complex_other = pose_dir / f"complex.{convert_format}" if convert_format else None
            with open(pose_pdbqt, "w", encoding="utf-8") as fh:
                fh.write("\n".join(block) + "\n")
            ok_pdb = _convert_pdbqt_pose_to_pdb(pose_pdbqt, pose_pdb)
            if ok_pdb:
                _merge_pdb_files(receptor_path, pose_pdb, complex_pdb)
                if complex_other:
                    _convert_pdb_to_format(complex_pdb, complex_other)
            results.append({
                "pose_index": idx,
                "ligand_pdbqt": str(pose_pdbqt),
                "ligand_pdb": str(pose_pdb) if ok_pdb else None,
                "complex_pdb": str(complex_pdb) if ok_pdb and complex_pdb.exists() else None,
                "complex_converted": str(complex_other) if (complex_other and complex_other.exists()) else None,
            })
        return results
    except Exception:
        return results

def _float_or_none(v):
    try:
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None

def _summarize_distribution(values):
    arr = np.array([float(x) for x in values], dtype=float)
    if arr.size == 0:
        return None
    return {
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr, ddof=0)),
    }

def _build_docking_analysis_report(scores, lang: str = "English", top_n: int = 5):
    rows = []
    for i, s in enumerate(scores or [], start=1):
        try:
            rank = int(s.get("rank", i))
        except Exception:
            rank = i
        aff = _float_or_none(s.get("affinity"))
        lb = _float_or_none(s.get("rmsd_lb"))
        ub = _float_or_none(s.get("rmsd_ub"))
        if aff is None:
            continue
        rows.append({"rank": rank, "affinity": aff, "rmsd_lb": lb, "rmsd_ub": ub})

    if not rows:
        return None

    rows = sorted(rows, key=lambda x: x["rank"])
    is_zh = _is_zh_lang(lang)
    n_pose = len(rows)

    # Affinity distribution
    aff_vals = [r["affinity"] for r in rows]
    best_aff = min(aff_vals)
    aff_stats = _summarize_distribution(aff_vals)
    spread = (aff_stats["max"] - aff_stats["min"]) if aff_stats else None
    deltas = [a - best_aff for a in aff_vals]
    close_count = sum(1 for d in deltas if d <= 0.5)
    medium_count = sum(1 for d in deltas if 0.5 < d <= 1.5)
    weak_count = sum(1 for d in deltas if d > 1.5)

    # RMSD distribution (exclude rank 1 because Vina RMSD is relative to best mode)
    alt_rows = [r for r in rows if r["rank"] != 1]
    if not alt_rows and len(rows) > 1:
        alt_rows = rows[1:]
    lb_vals = [r["rmsd_lb"] for r in alt_rows if r["rmsd_lb"] is not None]
    ub_vals = [r["rmsd_ub"] for r in alt_rows if r["rmsd_ub"] is not None]
    lb_stats = _summarize_distribution(lb_vals) if lb_vals else None
    ub_stats = _summarize_distribution(ub_vals) if ub_vals else None
    lb_similar = sum(1 for x in lb_vals if x < 2.0)
    lb_shifted = sum(1 for x in lb_vals if 2.0 <= x <= 4.0)
    lb_distinct = sum(1 for x in lb_vals if x > 4.0)
    ub_tight = sum(1 for x in ub_vals if x < 3.0)
    ub_mid = sum(1 for x in ub_vals if 3.0 <= x <= 6.0)
    ub_wide = sum(1 for x in ub_vals if x > 6.0)

    def fmt(x, nd=3, unit=""):
        if x is None:
            return "N/A"
        return f"{x:.{nd}f}{unit}"

    lines = []
    if is_zh:
        lines.append(f"对接完成。共得到 **{n_pose}** 个构象，最佳结合能为 **{best_aff:.3f} kcal/mol**。")
        lines.append("")
        lines.append("**1) 对接打分分布（Affinity, kcal/mol）**")
        lines.append(
            f"- 最优 / 中位 / 最差: {fmt(aff_stats['min'])} / {fmt(aff_stats['median'])} / {fmt(aff_stats['max'])}"
        )
        lines.append(
            f"- 平均值 ± 标准差: {fmt(aff_stats['mean'])} ± {fmt(aff_stats['std'])}，分布跨度: {fmt(spread)}"
        )
        lines.append(
            f"- 与最佳构象相比 ΔAffinity ≤0.5 / 0.5–1.5 / >1.5 kcal/mol 的构象数: "
            f"{close_count} / {medium_count} / {weak_count}"
        )
        lines.append("")
        lines.append("**2) RMSD l.b 分布（相对最佳构象）**")
        if lb_stats:
            lines.append(
                f"- min / median / max: {fmt(lb_stats['min'])} / {fmt(lb_stats['median'])} / {fmt(lb_stats['max'])} Å"
            )
            lines.append(
                f"- <2 Å（近似同一结合模式）/ 2–4 Å（中等偏移）/ >4 Å（明显不同）: "
                f"{lb_similar} / {lb_shifted} / {lb_distinct}"
            )
        else:
            lines.append("- 仅有最佳构象或缺少 RMSD l.b 数据，无法统计分布。")
        lines.append("")
        lines.append("**3) RMSD u.b 分布（相对最佳构象）**")
        if ub_stats:
            lines.append(
                f"- min / median / max: {fmt(ub_stats['min'])} / {fmt(ub_stats['median'])} / {fmt(ub_stats['max'])} Å"
            )
            lines.append(
                f"- <3 Å（构象差异较集中）/ 3–6 Å（中等离散）/ >6 Å（离散较大）: "
                f"{ub_tight} / {ub_mid} / {ub_wide}"
            )
        else:
            lines.append("- 仅有最佳构象或缺少 RMSD u.b 数据，无法统计分布。")
        lines.append("")
        lines.append("**4) 代表意义（如何解读）**")
        lines.append("- Affinity 越负，预测结合越有利；但能量差 <0.5 kcal/mol 通常视为竞争性接近。")
        lines.append("- RMSD l.b 反映“至少有多相似”，RMSD u.b 反映“最多可能有多不同”。")
        lines.append("- 若低能构象同时 RMSD 分散较大，提示存在多个可能结合姿态，建议结合 2D/3D 相互作用进一步筛选。")
        lines.append("")
        lines.append(f"**Top {min(top_n, n_pose)} 构象概览**")
    else:
        lines.append(f"Docking finished. **{n_pose}** poses were generated; best affinity is **{best_aff:.3f} kcal/mol**.")
        lines.append("")
        lines.append("**1) Affinity Distribution (kcal/mol)**")
        lines.append(
            f"- Best / Median / Worst: {fmt(aff_stats['min'])} / {fmt(aff_stats['median'])} / {fmt(aff_stats['max'])}"
        )
        lines.append(
            f"- Mean ± SD: {fmt(aff_stats['mean'])} ± {fmt(aff_stats['std'])}; spread: {fmt(spread)}"
        )
        lines.append(
            f"- Pose counts with ΔAffinity vs best at ≤0.5 / 0.5–1.5 / >1.5 kcal/mol: "
            f"{close_count} / {medium_count} / {weak_count}"
        )
        lines.append("")
        lines.append("**2) RMSD l.b Distribution (vs best mode)**")
        if lb_stats:
            lines.append(
                f"- min / median / max: {fmt(lb_stats['min'])} / {fmt(lb_stats['median'])} / {fmt(lb_stats['max'])} Å"
            )
            lines.append(
                f"- <2 Å (near-similar binding mode) / 2–4 Å (moderate shift) / >4 Å (distinct mode): "
                f"{lb_similar} / {lb_shifted} / {lb_distinct}"
            )
        else:
            lines.append("- Only one pose or missing RMSD l.b data; distribution is unavailable.")
        lines.append("")
        lines.append("**3) RMSD u.b Distribution (vs best mode)**")
        if ub_stats:
            lines.append(
                f"- min / median / max: {fmt(ub_stats['min'])} / {fmt(ub_stats['median'])} / {fmt(ub_stats['max'])} Å"
            )
            lines.append(
                f"- <3 Å (compact) / 3–6 Å (moderately dispersed) / >6 Å (highly dispersed): "
                f"{ub_tight} / {ub_mid} / {ub_wide}"
            )
        else:
            lines.append("- Only one pose or missing RMSD u.b data; distribution is unavailable.")
        lines.append("")
        lines.append("**4) Practical Interpretation**")
        lines.append("- More negative affinity suggests more favorable binding, but differences <0.5 kcal/mol are usually close.")
        lines.append("- RMSD l.b reflects minimum possible similarity; RMSD u.b reflects maximum possible divergence.")
        lines.append("- If low-energy poses are also RMSD-dispersed, multiple plausible binding modes may exist; inspect 2D/3D contacts.")
        lines.append("")
        lines.append(f"**Top {min(top_n, n_pose)} Pose Snapshot**")

    for r in rows[:top_n]:
        delta = r["affinity"] - best_aff
        if is_zh:
            lines.append(
                f"- Pose {r['rank']}: Affinity {fmt(r['affinity'])} kcal/mol, "
                f"RMSD l.b {fmt(r['rmsd_lb'])} Å, RMSD u.b {fmt(r['rmsd_ub'])} Å, "
                f"ΔAffinity(best) {delta:+.3f}"
            )
        else:
            lines.append(
                f"- Pose {r['rank']}: Affinity {fmt(r['affinity'])} kcal/mol, "
                f"RMSD l.b {fmt(r['rmsd_lb'])} Å, RMSD u.b {fmt(r['rmsd_ub'])} Å, "
                f"ΔAffinity(best) {delta:+.3f}"
            )

    return "\n".join(lines)

def run_docking(center_x, center_y, center_z, size_x, size_y, size_z, exhaustiveness, lang: str = "English"):
    """Run vina docking using prepared files from docking_preparation."""
    try:
        tmp_dir = _DOCKING_STATE.get("tmp_dir")
        lig_pdbqt = _DOCKING_STATE.get("ligand_pdbqt")
        rec_pdbqt = _DOCKING_STATE.get("receptor_pdbqt")
        if not tmp_dir or not lig_pdbqt or not rec_pdbqt:
            return {"error": "Please run docking preparation first (docking_preparation)."}

        # Lock the docking box to the full receptor for global docking.
        receptor_path = _DOCKING_STATE.get("receptor_path")
        if receptor_path:
            try:
                center, size = _compute_box(Path(receptor_path))
                center_x, center_y, center_z = [float(x) for x in center]
                size_x, size_y, size_z = [float(x) for x in size]
            except Exception:
                # Fallback to provided values if auto box computation fails.
                pass

        cmd = [
            Config.VINA,
            "--receptor", rec_pdbqt.name,
            "--ligand", lig_pdbqt.name,
            "--out", "out.pdbqt",
            "--center_x", f"{float(center_x):.3f}",
            "--center_y", f"{float(center_y):.3f}",
            "--center_z", f"{float(center_z):.3f}",
            "--size_x", f"{float(size_x):.3f}",
            "--size_y", f"{float(size_y):.3f}",
            "--size_z", f"{float(size_z):.3f}",
            "--exhaustiveness", str(int(exhaustiveness)),
        ]
        _run_cmd(cmd, cwd=tmp_dir, label="vina", log_file="vina.log")
        out_pdbqt = tmp_dir / "out.pdbqt"
        scores = _parse_vina_scores(out_pdbqt)

        # Generate a PDB for visualization
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # 单独输出配体供 Molstar 双轨加载
        docked_ligand_pdb = Config.OUTPUT_DIR / "docked_ligand.pdb"
        _convert_pdbqt_to_pdb(out_pdbqt, docked_ligand_pdb)

        # Split poses -> build complexes -> convert to other format for 2D tools
        pose_results = []
        try:
            receptor_path = Path(receptor_path) if receptor_path else None
            if receptor_path and receptor_path.exists() and out_pdbqt.exists():
                poses_dir = Config.OUTPUT_DIR / "Docking_Poses"
                pose_results = prepare_pose_complexes(
                    receptor_path,
                    out_pdbqt,
                    poses_dir,
                    convert_format="mol2",
                )
        except Exception:
            pose_results = []

        analysis = _build_docking_analysis_report(scores, lang=lang)

        return {
            "docking_scores": scores,
            "analysis_result": analysis,
            "pose_results": pose_results,
        }
    except Exception as e:
        return {"error": f"Docking error: {str(e)}"}

# ========== 辅助功能 (PubChem & AI) ==========
import requests
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors

# 靶点详细生理效应描述库
TARGET_EFFECTS = {
    'ESR1': """雌激素受体α (ERα) 生理效应：
1) 生殖系统：介导子宫内膜增生、乳腺导管发育，过度激活与 ER+ 乳腺/子宫内膜癌相关。
2) 骨骼健康：促进成骨、抑制破骨，维持骨密度。
3) 代谢调节：参与葡萄糖稳态、脂质代谢，具抗肥胖作用。
4) 心血管：促进一氧化氮生成、扩张血管，具血管保护作用。
5) 神经系统：可能影响认知和情绪调节。
潜在风险：拮抗剂可致潮热、骨质流失；激动剂可能增加血栓和增殖相关风险。""",
    'ESR2': """雌激素受体β (ERβ) 生理效应：
1) 抗增殖：在乳腺/子宫/前列腺等组织拮抗 ERα 的增殖效应，具潜在肿瘤抑制作用。
2) 神经系统：参与认知保护、抗焦虑/抗抑郁、神经元存活。
3) 免疫与炎症：调节免疫反应和炎症通路。
4) 心血管：参与血压和血管反应性的调控。
5) 代谢：可能与代谢和能量平衡相关。
潜在风险：选择性调节剂长期安全性仍在研究，组织特异性作用复杂。""",
    'GPER1': """G 蛋白偶联雌激素受体1 (GPER1/GPR30) 生理效应：
1) 快速血管效应：介导快速血管舒张，调节血压。
2) 代谢调节：促进胰岛β细胞分泌胰岛素，参与糖脂代谢。
3) 细胞生长：在部分乳腺/卵巢癌中上调，可能促进增殖与迁移。
4) 神经系统：参与神经保护和痛觉调节。
5) 免疫与炎症：可能影响炎症相关信号。
潜在风险：在肿瘤进展中的作用具两面性，需具体分析，某些情况下可能促肿瘤。"""
}

TARGET_INTRO = {
    "ESR1": {
        "title": "雌激素受体α (ERα / ESR1)",
        "physiology": [
            "生殖系统：调控子宫内膜增生与乳腺发育，参与生殖轴与第二性征维持。",
            "骨骼：抑制骨吸收并维持骨密度，参与骨重塑平衡。",
            "心血管：影响血管张力与内皮功能，参与血压/血流动力学调控。",
            "代谢：参与糖脂代谢与体脂分布调节。",
            "神经：参与情绪、认知及部分神经保护通路。",
        ],
        "inhibit": {
            "value": [
                "用于雌激素依赖性疾病的治疗策略（如 ER+ 乳腺癌的内分泌治疗：拮抗/降解 ERα）。",
                "在某些增殖相关疾病中降低雌激素信号驱动的组织增生风险（需结合组织特异性与药物类型评估）。",
            ],
            "harm": [
                "类绝经样不良反应：潮热、情绪/睡眠受影响、泌尿生殖道萎缩相关症状。",
                "骨密度下降、骨折风险上升（长期抑制时更需关注）。",
                "代谢与心血管风险谱可能改变（与个体基础风险和药物类别相关）。",
            ],
        },
        "activate": {
            "value": [
                "缓解绝经相关血管舒缩症状并改善泌尿生殖道萎缩（激素治疗/选择性调节剂场景）。",
                "骨保护：降低骨丢失与骨折风险（需权衡组织选择性与疗程）。",
                "部分代谢/血管获益可能存在（依赖剂量、给药方式与人群）。",
            ],
            "harm": [
                "增殖相关风险：可能增加乳腺/子宫内膜等雌激素敏感组织的增殖与肿瘤风险（与方案相关）。",
                "血栓栓塞/卒中风险可能上升（尤其在特定人群或口服雌激素方案）。",
            ],
        },
    },
    "ESR2": {
        "title": "雌激素受体β (ERβ / ESR2)",
        "physiology": [
            "在多组织中对炎症、细胞增殖与分化具有调控作用，部分情况下可拮抗 ERα 的增殖信号。",
            "神经：参与神经保护、情绪与认知相关通路。",
            "免疫与炎症：调控免疫反应与炎症信号网络。",
            "心血管与代谢：参与血管反应性及能量代谢相关调控（组织/情境依赖）。",
        ],
        "inhibit": {
            "value": [
                "临床价值依赖疾病背景：在少数疾病/肿瘤情境下，抑制 ERβ 可能具有潜在治疗意义（证据与适应证仍在演进）。",
            ],
            "harm": [
                "可能削弱其在抗炎/神经保护等方面的潜在保护作用，带来情绪、认知或炎症相关风险上升的可能性。",
                "长期安全性与组织特异性效应存在不确定性。",
            ],
        },
        "activate": {
            "value": [
                "潜在抗增殖/抗炎获益：在部分组织中可能抑制 ERα 相关增殖信号，具有疾病修饰潜力（研究与适应证需具体评估）。",
                "神经系统潜在获益：可能具有抗焦虑/神经保护方向的研究与转化价值。",
            ],
            "harm": [
                "组织特异性效应复杂：不同组织/剂量/选择性决定获益与风险，长期安全性仍需更多证据。",
            ],
        },
    },
    "GPER1": {
        "title": "G 蛋白偶联雌激素受体 1 (GPER1 / GPR30)",
        "physiology": [
            "介导部分雌激素的快速非基因组效应，参与血管舒张与血流动力学调控。",
            "代谢：影响胰岛素分泌与糖脂代谢相关信号（情境依赖）。",
            "神经与免疫：参与神经保护、痛觉与炎症相关通路调控。",
            "细胞生长：在部分肿瘤中可能参与增殖/迁移信号，作用具两面性。",
        ],
        "inhibit": {
            "value": [
                "在某些肿瘤或增殖/迁移相关情境下，抑制 GPER1 可能具有潜在治疗价值（需结合具体肿瘤类型与证据）。",
            ],
            "harm": [
                "可能削弱其在血管与代谢方面的潜在有益效应，影响血管反应性或代谢稳态的风险需关注。",
            ],
        },
        "activate": {
            "value": [
                "潜在血管与代谢获益：可能促进血管舒张、改善部分代谢相关通路（取决于剂量与人群）。",
                "神经系统方向可能具有研究与转化价值（神经保护/痛觉调节等）。",
            ],
            "harm": [
                "在特定肿瘤背景下可能促进增殖或迁移，存在促肿瘤风险（高度情境依赖）。",
                "血流动力学改变相关不良反应可能存在（个体差异明显）。",
            ],
        },
    },
}

TARGET_INTRO_EN = {
    "ESR1": {
        "title": "Estrogen receptor alpha (ERα / ESR1)",
        "physiology": [
            "Reproductive system: regulates endometrial proliferation and mammary development; supports reproductive axis and secondary sex characteristics.",
            "Bone: suppresses bone resorption and maintains bone density; involved in bone remodeling balance.",
            "Cardiovascular: modulates vascular tone and endothelial function; participates in blood pressure/hemodynamics.",
            "Metabolism: involved in glucose/lipid metabolism and fat distribution.",
            "Neuro: participates in mood, cognition, and some neuroprotective pathways.",
        ],
        "inhibit": {
            "value": [
                "Therapeutic strategy for estrogen-dependent diseases (e.g., ER+ breast cancer endocrine therapy via ERα antagonism/degradation).",
                "May reduce estrogen-driven hyperplasia in certain proliferative conditions (context- and tissue-specific).",
            ],
            "harm": [
                "Menopause-like adverse effects: hot flashes, mood/sleep changes, and urogenital atrophy symptoms.",
                "Reduced bone density and increased fracture risk (especially with long-term suppression).",
                "Potential shifts in metabolic and cardiovascular risk profiles (dependent on baseline risk and drug class).",
            ],
        },
        "activate": {
            "value": [
                "Relief of menopausal vasomotor symptoms and urogenital atrophy (hormone therapy/selected modulators).",
                "Bone protection: reduced bone loss and fracture risk (balanced by tissue selectivity and duration).",
                "Possible metabolic/vascular benefits (dose-, route-, and population-dependent).",
            ],
            "harm": [
                "Proliferation-related risk: may increase proliferation and tumor risk in estrogen-sensitive tissues (regimen-dependent).",
                "Possible increase in thromboembolism/stroke risk (especially in specific populations or oral estrogen regimens).",
            ],
        },
    },
    "ESR2": {
        "title": "Estrogen receptor beta (ERβ / ESR2)",
        "physiology": [
            "Regulates inflammation, proliferation, and differentiation across tissues; can counter ERα proliferative signaling in some contexts.",
            "Neuro: involved in neuroprotection, mood and cognition pathways.",
            "Immune/inflammation: modulates immune response and inflammatory signaling.",
            "Cardiovascular/metabolic: influences vascular reactivity and energy metabolism (context dependent).",
        ],
        "inhibit": {
            "value": [
                "Clinical value depends on disease context: in select tumor settings, ERβ inhibition may have therapeutic potential (evidence and indications are evolving).",
            ],
            "harm": [
                "May weaken anti-inflammatory and neuroprotective effects, with possible mood, cognition, or inflammation-related risks.",
                "Long-term safety and tissue-specific effects remain uncertain.",
            ],
        },
        "activate": {
            "value": [
                "Potential anti-proliferative/anti-inflammatory benefits: may counter ERα-driven proliferation in some tissues (requires context-specific evaluation).",
                "Possible neuroprotective/anxiolytic benefits in research and translational settings.",
            ],
            "harm": [
                "Complex tissue-specific effects: benefits and risks depend on tissue, dose, and selectivity; long-term safety needs more evidence.",
            ],
        },
    },
    "GPER1": {
        "title": "G protein-coupled estrogen receptor 1 (GPER1 / GPR30)",
        "physiology": [
            "Mediates rapid non-genomic estrogen effects; involved in vasodilation and hemodynamic control.",
            "Metabolism: affects insulin secretion and glucose/lipid signaling (context dependent).",
            "Neuro/immune: participates in neuroprotection, pain and inflammation pathways.",
            "Cell growth: may contribute to proliferation/migration signaling in some tumors; context dependent.",
        ],
        "inhibit": {
            "value": [
                "In certain tumors or proliferation/migration contexts, GPER1 inhibition may have therapeutic value (dependent on tumor type and evidence).",
            ],
            "harm": [
                "May reduce potential vascular and metabolic benefits; risks to vascular reactivity or metabolic homeostasis should be considered.",
            ],
        },
        "activate": {
            "value": [
                "Potential vascular and metabolic benefits: may promote vasodilation and improve some metabolic pathways (dose- and population-dependent).",
                "Possible neuroprotective/analgesic research value.",
            ],
            "harm": [
                "In specific tumor contexts, activation may promote proliferation/migration with pro-tumor risk (highly context dependent).",
                "Potential hemodynamic adverse effects (individual variability).",
            ],
        },
    },
}

def render_target_intro_html(target_name: str, has_toxic_alerts: bool, lang: str = "中文") -> str:
    """Render target intro with conditional clinical value vs adverse reactions."""
    is_zh = str(lang).strip() == "中文" or str(lang).lower().startswith("zh")
    if not target_name:
        return "<p><i>暂无靶点信息</i></p>" if is_zh else "<p><i>No target information available.</i></p>"

    key = str(target_name).strip()
    if key.lower() == "decoy":
        if is_zh:
            return "<p><b>非雌激素受体靶向性：</b>预测为 Decoy，提示该化合物可能不靶向 ESR1/ESR2/GPER1。</p>"
        return "<p><b>Non-ER targeting:</b> Predicted as Decoy, suggesting the compound is unlikely to target ESR1/ESR2/GPER1.</p>"

    info = TARGET_INTRO.get(key) if is_zh else TARGET_INTRO_EN.get(key)
    if not info:
        # Fallback to legacy text if present (Chinese only)
        if is_zh:
            legacy = TARGET_EFFECTS.get(key)
            if legacy:
                legacy_html = "<br>".join([s.strip() for s in str(legacy).splitlines() if s.strip()])
                return f"<div><h4>靶点生理作用</h4><p>{legacy_html}</p></div>"
            return f"<p><i>暂无 {key} 的靶点介绍</i></p>"
        return f"<p><i>No introduction available for {key}.</i></p>"

    physiology_items = "\n".join([f"<li>{x}</li>" for x in info.get("physiology", [])])

    if has_toxic_alerts:
        inhibit_title = "靶点被抑制后的有害反应" if is_zh else "Adverse effects when inhibited"
        activate_title = "靶点被激活后的有害反应" if is_zh else "Adverse effects when activated"
        inhibit_items = info.get("inhibit", {}).get("harm", [])
        activate_items = info.get("activate", {}).get("harm", [])
    else:
        inhibit_title = "靶点被抑制后的临床价值" if is_zh else "Potential clinical value when inhibited"
        activate_title = "靶点被激活后的临床价值" if is_zh else "Potential clinical value when activated"
        inhibit_items = info.get("inhibit", {}).get("value", [])
        activate_items = info.get("activate", {}).get("value", [])

    inhibit_html = "\n".join([f"<li>{x}</li>" for x in inhibit_items]) or "<li>暂无</li>"
    activate_html = "\n".join([f"<li>{x}</li>" for x in activate_items]) or "<li>暂无</li>"
    if not is_zh:
        if "<li>暂无</li>" in inhibit_html:
            inhibit_html = "<li>None</li>"
        if "<li>暂无</li>" in activate_html:
            activate_html = "<li>None</li>"

    return f"""
<div>
  <h4>{info.get('title','')}</h4>
  <h4>{"靶点生理作用" if is_zh else "Target Physiology"}</h4>
  <ul>{physiology_items}</ul>
  <h4>{inhibit_title}</h4>
  <ul>{inhibit_html}</ul>
  <h4>{activate_title}</h4>
  <ul>{activate_html}</ul>
</div>
""".strip()

# PubChem request helpers (retry + friendly errors)
_PUBCHEM_HEADERS = {"User-Agent": "SRRSH-ER/1.0"}
_PUBCHEM_RETRY_STATUS = {429, 500, 502, 503, 504}
_PUBCHEM_CACHE_DIR = None
_PUBCHEM_CACHE_ENABLED = os.getenv("PUBCHEM_CACHE", "0").lower() in {"1", "true", "yes", "y"}
_PUBCHEM_MAX_WAIT_SECONDS = int(os.getenv("PUBCHEM_MAX_WAIT_SECONDS", "60"))

def _pubchem_cache_dir():
    global _PUBCHEM_CACHE_DIR
    if _PUBCHEM_CACHE_DIR is None:
        try:
            base = Config.OUTPUT_DIR / "PubChemCache"
        except Exception:
            base = Path(".") / "Output" / "PubChemCache"
        base.mkdir(parents=True, exist_ok=True)
        _PUBCHEM_CACHE_DIR = base
    return _PUBCHEM_CACHE_DIR

def _pubchem_cache_key(prefix: str, value: str):
    raw = f"{prefix}:{value}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()

def _pubchem_cache_get(prefix: str, value: str, max_age_seconds=None):
    if not _PUBCHEM_CACHE_ENABLED:
        return None
    try:
        key = _pubchem_cache_key(prefix, value)
        path = _pubchem_cache_dir() / f"{key}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if max_age_seconds is not None:
            ts = payload.get("ts", 0)
            if ts and (time.time() - ts) > max_age_seconds:
                return None
        return payload.get("data")
    except Exception:
        return None

def _pubchem_cache_set(prefix: str, value: str, data):
    if not _PUBCHEM_CACHE_ENABLED:
        return
    try:
        key = _pubchem_cache_key(prefix, value)
        path = _pubchem_cache_dir() / f"{key}.json"
        payload = {"ts": time.time(), "data": data}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _pubchem_request(method, url, *, data=None, params=None, timeout=10, retries=3):
    last_resp = None
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method,
                url,
                data=data,
                params=params,
                timeout=timeout,
                headers=_PUBCHEM_HEADERS,
            )
            last_resp = resp
        except Exception as e:
            last_err = str(e)
            resp = None

        if resp is not None and resp.status_code == 200:
            return resp, None

        retryable = False
        if resp is None:
            retryable = True
        elif resp.status_code in _PUBCHEM_RETRY_STATUS:
            retryable = True
            last_err = f"status {resp.status_code}"

        if not retryable:
            if resp is not None:
                last_err = last_err or f"status {resp.status_code}"
            return resp, last_err

        if attempt < retries:
            sleep_s = 0.8 * (1.7 ** attempt)
            if resp is not None:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(float(retry_after), sleep_s)
                    except Exception:
                        pass
            # add jitter to reduce thundering herd
            sleep_s = sleep_s + random.uniform(0, 0.6)
            time.sleep(min(sleep_s, 8.0))

    return last_resp, last_err or (f"status {last_resp.status_code}" if last_resp is not None else "request failed")

def _pubchem_request_wait(method, url, *, data=None, params=None, timeout=10, max_wait_seconds=None):
    """Retry on 429/5xx until success or max_wait_seconds (forced <=60s by default)."""
    start = time.time()
    attempt = 0
    max_wait = _PUBCHEM_MAX_WAIT_SECONDS if max_wait_seconds is None else max_wait_seconds
    # Enforce a hard stop (no infinite retry)
    if max_wait is None or max_wait <= 0:
        max_wait = 60
    while True:
        resp, err = _pubchem_request(method, url, data=data, params=params, timeout=timeout, retries=0)
        if resp is not None and resp.status_code == 200:
            return resp, None

        status = resp.status_code if resp is not None else None
        retryable = (resp is None) or (status in _PUBCHEM_RETRY_STATUS)
        if not retryable:
            return resp, err or (f"status {status}" if status else "request failed")

        if max_wait and (time.time() - start) >= max_wait:
            return resp, "busy_timeout"

        # Exponential backoff with jitter
        sleep_s = min(2.0 * (1.7 ** attempt), 30.0) + random.uniform(0, 1.0)
        time.sleep(sleep_s)
        attempt += 1

def _is_zh_lang(lang: str) -> bool:
    lab = str(lang or "").strip()
    if lab in {"中文", "简体中文"}:
        return True
    low = lab.lower()
    return low.startswith("zh") or ("chinese" in low) or ("中文" in lab)

def _pubchem_error_message(label, status=None, err=None):
    return _pubchem_error_message_lang(label, status=status, err=err, lang="中文")

def _pubchem_error_message_lang(label, status=None, err=None, lang: str = "中文"):
    is_zh = _is_zh_lang(lang)
    if err == "busy_timeout":
        return (
            f"{label}获取失败：PubChem 服务繁忙，已等待较长时间仍无响应，请稍后重试。"
            if is_zh
            else f"{label} failed: PubChem is busy and did not respond in time. Please try again later."
        )
    if status == 404:
        return (
            f"{label}获取失败：PubChem 中未找到对应数据。"
            if is_zh
            else f"{label} failed: no matching record found on PubChem."
        )
    if status in (429, 503):
        return (
            f"{label}获取失败：PubChem 服务繁忙 (HTTP {status})，请稍后重试。"
            if is_zh
            else f"{label} failed: PubChem is busy (HTTP {status}). Please try again later."
        )
    if status:
        return (
            f"{label}获取失败：PubChem 请求错误 (HTTP {status})，请稍后重试。"
            if is_zh
            else f"{label} failed: PubChem request error (HTTP {status}). Please try again later."
        )
    if err:
        return (
            f"{label}获取失败：网络连接异常，无法连接 PubChem。"
            if is_zh
            else f"{label} failed: network error; cannot connect to PubChem."
        )
    return (
        f"{label}获取失败：PubChem 服务暂时不可用。"
        if is_zh
        else f"{label} failed: PubChem service unavailable."
    )

def _fetch_pubchem_properties(smiles: str, props=None):
    if not smiles:
        return None, "empty smiles", None
    prop_list = props or [
        "Title",
        "IUPACName",
        "MolecularFormula",
        "MolecularWeight",
        "ExactMass",
        "CanonicalSMILES",
        "IsomericSMILES",
        "InChIKey",
        "XLogP",
        "HBondDonorCount",
        "HBondAcceptorCount",
        "RotatableBondCount",
        "TPSA",
        "CID",
    ]
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/{','.join(prop_list)}/JSON"
    cache_key = f"{smiles}|{','.join(prop_list)}"
    cached = _pubchem_cache_get("pug_props", cache_key, max_age_seconds=60 * 60 * 24 * 30)
    if cached:
        return cached, None, None
    resp, err = _pubchem_request_wait("POST", url, data={"smiles": smiles}, timeout=15)
    if resp is not None and resp.status_code == 200:
        try:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                _pubchem_cache_set("pug_props", cache_key, props[0])
                return props[0], None, None
            return None, "no properties", 404
        except Exception as e:
            return None, f"json error: {e}", resp.status_code
    status = resp.status_code if resp is not None else None
    return None, err, status

def _swm_to_html(items):
    if not items:
        return ""
    parts = []
    for item in items:
        if not isinstance(item, dict):
            if item:
                parts.append(html.escape(str(item)))
            continue
        s = item.get("String", "")
        if s:
            parts.append(html.escape(str(s)).replace("\n", "<br>"))
        for m in item.get("Markup", []) or []:
            if m.get("Type") == "Icon" and m.get("URL"):
                parts.append(f"<img src=\"{m['URL']}\" style=\"width:28px;margin:2px;vertical-align:middle;\">")
    return "<br>".join([p for p in parts if p])

def _value_to_html(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return html.escape(value).replace("\n", "<br>")
    if isinstance(value, (int, float, bool)):
        return html.escape(str(value))
    if isinstance(value, list):
        return "<br>".join([_value_to_html(v) for v in value if v is not None])
    if isinstance(value, dict):
        if "StringWithMarkup" in value:
            return _swm_to_html(value.get("StringWithMarkup") or [])
        if "String" in value:
            return html.escape(str(value.get("String"))).replace("\n", "<br>")
        if "Number" in value:
            return html.escape(str(value.get("Number")))
        if "Boolean" in value:
            return "Yes" if value.get("Boolean") else "No"
        if "Table" in value:
            return _table_to_html(value.get("Table") or {})
        if "ExternalData" in value:
            ext = value.get("ExternalData") or {}
            return html.escape(ext.get("URL", "") or ext.get("SourceName", "") or "External Data")
    return html.escape(str(value))

def _table_to_html(table):
    if not isinstance(table, dict):
        return ""
    columns = table.get("ColumnLabel") or table.get("ColumnHeader") or []
    rows = table.get("Row") or table.get("TableRow") or []
    row_labels = table.get("RowLabel") or []
    html_rows = []
    if columns:
        header_cells = "".join([f"<th style='border:1px solid #ddd;padding:4px 6px;background:#f7f7f7'>{html.escape(str(c))}</th>" for c in columns])
        if row_labels:
            header_cells = "<th style='border:1px solid #ddd;padding:4px 6px;background:#f7f7f7'></th>" + header_cells
        html_rows.append(f"<tr>{header_cells}</tr>")
    for idx, r in enumerate(rows):
        if isinstance(r, dict):
            cells = r.get("Cell") or r.get("TableCell") or []
        elif isinstance(r, list):
            cells = r
        else:
            cells = [r]
        cell_html = []
        if row_labels and idx < len(row_labels):
            cell_html.append(f"<td style='border:1px solid #ddd;padding:4px 6px;background:#fafafa'>{html.escape(str(row_labels[idx]))}</td>")
        for c in cells:
            cell_html.append(f"<td style='border:1px solid #ddd;padding:4px 6px'>{_value_to_html(c)}</td>")
        html_rows.append(f"<tr>{''.join(cell_html)}</tr>")
    if not html_rows:
        return ""
    return "<table style='border-collapse:collapse; width:100%; font-size:0.9em;'>" + "".join(html_rows) + "</table>"

def _extract_section(data, heading=None):
    if not data:
        return None
    sections = data.get("Record", {}).get("Section", [])
    if not sections:
        return None
    if heading:
        sec = _find_section_recursive(sections, heading)
        if sec:
            return sec
        target = heading.strip().lower()
        if target:
            stack = list(sections)
            while stack:
                cur = stack.pop(0)
                cur_head = (cur.get("TOCHeading") or "").strip().lower()
                if target in cur_head:
                    return cur
                for child in cur.get("Section") or []:
                    stack.append(child)
    if len(sections) == 1:
        return sections[0]
    return None

def _render_pubchem_section(data, heading=None, level=3):
    sec = _extract_section(data, heading)
    if not sec:
        return ""

    def walk(s, lvl):
        parts = []
        title = s.get("TOCHeading") or ""
        if title:
            h = min(max(lvl, 3), 6)
            parts.append(f"<h{h} style='margin:6px 0 4px 0;'>{html.escape(title)}</h{h}>")
        info_list = s.get("Information") or []
        for info in info_list:
            name = info.get("Name") or ""
            val = _value_to_html(info.get("Value"))
            if not val:
                continue
            if name:
                parts.append(f"<div style='margin:2px 0;'><b>{html.escape(name)}:</b> {val}</div>")
            else:
                parts.append(f"<div style='margin:2px 0;'>{val}</div>")
        for child in s.get("Section") or []:
            parts.append(walk(child, lvl + 1))
        return "".join(parts)

    return walk(sec, level)

def _value_to_text(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "; ".join([_value_to_text(v) for v in value if v is not None])
    if isinstance(value, dict):
        if "StringWithMarkup" in value:
            items = value.get("StringWithMarkup") or []
            texts = []
            for item in items:
                if isinstance(item, dict) and item.get("String"):
                    texts.append(str(item.get("String")))
            return "; ".join([t for t in texts if t])
        if "String" in value:
            return str(value.get("String"))
        if "Number" in value:
            return str(value.get("Number"))
        if "Boolean" in value:
            return "Yes" if value.get("Boolean") else "No"
    return str(value)

def _section_to_text(data, heading=None, limit=4000):
    sec = _extract_section(data, heading)
    if not sec:
        return ""
    lines = []

    def walk(s, lvl):
        title = s.get("TOCHeading") or ""
        if title:
            lines.append(("  " * lvl) + title)
        info_list = s.get("Information") or []
        for info in info_list:
            name = info.get("Name") or ""
            val = _value_to_text(info.get("Value"))
            if not val:
                continue
            if name:
                lines.append(("  " * lvl) + f"{name}: {val}")
            else:
                lines.append(("  " * lvl) + val)
        for child in s.get("Section") or []:
            walk(child, lvl + 1)

    walk(sec, 0)
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit] + "..."
    return text

def get_molecule_info(smiles):
    """从 PubChem 获取化合物信息"""
    if not smiles: return None
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    info = {
        'pubchem_name': 'Unknown',
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'mol_weight': Descriptors.ExactMolWt(mol),
        'image': Draw.MolToImage(mol, size=(300, 300))
    }
    # Toxic functional group check (local SMARTS)
    alerts = detect_toxic_alerts(mol)
    info['toxic_alerts'] = alerts
    info['toxic_flag'] = bool(alerts)
    try:
        props, err, status = _fetch_pubchem_properties(smiles)
        if props:
            info['pubchem_name'] = props.get('Title', info['pubchem_name'])
            info['iupac_name'] = props.get('IUPACName')
            info['pubchem_formula'] = props.get('MolecularFormula')
            info['pubchem_mol_weight'] = props.get('MolecularWeight')
            info['pubchem_exact_mass'] = props.get('ExactMass')
            info['canonical_smiles'] = props.get('CanonicalSMILES')
            info['isomeric_smiles'] = props.get('IsomericSMILES')
            info['inchikey'] = props.get('InChIKey')
            info['xlogp'] = props.get('XLogP')
            info['hbd'] = props.get('HBondDonorCount')
            info['hba'] = props.get('HBondAcceptorCount')
            info['rotatable_bonds'] = props.get('RotatableBondCount')
            info['tpsa'] = props.get('TPSA')
            info['cid'] = props.get('CID')
            if info.get('cid'):
                info['pubchem_url'] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{info['cid']}"
        if not info.get('iupac_name'):
            iupac, _ = get_iupac_name(smiles)
            if iupac:
                info['iupac_name'] = iupac
        else:
            if status:
                print(f"PubChem API Warning: status {status}")
            elif err:
                print(f"PubChem API Warning: {err}")
    except Exception as e:
        print(f"PubChem API Warning: {e}")
    return info

def get_iupac_name(smiles: str):
    """Fetch IUPAC name from PubChem for a given SMILES."""
    if not smiles:
        return None, "empty smiles"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "invalid smiles"
    except Exception:
        return None, "invalid smiles"

    last_err = None
    try:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/IUPACName/JSON"
        resp, err = _pubchem_request("POST", url, data={"smiles": smiles}, timeout=10, retries=1)
        if resp is None or resp.status_code != 200:
            last_err = f"PubChem error {resp.status_code if resp is not None else err}"
        else:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props and "IUPACName" in props[0]:
                return props[0]["IUPACName"], None
            last_err = "IUPAC not found"
    except Exception as e:
        last_err = str(e)

    # Fallback: CACTUS (NCI)
    try:
        q = quote(smiles, safe="")
        url = f"https://cactus.nci.nih.gov/chemical/structure/{q}/iupac_name"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            text = (resp.text or "").strip()
            if text and "not found" not in text.lower() and "error" not in text.lower():
                return text, None
        last_err = last_err or f"CACTUS error {resp.status_code}"
    except Exception as e:
        last_err = last_err or str(e)

    return None, last_err or "IUPAC not found"

def _fetch_pubchem_cid(smiles: str):
    """Fetch CID via PUG REST (properties preferred), with cache."""
    if not smiles:
        return None, "empty smiles", None
    props, err, status = _fetch_pubchem_properties(smiles, props=["CID"])
    if props and props.get("CID"):
        return props.get("CID"), None, None
    # Fallback to PUG REST cids endpoint
    cache_key = f"{smiles}"
    cached = _pubchem_cache_get("pug_cid", cache_key, max_age_seconds=60 * 60 * 24 * 30)
    if cached:
        return cached, None, None
    search_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON"
    resp, err = _pubchem_request_wait("POST", search_url, data={"smiles": smiles}, timeout=10)
    if resp is not None and resp.status_code == 200:
        try:
            data = resp.json()
            cid_list = data.get("IdentifierList", {}).get("CID", [])
            if cid_list:
                cid = cid_list[0]
                _pubchem_cache_set("pug_cid", cache_key, cid)
                return cid, None, None
        except Exception as e:
            return None, f"json error: {e}", resp.status_code
    status = resp.status_code if resp is not None else None
    return None, err, status

def canonicalize_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None

def check_plip_tool(lang: str = "中文"):
    is_zh = _is_zh_lang(lang)
    if not _cmd_exists(Config.PLIP_CMD):
        return (
            False,
            (f"PLIP 未安装或未在 PATH 中 (PLIP_CMD={Config.PLIP_CMD})"
             if is_zh
             else f"PLIP is not installed or not found in PATH (PLIP_CMD={Config.PLIP_CMD})"),
        )
    return True, ""

def _fetch_pubchem_section(cid, heading):
    """Helper to fetch specific section from PubChem PUG VIEW"""
    from urllib.parse import quote
    heading_q = quote(str(heading), safe="%")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading={heading_q}"
    cache_key = f"{cid}:{heading}".lower()
    cached = _pubchem_cache_get("pug_view_section", cache_key, max_age_seconds=60 * 60 * 24 * 30)
    if cached:
        return cached, None, None
    # Increased timeout to 30s for PubChem
    resp, err = _pubchem_request_wait("GET", url, timeout=30)
    if resp is not None and resp.status_code == 200:
        try:
            data = resp.json()
            _pubchem_cache_set("pug_view_section", cache_key, data)
            return data, None, None
        except Exception as e:
            return None, f"json error: {e}", resp.status_code
    status = resp.status_code if resp is not None else None
    if status:
        print(f"PubChem API Error ({heading}): Status {status}")
    elif err:
        print(f"PubChem API Request Failed ({heading}): {err}")
    return None, err, status

def _fetch_pubchem_record(cid):
    """Fetch full PubChem PUG VIEW record (fallback)."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    cache_key = f"{cid}".lower()
    cached = _pubchem_cache_get("pug_view_record", cache_key, max_age_seconds=60 * 60 * 24 * 30)
    if cached:
        return cached, None, None
    resp, err = _pubchem_request_wait("GET", url, timeout=30)
    if resp is not None and resp.status_code == 200:
        try:
            data = resp.json()
            _pubchem_cache_set("pug_view_record", cache_key, data)
            return data, None, None
        except Exception as e:
            return None, f"json error: {e}", resp.status_code
    status = resp.status_code if resp is not None else None
    return None, err, status

def _find_section_recursive(sections, target_name):
    """Recursively find a section by TOCHeading"""
    target = (target_name or "").strip().lower()
    for sec in sections:
        heading = (sec.get('TOCHeading') or '').strip().lower()
        if heading == target:
            return sec
        if 'Section' in sec:
            found = _find_section_recursive(sec['Section'], target_name)
            if found: return found
    return None

def _parse_ghs_data(data, lang: str = "中文"):
    """Parse GHS Classification from PubChem PUG-View JSON into (HTML, text)."""
    is_zh = _is_zh_lang(lang)
    if not data:
        return ("未找到 GHS 数据 (数据为空)" if is_zh else "GHS data not found (empty record)"), ""
    
    html_parts = []
    text_parts = []
    
    try:
        root_sections = data.get('Record', {}).get('Section', [])
        ghs_sec = _find_section_recursive(root_sections, 'GHS Classification')
        
        if not ghs_sec:
            return (
                "未找到 GHS 分类 (无 GHS Classification 章节)"
                if is_zh
                else "GHS Classification section not found"
            ), ""
        
        info_list = ghs_sec.get('Information', [])
        for info in info_list:
            name = info.get('Name')
            val = info.get('Value', {}).get('StringWithMarkup', [])
            
            if name == 'Pictogram(s)':
                img_html = ""
                for item in val:
                    for m in item.get('Markup', []):
                        if m.get('Type') == 'Icon' and 'URL' in m:
                            img_html += f'<img src="{m["URL"]}" style="width:50px;margin:5px;">'
                if img_html:
                    label = "象形图" if is_zh else "Pictograms"
                    html_parts.append(f"<div><b>{label}:</b><br>{img_html}</div>")
            
            if name == 'Signal':
                if val and 'String' in val[0]:
                    sig = val[0]['String']
                    label = "信号词" if is_zh else "Signal word"
                    html_parts.append(f"<div><b>{label}:</b> {sig}</div>")
                    text_parts.append(f"{label}: {sig}")

            if name == 'GHS Hazard Statements':
                stmts = [s.get('String', '') for s in val]
                if stmts:
                    label = "危害声明" if is_zh else "Hazard statements"
                    html_parts.append(
                        "<div><b>"
                        + f"{label}:"
                        + "</b><ul>"
                        + "".join([f"<li>{s}</li>" for s in stmts[:8]])
                        + "</ul></div>"
                    )
                    text_parts.append(f"{label}: " + "; ".join(stmts))
        
        if not html_parts:
            return (
                "找到 GHS 章节但无详细数据"
                if is_zh
                else "GHS section found but no detailed items"
            ), ""
            
        return "".join(html_parts), "\n".join(text_parts)
    except Exception as e:
        return (f"解析 GHS 出错: {str(e)}" if is_zh else f"Failed to parse GHS: {str(e)}"), ""

def _is_ghs_missing(ghs_html: str) -> bool:
    if not ghs_html:
        return True
    text = str(ghs_html)
    markers = [
        "未找到 GHS 数据",
        "未找到 GHS 分类",
        "找到 GHS 章节但无详细数据",
        "解析 GHS 出错",
        "GHS data not found",
        "GHS Classification section not found",
        "GHS section found but no detailed items",
        "Failed to parse GHS",
    ]
    return any(m in text for m in markers)

def _parse_toxicity_data(data, lang: str = "中文"):
    """Parse Toxicological Information (lightweight index-only summary)."""
    is_zh = _is_zh_lang(lang)
    if not data:
        return ("未找到详细毒理学数据 (数据为空)" if is_zh else "Toxicity data not found (empty record)"), ""
    try:
        root_sections = data.get('Record', {}).get('Section', [])
        tox_sec = _find_section_recursive(root_sections, 'Toxicological Information')
        
        if not tox_sec:
            return ("无毒理学信息章节" if is_zh else "Toxicological Information section not found"), ""
        
        headings = [s.get('TOCHeading') for s in tox_sec.get('Section', [])]
        html = ""
        # Update: Changed text to be more subtle or hidden as requested
        if headings:
            label = "数据索引" if is_zh else "Index"
            html = f"<div style='font-size:0.85em; color:#777;'><b>{label}:</b> {', '.join(headings[:20])}...</div>"
        else:
            html = "<span style='color:#999'>暂无详细章节信息</span>" if is_zh else "<span style='color:#999'>No section headings found</span>"
        
        text_summary = (f"包含数据章节: {', '.join(headings)}" if is_zh else f"Sections: {', '.join(headings)}") if headings else ""
        return html, text_summary
    except Exception as e:
        return (f"解析毒性数据出错: {str(e)}" if is_zh else f"Failed to parse toxicity section: {str(e)}"), ""


def _is_deepseek_error_text(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return True
    markers = [
        "Request Failed:",
        "API Error:",
        "Error 401",
        "API Key 无效",
        "read timed out",
        "Read timed out",
        "HTTPSConnectionPool",
    ]
    return any(m in t for m in markers)


def chat_with_deepseek(prompt_or_messages, api_key=None, model_name="deepseek-chat", timeout=60, max_retries=2):
    """
    DeepSeek API 对话接口
    """
    if isinstance(prompt_or_messages, list):
        messages = prompt_or_messages
    else:
        messages = [{"role": "user", "content": str(prompt_or_messages)}]

    key = api_key or Config.DEEPSEEK_API_KEY
    if not key:
        return "⚠️ 未检测到 API Key。请设置环境变量 'DEEPSEEK_API_KEY'。"

    model = (model_name or "deepseek-chat").strip()
    retryable_status = {408, 429, 500, 502, 503, 504}
    last_err = None
    retries = max(0, int(max_retries))

    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": 0.7},
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if resp.status_code == 401:
                return "⚠️ API Key 无效或已过期 (Error 401)。请检查 'DEEPSEEK_API_KEY'。"

            last_err = f"API Error: {resp.status_code}"
            if resp.status_code in retryable_status and attempt < retries:
                time.sleep(0.8 * (attempt + 1))
                continue
            return last_err
        except requests.exceptions.ReadTimeout as e:
            last_err = f"Request Failed: {str(e)}"
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
                continue
            return last_err
        except requests.exceptions.RequestException as e:
            last_err = f"Request Failed: {str(e)}"
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
                continue
            return last_err
        except Exception as e:
            return f"Request Failed: {str(e)}"

    return last_err or "Request Failed: Unknown error"

def translate_pubchem_html(html_text: str, target_language: str, api_key=None, model_name="deepseek-chat", timeout=60):
    """
    Translate PubChem HTML content into target_language using DeepSeek.
    Keep all HTML tags/attributes/URLs unchanged and only translate visible text.
    """
    if not html_text or not target_language:
        return html_text
    lang = str(target_language).strip()
    if lang.lower() in {"english", "en"}:
        return html_text
    key = api_key or Config.DEEPSEEK_API_KEY
    if not key:
        return html_text

    prompt = f"""
You are a professional scientific translator.
Translate the following HTML into {lang}.
Rules:
1) Keep ALL HTML tags, attributes, and structure exactly unchanged.
2) Do NOT translate URLs, IDs, CID, CAS, units, or chemical formulas. For hazard codes like Hxxx, keep the code itself unchanged (e.g., "H361"), but translate the descriptive text after the code.
3) Only translate visible text content.
4) Output ONLY the translated HTML. Do NOT add Markdown or code fences.

HTML:
{html_text}
"""
    try:
        translated = chat_with_deepseek(
            [{"role": "user", "content": prompt}],
            api_key=key,
            model_name=model_name,
            timeout=timeout,
            max_retries=2,
        )
        if not translated or _is_deepseek_error_text(translated):
            return html_text

        out = str(translated).strip()
        # Strip optional markdown fences if the model accidentally returns them.
        if out.startswith("```"):
            out = re.sub(r"^```[a-zA-Z]*\s*", "", out)
            out = re.sub(r"\s*```$", "", out).strip()

        # Keep HTML structure safety: if source is HTML but output loses tags, fallback.
        if ("<" in html_text and ">" in html_text) and ("<" not in out or ">" not in out):
            return html_text
        return out or html_text
    except Exception:
        return html_text

def comprehensive_toxicity_analysis(smiles, target_name, ac50_info, lang: str = "中文"):
    """
    PubChem GHS + Toxicity 注释（GHS Classification / Toxicological Information）
    """
    is_zh = _is_zh_lang(lang)
    label_ghs = "GHS 注释" if is_zh else "GHS annotation"
    label_tox = "Toxicity" if is_zh else "Toxicity"
    result = {
        "cid": None,
        "ghs_data": "暂无数据" if is_zh else "No data",
        "ghs_text": "",
        "ghs_codes": [],
        "toxicity_data": "暂无数据" if is_zh else "No data",
        "toxicity_text": "",
    }

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        result["ghs_data"] = (
            "SMILES 无效，无法进行 PubChem GHS 注释。"
            if is_zh
            else "Invalid SMILES; cannot retrieve PubChem GHS annotations."
        )
        return result

    cid = None
    try:
        cid, err, status = _fetch_pubchem_cid(smiles)
        if cid:
            result["cid"] = cid
        else:
            if status == 404:
                result["ghs_data"] = _pubchem_error_message_lang(label_ghs, 404, None, lang=lang)
            else:
                print(f"PubChem CID fetch failed: {err or status}")
                result["ghs_data"] = _pubchem_error_message_lang(label_ghs, status, err, lang=lang)
    except Exception as e:
        print(f"PubChem CID fetch failed: {e}")
        result["ghs_data"] = _pubchem_error_message_lang(label_ghs, None, str(e), lang=lang)

    if cid:
        try:
            ghs_data, ghs_err, ghs_status = _fetch_pubchem_section(cid, "GHS Classification")
            ghs_html, ghs_text = _parse_ghs_data(ghs_data, lang=lang)

            if _is_ghs_missing(ghs_html):
                alt_data, alt_err, alt_status = _fetch_pubchem_section(cid, "Safety and Hazards")
                alt_html, alt_text = _parse_ghs_data(alt_data, lang=lang)
                if not _is_ghs_missing(alt_html):
                    ghs_html, ghs_text = alt_html, alt_text
                    ghs_err, ghs_status = None, None
                else:
                    # preserve error info from whichever call failed
                    if alt_err or alt_status:
                        ghs_err, ghs_status = alt_err, alt_status

            if not _is_ghs_missing(ghs_html):
                result["ghs_data"] = ghs_html
                result["ghs_text"] = ghs_text or ""
                text_all = f"{ghs_text or ''} {ghs_html or ''}"
                result["ghs_codes"] = sorted(set(re.findall(r"H\d{3}", text_all)))
            else:
                if ghs_err or ghs_status:
                    result["ghs_data"] = _pubchem_error_message_lang(label_ghs, ghs_status, ghs_err, lang=lang)
                else:
                    result["ghs_data"] = "PubChem 中未找到 GHS 分类数据" if is_zh else "No GHS Classification data found on PubChem."

            # Toxicity (Toxicological Information preferred)
            tox_data, tox_err, tox_status = _fetch_pubchem_section(cid, "Toxicological Information")
            tox_html = _render_pubchem_section(tox_data, "Toxicological Information") if tox_data else ""
            tox_text = _section_to_text(tox_data, "Toxicological Information") if tox_data else ""

            if not tox_html:
                alt_data, alt_err, alt_status = _fetch_pubchem_section(cid, "Toxicity")
                alt_html = _render_pubchem_section(alt_data, "Toxicity") if alt_data else ""
                alt_text = _section_to_text(alt_data, "Toxicity") if alt_data else ""
                if alt_html:
                    tox_html, tox_text = alt_html, alt_text
                    tox_err, tox_status = None, None
                else:
                    if alt_err or alt_status:
                        tox_err, tox_status = alt_err, alt_status

            if tox_html:
                result["toxicity_data"] = tox_html
                result["toxicity_text"] = tox_text or ""
            else:
                if tox_err or tox_status:
                    result["toxicity_data"] = _pubchem_error_message_lang(label_tox, tox_status, tox_err, lang=lang)
                else:
                    result["toxicity_data"] = "PubChem 中未找到 Toxicity 数据" if is_zh else "No Toxicity section data found on PubChem."

        except Exception as e:
            print(f"PubChem GHS fetch failed: {e}")
            result["ghs_data"] = _pubchem_error_message_lang(label_ghs, None, str(e), lang=lang)
            result["toxicity_data"] = _pubchem_error_message_lang(label_tox, None, str(e), lang=lang)

    return result

def get_best_pose_path(pdbqt_path):
    """
    从 Vina 输出的 PDBQT 中提取第一个模型 (最佳得分)，保存为 PDB。
    """
    path = Path(pdbqt_path)
    out_pdb = path.with_name(path.stem + "_best.pdb")
    
    try:
        obabel = shutil.which("obabel")
        if obabel:
            subprocess.run([obabel, str(path), '-O', str(out_pdb), '-f', '1', '-l', '1'], 
                           check=True, capture_output=True)
            if out_pdb.exists():
                return out_pdb
    except Exception as e:
        print(f"Best pose extraction failed: {e}")
    
    return path

def get_pocket_residues(receptor_path, ligand_path, cutoff=5.0):
    """识别受体上距离配体指定范围内的残基"""
    try:
        rec = Chem.MolFromPDBFile(str(receptor_path), removeHs=True)
        lig = Chem.MolFromPDBFile(str(ligand_path), removeHs=True)
        if not rec or not lig: return []
        
        rec_conf = rec.GetConformer()
        lig_conf = lig.GetConformer()
        rec_pos = rec_conf.GetPositions()
        lig_pos = lig.GetConformer().GetPositions()
        
        dists = np.sqrt(((rec_pos[:, np.newaxis, :] - lig_pos[np.newaxis, :, :]) ** 2).sum(axis=2))
        close_indices = np.where(dists.min(axis=1) < cutoff)[0]
        
        pocket_res = set()
        for idx in close_indices:
            atom = rec.GetAtomWithIdx(int(idx))
            info = atom.GetPDBResidueInfo()
            if info:
                key = (info.GetChainId().strip(), info.GetResidueNumber(), info.GetResidueName().strip())
                pocket_res.add(key)
        
        return sorted(list(pocket_res), key=lambda x: x[1])
    except Exception as e:
        print(f"Pocket residues calc warning: {e}")
        return []

def analyze_interactions(receptor_path, ligand_path):
    """简单分析非共价相互作用 (氢键、疏水)，返回详细信息"""
    ints = []
    try:
        rec = Chem.MolFromPDBFile(str(receptor_path), removeHs=True)
        lig = Chem.MolFromPDBFile(str(ligand_path), removeHs=True)
        if not rec or not lig: return []
        
        rec_conf = rec.GetConformer()
        lig_conf = lig.GetConformer()
        rec_pos = rec_conf.GetPositions()
        lig_pos = lig.GetConformer().GetPositions()
        
        for i in range(lig.GetNumAtoms()):
            l_atom = lig.GetAtomWithIdx(i)
            l_sym = l_atom.GetSymbol()
            l_p = lig_pos[i]
            
            for j in range(rec.GetNumAtoms()):
                r_atom = rec.GetAtomWithIdx(j)
                r_sym = r_atom.GetSymbol()
                r_p = rec_pos[j]
                
                dist = np.linalg.norm(l_p - r_p)
                
                type_ = None
                if dist < 3.5 and l_sym in ['N','O'] and r_sym in ['N','O']:
                    type_ = 'Hydrogen Bond'
                elif dist < 4.5 and l_sym == 'C' and r_sym == 'C':
                    type_ = 'Hydrophobic'
                
                if type_:
                    r_info = r_atom.GetPDBResidueInfo()
                    if r_info:
                        res_name = r_info.GetResidueName().strip()
                        res_num = r_info.GetResidueNumber()
                        chain = r_info.GetChainId().strip()
                        r_atom_name = r_info.GetName().strip()
                        rec_str = f"{chain}:{res_name}{res_num}"
                    else:
                        rec_str = "Unknown"
                        r_atom_name = r_sym
                    
                    ints.append({
                        'type': type_,
                        'start': {'x': float(l_p[0]), 'y': float(l_p[1]), 'z': float(l_p[2])},
                        'end': {'x': float(r_p[0]), 'y': float(r_p[1]), 'z': float(r_p[2])},
                        'dist': float(dist),
                        'ligand_atom': f"{l_sym}{i+1}",
                        'receptor_residue': rec_str,
                        'receptor_atom': r_atom_name
                    })
    except Exception as e: print(f"Interaction analysis error: {e}")
    return ints

def list_contacts(receptor_pdb, ligand_path, cutoff=5.0):
    """wrapper to return DataFrame-compatible list"""
    ints = analyze_interactions(receptor_pdb, ligand_path)
    display_ints = []
    type_map = {'Hydrogen Bond': '氢键', 'Hydrophobic': '疏水作用'}
    for i in ints:
        display_ints.append({
            '相互作用类型': type_map.get(i['type'], i['type']),
            '化合物原子': i['ligand_atom'],
            '受体残基': i['receptor_residue'],
            '受体原子': i['receptor_atom'],
            '距离 (Å)': f"{i['dist']:.2f}"
        })
    return display_ints

def _rewrite_ligand_pdb_line(line: str, resname: str = "LIG", chain: str = "Z", resnum: int = 1):
    if not line.startswith(("ATOM", "HETATM")):
        return None
    try:
        serial = int(line[6:11])
    except Exception:
        serial = 1
    name = line[12:16]
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except Exception:
        return None
    occ = line[54:60].strip()
    temp = line[60:66].strip()
    try:
        occ_v = float(occ) if occ else 1.00
    except Exception:
        occ_v = 1.00
    try:
        temp_v = float(temp) if temp else 0.00
    except Exception:
        temp_v = 0.00
    element = line[76:78].strip()
    if not element:
        element = (name.strip()[:1] or "C").upper()
    return (
        f"HETATM{serial:5d} {name:<4} {resname:>3} {chain:1}"
        f"{resnum:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occ_v:6.2f}{temp_v:6.2f}          {element:>2}"
    )

def _build_ligand_pdb_with_bonds(ligand_path: Path, smiles: str, resname: str = "LIG", chain: str = "Z", resnum: int = 1):
    try:
        mol_pdb = Chem.MolFromPDBFile(str(ligand_path), removeHs=False)
        if mol_pdb is None:
            return None
        mol = mol_pdb
        if smiles:
            try:
                tmpl = Chem.MolFromSmiles(smiles)
                if tmpl is not None:
                    mol = Chem.AssignBondOrdersFromTemplate(tmpl, mol_pdb)
            except Exception:
                mol = mol_pdb

        # enforce residue info
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is None:
                info = Chem.AtomPDBResidueInfo()
            info.SetResidueName(resname)
            info.SetChainId(chain)
            info.SetResidueNumber(resnum)
            info.SetIsHeteroAtom(True)
            atom.SetMonomerInfo(info)

        block = Chem.MolToPDBBlock(mol)
        lines = []
        for line in block.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                rewritten = _rewrite_ligand_pdb_line(line, resname=resname, chain=chain, resnum=resnum)
                lines.append((rewritten or line).rstrip("\n"))
            elif line.startswith("CONECT"):
                lines.append(line.rstrip("\n"))
        return lines if lines else None
    except Exception:
        return None

def _merge_pdb_files(receptor_path: Path, ligand_path: Path, out_path: Path, ligand_smiles: str = None) -> Path:
    """Create a simple complex PDB by concatenating receptor and ligand.
    Ensure ligand is HETATM with residue name/chain for PLIP recognition.
    If SMILES is provided, try to preserve bond connectivity via CONECT."""
    rec_lines = []
    lig_lines = []
    try:
        with open(receptor_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith(("END", "ENDMDL")):
                    continue
                rec_lines.append(line.rstrip("\n"))
        lig_lines = _build_ligand_pdb_with_bonds(ligand_path, ligand_smiles, resname="LIG", chain="Z", resnum=1) or []
        if not lig_lines:
            with open(ligand_path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if line.startswith(("END", "ENDMDL")):
                        continue
                    if line.startswith(("ATOM", "HETATM")):
                        rewritten = _rewrite_ligand_pdb_line(line, resname="LIG", chain="Z", resnum=1)
                        lig_lines.append((rewritten or line).rstrip("\n"))
                        continue
                    if line.startswith(("TER",)):
                        continue
                    # Skip CONECT to avoid inconsistencies after rewrite
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            for line in rec_lines:
                fh.write(line + "\n")
            fh.write("TER\n")
            for line in lig_lines:
                fh.write(line + "\n")
            fh.write("END\n")
        return out_path
    except Exception:
        return out_path

def _find_latest(path: Path, patterns):
    best = None
    best_mtime = -1
    for pat in patterns:
        matches = list(path.rglob(pat))
        for m in matches:
            try:
                mt = m.stat().st_mtime
            except Exception:
                mt = 0
            if mt > best_mtime:
                best_mtime = mt
                best = m
    return best

def _normalize_key(key: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(key).upper())

def _pick_norm(row_norm: dict, keys):
    for k in keys:
        if k in row_norm and row_norm[k]:
            return row_norm[k]
    return ""

def _map_plip_type(name: str, *, lang: str = "中文") -> str:
    is_zh = _is_zh_lang(lang)
    if not name:
        return "相互作用" if is_zh else "Interaction"
    low = str(name).lower()
    if "hydrophobic" in low:
        return "疏水作用" if is_zh else "Hydrophobic"
    if "hydrogen" in low:
        return "氢键" if is_zh else "Hydrogen bond"
    if "salt" in low:
        return "盐桥" if is_zh else "Salt bridge"
    if "pi" in low and "stack" in low:
        return "π-π" if is_zh else "π-π stacking"
    if "pi" in low and "cation" in low:
        return "π-阳离子" if is_zh else "π-cation"
    if "halogen" in low:
        return "卤键" if is_zh else "Halogen bond"
    if "metal" in low:
        return "金属配位" if is_zh else "Metal coordination"
    if "water" in low:
        return "水桥" if is_zh else "Water bridge"
    return str(name)

def _parse_pdb_atoms(pdb_path: Path):
    atoms = []
    idx = 0
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                idx += 1
                try:
                    serial = int(line[6:11])
                except Exception:
                    serial = idx
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip()
                try:
                    resnum = int(line[22:26])
                except Exception:
                    resnum = 0
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except Exception:
                    x = y = z = 0.0
                element = line[76:78].strip()
                if not element and name:
                    element = name[0].upper()
                atoms.append({
                    "idx": idx,
                    "serial": serial,
                    "name": name,
                    "resname": resname,
                    "chain": chain,
                    "resnum": resnum,
                    "x": x, "y": y, "z": z,
                    "element": element or "X",
                })
    except Exception:
        return []
    return atoms

def _first_int(text: str):
    if not text:
        return None
    m = re.search(r"-?\d+", str(text))
    if not m:
        return None
    try:
        return int(m.group())
    except Exception:
        return None

def _first_float(text: str):
    if not text:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(text))
    if not m:
        return None
    try:
        return float(m.group())
    except Exception:
        return None

def _parse_coord_tuple(text: str):
    if not text:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    if len(nums) >= 3:
        try:
            return float(nums[0]), float(nums[1]), float(nums[2])
        except Exception:
            return None
    return None

def _assign_functional_groups(mol, *, lang: str = "中文"):
    """
    Assign a simple functional-group label for each atom using SMARTS matches.
    Output labels are localized (Chinese/English) based on `lang`.
    """
    is_zh = _is_zh_lang(lang)
    labels = [set() for _ in range(mol.GetNumAtoms())]  # store pattern ids
    patterns = [
        # Strongly ionizable / specific
        ("carboxylate", "羧酸盐", "Carboxylate", "[CX3](=O)[OX1-]"),
        ("sulfonate", "磺酸盐", "Sulfonate", "S(=O)(=O)[OX1-]"),
        ("phosphate", "磷酸盐", "Phosphate", "P(=O)(O)(O)"),
        ("quat_ammonium", "季铵盐", "Quaternary ammonium", "[NX4+]"),
        ("guanidinium", "胍(阳离子)", "Guanidinium", "N[CX3](=N)[N;+0,+1]"),
        # Carbonyl family
        ("anhydride", "酸酐", "Anhydride", "[CX3](=O)O[CX3](=O)"),
        ("imide", "酰亚胺", "Imide", "[NX3][CX3](=O)[CX3](=O)"),
        ("ester", "酯", "Ester", "[CX3](=O)O[#6]"),
        ("amide", "酰胺", "Amide", "[NX3][CX3](=O)[#6]"),
        ("urea", "脲/尿素", "Urea", "[NX3][CX3](=O)[NX3]"),
        ("carbamate", "氨基甲酸酯", "Carbamate", "[NX3][CX3](=O)O[#6]"),
        ("carboxylic_acid", "羧酸", "Carboxylic acid", "[CX3](=O)[OX2H]"),
        ("aldehyde", "醛", "Aldehyde", "[CX3H1](=O)[#6]"),
        ("ketone", "酮", "Ketone", "[#6][CX3](=O)[#6]"),
        # Nitrogen groups
        ("nitro", "硝基", "Nitro", "[NX3+](=O)[O-]"),
        ("nitrile", "腈", "Nitrile", "[CX2]#N"),
        ("amidine_imine", "脒/亚胺", "Amidine/Imine", "NC(=N)[#6]"),
        # Oxygen / sulfur
        ("phenol", "酚羟基", "Phenol", "c[OX2H]"),
        ("alcohol", "醇羟基", "Alcohol", "[OX2H][CX4;!$(C=O)]"),
        ("ether", "醚", "Ether", "[OD2]([#6])[#6]"),
        ("thiol", "硫醇", "Thiol", "[SX2H]"),
        ("thioether", "硫醚", "Thioether", "[SX2]([#6])[#6]"),
        ("sulfoxide", "亚砜", "Sulfoxide", "[SX3](=O)[#6]"),
        ("sulfone", "砜", "Sulfone", "[SX4](=O)(=O)[#6]"),
        ("sulfonamide", "磺酰胺", "Sulfonamide", "S(=O)(=O)N"),
        ("sulfonic_acid_ester", "磺酸/磺酸酯", "Sulfonic acid/ester", "S(=O)(=O)[OX2H,OX2]"),
        # Amines (neutral)
        ("amine_primary", "胺(伯)", "Amine (primary)", "[NX3;H2;!$(NC=O)]"),
        ("amine_secondary", "胺(仲)", "Amine (secondary)", "[NX3;H1;!$(NC=O)]"),
        ("amine_tertiary", "胺(叔)", "Amine (tertiary)", "[NX3;H0;!$(NC=O)]"),
        # Heterocycles / aromatics
        ("imidazole", "咪唑", "Imidazole", "n1cc[nH]c1"),
        ("indole", "吲哚", "Indole", "c1cc2ccccc2[nH]1"),
        ("hetero_aromatic_n", "吡啶/杂芳香N", "Heteroaromatic N", "n"),
        ("aromatic_ring", "芳香环", "Aromatic ring", "a"),
        # Halogen / unsaturation
        ("halogen", "卤素", "Halogen", "[F,Cl,Br,I]"),
        ("alkene", "烯烃", "Alkene", "[CX3]=[CX3]"),
        ("alkyne", "炔烃", "Alkyne", "[CX2]#C"),
    ]
    label_by_id = {pid: (zh if is_zh else en) for pid, zh, en, _ in patterns}
    priority = [pid for pid, _, _, _ in patterns]

    for pid, _zh_name, _en_name, smarts in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if not patt:
            continue
        for match in mol.GetSubstructMatches(patt):
            for idx in match:
                labels[idx].add(pid)

    out = []
    for i, atom in enumerate(mol.GetAtoms()):
        if labels[i]:
            chosen = None
            for pid in priority:
                if pid in labels[i]:
                    chosen = pid
                    break
            if chosen:
                out.append(label_by_id[chosen])
            else:
                out.append("/".join(label_by_id[x] for x in sorted(labels[i])))
        else:
            out.append(atom.GetSymbol())
    return out

def _ligand_fg_by_complex_index(ligand_pdb: Path, complex_pdb: Path, *, lang: str = "中文"):
    mapping = {}
    try:
        mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=False)
        if mol is None:
            return mapping
        conf = mol.GetConformer()
        fg_labels = _assign_functional_groups(mol, lang=lang)
        rd_atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            rd_atoms.append((i, (pos.x, pos.y, pos.z), fg_labels[i]))
        complex_atoms = _parse_pdb_atoms(complex_pdb)
        lig_atoms = [a for a in complex_atoms if a["resname"] in {"UNL", "LIG", "UNK", "MOL"} or a["chain"] == "Z"]
        # map by nearest coordinate
        for a in lig_atoms:
            ax, ay, az = a["x"], a["y"], a["z"]
            best = None
            best_d = 1e9
            for i, (x, y, z), label in rd_atoms:
                d = (ax - x) ** 2 + (ay - y) ** 2 + (az - z) ** 2
                if d < best_d:
                    best_d = d
                    best = label
            if best is not None and best_d <= 1.0:  # ~1.0A^2 tolerance
                mapping[a["idx"]] = best
    except Exception:
        return mapping
    return mapping

def _residue_functional_group(resname: str, *, lang: str = "中文"):
    if not resname:
        return ""
    is_zh = _is_zh_lang(lang)
    res = resname.upper()
    mapping = {
        "ASP": ("羧酸盐(酸性)", "Carboxylate (acidic)"),
        "GLU": ("羧酸盐(酸性)", "Carboxylate (acidic)"),
        "ASN": ("酰胺(极性)", "Amide (polar)"),
        "GLN": ("酰胺(极性)", "Amide (polar)"),
        "LYS": ("胺(伯,阳离子)", "Amine (cationic)"),
        "ARG": ("胍(阳离子)", "Guanidinium (cationic)"),
        "HIS": ("咪唑(可质子化)", "Imidazole (protonatable)"),
        "SER": ("羟基(极性)", "Hydroxyl (polar)"),
        "THR": ("羟基(极性)", "Hydroxyl (polar)"),
        "TYR": ("酚羟基/芳香环", "Phenol / aromatic ring"),
        "CYS": ("硫醇(极性)", "Thiol (polar)"),
        "MET": ("硫醚(疏水)", "Thioether (hydrophobic)"),
        "PHE": ("芳香环(苯基)", "Aromatic ring (phenyl)"),
        "TRP": ("芳香环(吲哚)", "Aromatic ring (indole)"),
        "ILE": ("脂肪族(疏水)", "Aliphatic (hydrophobic)"),
        "LEU": ("脂肪族(疏水)", "Aliphatic (hydrophobic)"),
        "VAL": ("脂肪族(疏水)", "Aliphatic (hydrophobic)"),
        "ALA": ("脂肪族(疏水)", "Aliphatic (hydrophobic)"),
        "PRO": ("脂肪族(疏水)", "Aliphatic (hydrophobic)"),
        "GLY": ("脂肪族(疏水)", "Aliphatic (hydrophobic)"),
    }
    val = mapping.get(res)
    if not val:
        return res
    return val[0] if is_zh else val[1]

def _parse_plip_report_txt(report_path: Path):
    interactions = []
    try:
        lines = report_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return interactions

    current_type = None
    headers = None
    for line in lines:
        text = (line or "").strip()
        if not text:
            continue
        if "Interaction" in text:
            current_type = text
            headers = None
            continue
        if text.startswith("|") and text.endswith("|"):
            cells = [c.strip() for c in text.strip("|").split("|")]
            if all(re.fullmatch(r"[-= ]*", c or "") for c in cells):
                continue
            if headers is None:
                headers = cells
                continue
            if len(cells) != len(headers):
                continue
            row = dict(zip(headers, cells))
            interactions.append((current_type, row))
    return interactions

def _normalize_pose_dir(label: str):
    if not label:
        return None
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(label).strip()).strip("_")
    if not s:
        return None
    low = s.lower()
    if low.startswith("pose_"):
        return low
    if low.startswith("pose"):
        rest = re.sub(r"^pose_?", "", low)
        return f"pose_{rest}" if rest else "pose"
    return low


def run_plip_2d_analysis(
    receptor_path: str,
    ligand_path: str,
    ligand_smiles: str = None,
    complex_path: str = None,
    pose_label: str = None,
    lang: str = "中文",
):
    is_zh = _is_zh_lang(lang)
    ok, msg = check_plip_tool(lang=lang)
    if not ok:
        return {"error": msg}
    try:
        receptor_path = Path(receptor_path) if receptor_path else None
        ligand_path = Path(ligand_path) if ligand_path else None
        base_out_dir = Config.OUTPUT_DIR / "PLIP"
        pose_dir = None
        if complex_path:
            try:
                parent = Path(complex_path).resolve().parent
                if parent.name.lower().startswith("pose_"):
                    pose_dir = parent.name.lower()
            except Exception:
                pose_dir = None
        if pose_dir is None:
            pose_dir = _normalize_pose_dir(pose_label)
        out_dir = base_out_dir / pose_dir if pose_dir else base_out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if not ligand_smiles:
            try:
                smi_path = Config.OUTPUT_DIR / "docked_ligand.smiles"
                if smi_path.exists():
                    ligand_smiles = smi_path.read_text(encoding="utf-8").strip()
            except Exception:
                ligand_smiles = None

        complex_pdb = None
        if complex_path:
            try:
                cand = Path(complex_path)
                if cand.exists():
                    complex_pdb = cand
                    # Keep a local copy under PLIP output for easier inspection.
                    try:
                        dst = out_dir / "complex.pdb"
                        if cand.resolve() != dst.resolve():
                            shutil.copyfile(cand, dst)
                        complex_pdb = dst
                    except Exception:
                        # If copy fails, still proceed with the original path.
                        complex_pdb = cand
            except Exception:
                complex_pdb = None

        if complex_pdb is None:
            if not receptor_path or not ligand_path:
                return {
                    "error": (
                        "缺少受体/配体文件，无法生成复合物用于 PLIP 分析。"
                        if is_zh
                        else "Missing receptor/ligand files; cannot build a complex for PLIP analysis."
                    )
                }
            complex_pdb = _merge_pdb_files(
                receptor_path,
                ligand_path,
                out_dir / "complex.pdb",
                ligand_smiles=ligand_smiles,
            )
            if not complex_pdb.exists():
                return {
                    "error": (
                        "无法生成复合物 PDB 文件，PLIP 分析终止。"
                        if is_zh
                        else "Failed to generate a complex PDB file; PLIP analysis aborted."
                    )
                }

        use_plipfix = False
        cmd = [
            Config.PLIP_CMD,
            "-f", str(complex_pdb),
            "-o", str(out_dir),
            "-x",
            "-t",
            "-p",
        ]
        if not use_plipfix:
            cmd.append("--nofix")
        _run_cmd(cmd, cwd=out_dir, label="plip", log_file="plip.log")

        img_path = _find_latest(out_dir, ["interaction_image.png", "*interaction*.png", "*.png"])
        report_txt = _find_latest(out_dir, ["*report*.txt", "report.txt"])
        report_xml = _find_latest(out_dir, ["*report*.xml", "report.xml"])
        fixed_complex = complex_pdb if not use_plipfix else _find_latest(
            out_dir,
            ["plipfixed.*.pdb", "*fixed*.pdb", "complex_protonated.pdb", "complex.pdb"],
        )

        interactions = []
        draw_interactions = []
        lig_fg_map = {}
        lig_atoms = []
        if fixed_complex and ligand_path:
            lig_fg_map = _ligand_fg_by_complex_index(Path(ligand_path), Path(fixed_complex), lang=lang)
            try:
                complex_atoms = _parse_pdb_atoms(Path(fixed_complex))
                for a in complex_atoms:
                    if a["resname"] in {"UNL", "LIG", "UNK", "MOL"} or a["chain"] == "Z":
                        lig_atoms.append({
                            "x": a["x"], "y": a["y"], "z": a["z"],
                            "label": lig_fg_map.get(a["idx"], ""),
                        })
            except Exception:
                lig_atoms = []
        if report_txt:
            parsed = _parse_plip_report_txt(report_txt)
            for typ, row in parsed:
                row_norm = {_normalize_key(k): v for k, v in row.items()}
                lig_atom_raw = _pick_norm(row_norm, [
                    "LIGATOM", "LIGATOMNAME", "LIGANDATOM", "ATOMLIG",
                    "LIGCARBONIDX", "LIG_IDX_LIST", "LIGATOMIDX"
                ])
                prot_atom = _pick_norm(row_norm, [
                    "PROTATOM", "PROTATOMNAME", "RECEPTORATOM", "ATOMPROT", "ATOM",
                    "PROTCARBONIDX", "PROT_IDX_LIST"
                ])
                lig_coord = _pick_norm(row_norm, ["LIGCOO", "LIGCOORD", "LIGCOORDS"])
                resname = _pick_norm(row_norm, ["RESTYPE", "RESNAME", "RES"])
                resnum = _pick_norm(row_norm, ["RESNR", "RESNUM", "RESID"])
                chain = _pick_norm(row_norm, ["RESCHAIN", "CHAIN", "CHAINID"])
                dist = _pick_norm(row_norm, ["DIST", "DISTANCE", "DISTA", "CENTDIST"])
                residue = ""
                if resname or resnum:
                    residue = f"{resname}{resnum}".strip()
                    if chain:
                        residue = f"{chain}:{residue}"
                # map functional groups
                lig_fg = ""
                lig_idx = _first_int(lig_atom_raw)
                if lig_idx is not None:
                    lig_fg = lig_fg_map.get(lig_idx, "")
                if (not lig_fg) and lig_coord and lig_atoms:
                    try:
                        lx, ly, lz = lig_coord
                        best_label = ""
                        best_d = 1e9
                        for a in lig_atoms:
                            dx = lx - a["x"]
                            dy = ly - a["y"]
                            dz = lz - a["z"]
                            d = dx * dx + dy * dy + dz * dz
                            if d < best_d and a["label"]:
                                best_d = d
                                best_label = a["label"]
                        if best_label and best_d <= 1.0:
                            lig_fg = best_label
                    except Exception:
                        pass
                if not lig_fg:
                    lig_fg = "官能团未知" if is_zh else "Unknown functional group"

                prot_fg = _residue_functional_group(resname, lang=lang)
                if not prot_fg:
                    prot_fg = "官能团未知" if is_zh else "Unknown functional group"
                dist_val = _first_float(dist)
                lig_xyz = _parse_coord_tuple(lig_coord)
                if lig_atom_raw or residue:
                    draw_interactions.append({
                        "type": typ,
                        "ligand_atom": str(lig_atom_raw) if lig_atom_raw is not None else "",
                        "ligand_xyz": lig_xyz,
                        "receptor_residue": residue,
                        "dist": dist_val,
                    })

                interactions.append({
                    ("相互作用类型" if is_zh else "Interaction Type"): _map_plip_type(typ, lang=lang),
                    ("化合物原子" if is_zh else "Ligand Atom"): str(lig_atom_raw) if lig_atom_raw is not None else "",
                    ("化合物官能团" if is_zh else "Ligand Functional Group"): lig_fg,
                    ("受体残基" if is_zh else "Receptor Residue"): residue,
                    ("氨基酸官能团" if is_zh else "Residue Functional Group"): prot_fg,
                    ("受体原子" if is_zh else "Receptor Atom"): str(prot_atom) if prot_atom is not None else "",
                    ("距离 (Å)" if is_zh else "Distance (Å)"): dist,
                })

        if not interactions and report_xml:
            # XML parsing can be added later if needed.
            pass

        return {
            "image_path": str(img_path) if img_path else None,
            "interactions": interactions,
            "draw_interactions": draw_interactions,
            "report_txt": str(report_txt) if report_txt else None,
            "report_xml": str(report_xml) if report_xml else None,
        }
    except Exception as e:
        return {"error": (f"PLIP 分析失败: {e}" if is_zh else f"PLIP analysis failed: {e}")}

# ==========================================
# 新增：Maestro 风格 2D 相互作用绘图引擎 (RDKit)
# ==========================================
def generate_maestro_2d_map(ligand_path, interactions, width=600, height=450):
    """
    生成 Maestro 风格的 2D 相互作用图 (SVG格式)
    保留化学结构，并绘制相互作用连线
    """
    try:
        # 1. 读取配体并处理 2D 坐标
        mol = Chem.MolFromPDBFile(str(ligand_path), removeHs=True)
        if not mol: return None
        
        # 尝试保留 3D 构象的投影，或者重新生成清晰的 2D 构象
        try:
            AllChem.Compute2DCoords(mol)
        except:
            pass

        # 2. 初始化绘图对象
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()

        # Match reference style: ligand = orange, residues = blue
        ligand_color = (0.82, 0.55, 0.18)
        residue_color = (0.18, 0.41, 0.67)
        try:
            opts.atomColourPalette = {i: ligand_color for i in range(1, 119)}
        except Exception:
            pass
        
        # ⚠️ 重要修正：防止透明背景导致在白色网页中看不见
        # 不使用 clearBackground=False，而是手动设置背景色为白色
        opts.setBackgroundColour((1, 1, 1)) 
        opts.padding = 0.1  # 留出空间给外部标签
        
        # 3. 准备高亮数据
        highlight_atoms = []
        highlight_colors = {}

        def _interaction_color(itype: str):
            low = (itype or "").lower()
            if "hydrophobic" in low:
                return (0.55, 0.55, 0.55)  # grey
            if "hydrogen" in low:
                return (0.20, 0.35, 0.80)  # blue
            if "salt" in low:
                return (0.90, 0.75, 0.20)  # yellow
            if "water" in low:
                return (0.45, 0.70, 0.95)  # light blue
            return (0.60, 0.60, 0.60)

        def _interaction_line_width(itype: str):
            low = (itype or "").lower()
            if "hydrogen" in low or "salt" in low:
                return 2
            return 1

        def _draw_dashed(drawer, p1, p2, dash=6, gap=4):
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length = math.hypot(dx, dy)
            if length <= 0:
                return
            ux = dx / length
            uy = dy / length
            dist = 0.0
            while dist < length:
                start = dist
                end = min(dist + dash, length)
                sp = Geometry.Point2D(p1.x + ux * start, p1.y + uy * start)
                ep = Geometry.Point2D(p1.x + ux * end, p1.y + uy * end)
                drawer.DrawLine(sp, ep, True)
                dist += dash + gap

        # Build ligand atom name -> index map (from PDB residue info)
        name_to_idx = {}
        name_to_idx_upper = {}
        for atom in mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if not info:
                continue
            name = (info.GetName() or "").strip()
            if name:
                name_to_idx[name] = atom.GetIdx()
                name_to_idx_upper[name.upper()] = atom.GetIdx()

        # Build coordinates for nearest-neighbor fallback
        coord_idx = []
        try:
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coord_idx.append((atom.GetIdx(), (pos.x, pos.y, pos.z)))
        except Exception:
            coord_idx = []
        
        # 提取相互作用的原子索引
        atom_interactions = {} # {atom_idx: [list of interactions]}
        for inter in interactions:
            # 找到配体原子的索引 (我们之前保存的是 "C15" 这种名字)
            # 需要反查 index
            atom_name = inter.get('ligand_atom', "")
            atom_idx = -1

            atom_names = []
            if atom_name:
                for part in re.split(r"[;,\\s]+", str(atom_name)):
                    p = part.strip()
                    if p:
                        atom_names.append(p)

            # 1) Match PDB atom names
            for an in atom_names:
                if an in name_to_idx:
                    atom_idx = name_to_idx[an]
                    break
                au = an.upper()
                if au in name_to_idx_upper:
                    atom_idx = name_to_idx_upper[au]
                    break

            # 2) Match symbol+index
            if atom_idx == -1 and atom_names:
                for an in atom_names:
                    for atom in mol.GetAtoms():
                        if atom.GetSymbol() + str(atom.GetIdx() + 1) == an:
                            atom_idx = atom.GetIdx()
                            break
                    if atom_idx != -1:
                        break

            # 3) Numeric fallback (e.g. C15 -> 15 -> idx 14)
            if atom_idx == -1 and atom_names:
                for an in atom_names:
                    idx_match = re.search(r'\\d+', an)
                    if idx_match:
                        try:
                            atom_idx = int(idx_match.group()) - 1
                            break
                        except Exception:
                            continue

            # 4) Coordinate fallback (match nearest atom to LIGCOO)
            if atom_idx == -1 and coord_idx:
                lig_xyz = inter.get("ligand_xyz")
                if lig_xyz:
                    try:
                        lx, ly, lz = lig_xyz
                        best_idx = -1
                        best_d = 1e9
                        for idx, (x, y, z) in coord_idx:
                            d = (x - lx) ** 2 + (y - ly) ** 2 + (z - lz) ** 2
                            if d < best_d:
                                best_d = d
                                best_idx = idx
                        if best_idx >= 0:
                            atom_idx = best_idx
                    except Exception:
                        pass
            
            if atom_idx >= 0 and atom_idx < mol.GetNumAtoms():
                if atom_idx not in atom_interactions:
                    atom_interactions[atom_idx] = []
                atom_interactions[atom_idx].append(inter)
                highlight_atoms.append(atom_idx)
                
                # 设置高亮颜色: 氢键优先(绿色)，其他相互作用(橙色)
                itype = str(inter.get("type", ""))
                color = _interaction_color(itype)
                if atom_idx not in highlight_colors:
                    highlight_colors[atom_idx] = (color[0], color[1], color[2], 0.5)

        # If nothing could be mapped, let caller fall back to PLIP image.
        if not atom_interactions:
            return None

        # 4. 绘制分子结构 (但先不结束)
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
        
        # 5. 在 Canvas 上额外绘制相互作用连线和标签
        # 获取原子的绘制坐标
        # 注意：需要先 DrawMolecule 才能有坐标映射，但 SVG 没 finish 就可以继续画
        
        for atom_idx, inter_list in atom_interactions.items():
            # 获取原子在画布上的坐标 (Pixel Coordinates)
            try:
                p_atom = drawer.GetDrawCoords(atom_idx)
            except:
                continue
            
            # 简单的偏移量
            offset_x = 0
            offset_y = 0
            
            # 遍历该原子的所有相互作用 (可能有多个残基作用于同一个原子)
            for i, inter in enumerate(inter_list):
                # 稍微错开多个标签
                spread_angle = (i - len(inter_list)/2) * 0.5
                
                label_dist = 60 # 连线长度 (Pixels)
                
                # 简易方向计算：如果原子在画布左侧，标签往左画；在右侧往右画
                canvas_center_x = width / 2
                canvas_center_y = height / 2
                
                dir_x = 1 if p_atom.x > canvas_center_x else -1
                dir_y = 1 if p_atom.y > canvas_center_y else -1
                
                label_x = p_atom.x + (dir_x * label_dist) + (spread_angle * 20)
                label_y = p_atom.y + (dir_y * label_dist * 0.5)
                
                p_label = Geometry.Point2D(label_x, label_y)
                
                # 设置线条样式 (虚线 + 颜色)
                itype = str(inter.get("type", ""))
                color = _interaction_color(itype)
                line_width = _interaction_line_width(itype)
                drawer.SetColour(color)
                drawer.SetLineWidth(line_width)
                _draw_dashed(drawer, p_atom, p_label, dash=7 if line_width > 1 else 5, gap=4)
                
                # 绘制文字 (氨基酸名称)
                res_label = inter.get('receptor_residue', "") # e.g. A:ARG394
                # 简化标签，去掉链名，只留 ARG394
                if ":" in res_label:
                    res_label = res_label.split(":")[1]
                
                # ⚠️ 重要修正：增大字体大小 (Pixel Mode)
                drawer.SetFontSize(14) # 字体大小 (Pixel)
                drawer.SetColour(residue_color) # 残基标签颜色
                
                # 调整文字位置
                text_pos = Geometry.Point2D(label_x + (5 if dir_x>0 else -25), label_y)
                drawer.DrawString(res_label, text_pos, True)
                
                # 标注作用距离
                dist_val = inter.get('dist', None)
                dist_label = f"{float(dist_val):.1f}A" if dist_val is not None else ""
                drawer.SetFontSize(12) # 字体大小 (Pixel)
                drawer.SetColour((0.5, 0.5, 0.5))
                mid_x = (p_atom.x + label_x) / 2
                mid_y = (p_atom.y + label_y) / 2
                if dist_label:
                    drawer.DrawString(dist_label, Geometry.Point2D(mid_x, mid_y), True)

        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    except Exception as e:
        print(f"Maestro map error: {e}")
        return None
