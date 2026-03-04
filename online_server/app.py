import os
import time
import html
import json
import base64
import signal
import threading
from functools import lru_cache
from contextlib import contextmanager
import py3Dmol
from stmol import showmol
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Draw

# Molstar docking viewer (compat-patched build for WebGL initialization robustness)
MOLSTAR_VIEWER_IMPL = "native"
MOLSTAR_COMPAT_STATUS = {}
MOLSTAR_COMPAT_ERROR = ""
try:
    import molstar_docking_compat as _molstar_compat
    from molstar_docking_compat import st_molstar_docking
    MOLSTAR_VIEWER_IMPL = "compat"
    try:
        MOLSTAR_COMPAT_STATUS = _molstar_compat.get_compat_status()
    except Exception:
        MOLSTAR_COMPAT_STATUS = {}
except Exception as _molstar_err:
    MOLSTAR_COMPAT_ERROR = str(_molstar_err)
    from streamlit_molstar.docking import st_molstar_docking

# Backend module (local file: online_server/backend.py)
try:
    import backend
    from backend import Config
except ImportError as e:
    st.error(f"无法导入后端模块 (Import Error): {e}")
    st.stop()


# Must be called before other Streamlit commands.
st.set_page_config(
    page_title="QS-ER Screen",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Auto-shutdown (when all browser sessions are closed) ----
AUTO_SHUTDOWN_ON_IDLE = os.getenv("AUTO_SHUTDOWN_ON_IDLE", "1").lower() in {"1", "true", "yes", "y"}
AUTO_SHUTDOWN_IDLE_SECONDS = int(os.getenv("AUTO_SHUTDOWN_IDLE_SECONDS", "0"))
AUTO_SHUTDOWN_CHECK_SECONDS = int(os.getenv("AUTO_SHUTDOWN_CHECK_SECONDS", "1"))

def _get_active_session_count():
    try:
        from streamlit.runtime import get_instance as _get_runtime
        rt = _get_runtime()
        if rt is None:
            return None
        if hasattr(rt, "get_active_session_infos"):
            infos = rt.get_active_session_infos()
            return len(infos) if infos is not None else 0
        mgr = getattr(rt, "_session_mgr", None)
        if mgr and hasattr(mgr, "list_sessions"):
            return len(mgr.list_sessions())
    except Exception:
        return None
    return None

def _shutdown_process():
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)
    finally:
        os._exit(0)

def _auto_shutdown_loop():
    last_active = time.time()
    had_active = False
    while True:
        count = _get_active_session_count()
        if count is None:
            # If we cannot detect sessions reliably, disable auto-shutdown.
            return
        if count > 0:
            last_active = time.time()
            had_active = True
        else:
            if had_active and (time.time() - last_active) >= AUTO_SHUTDOWN_IDLE_SECONDS:
                _shutdown_process()
                return
        time.sleep(AUTO_SHUTDOWN_CHECK_SECONDS)

@lru_cache(maxsize=1)
def _start_auto_shutdown():
    if not AUTO_SHUTDOWN_ON_IDLE:
        return
    t = threading.Thread(target=_auto_shutdown_loop, daemon=True)
    t.start()

_start_auto_shutdown()


api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_AVATAR_B64 = ""
_avatar_path = Path(__file__).resolve().parent.parent / "Deepseek.jpg"
try:
    if _avatar_path.exists():
        DEEPSEEK_AVATAR_B64 = base64.b64encode(_avatar_path.read_bytes()).decode("utf-8")
except Exception:
    DEEPSEEK_AVATAR_B64 = ""


TRANSLATIONS = {
    "中文": {
        "page_title": "QS-ER 毒性筛查平台",
        "header_title": "QS-ER 雌激素受体干扰物筛查平台",
        "header_desc": "集成靶点预测、毒性结构警示、PubChem GHS/Toxicity 注释与分子对接分析。",
        "sidebar_global_settings": "全局设置",
        "sidebar_language": "界面语言 (Language)",
        "sidebar_ai_model": "AI 模型选择",
        "tab_pred": "靶点预测",
        "tab_dock": "分子对接",
        "module_intro_label": "功能介绍",
        "pred_module_intro": "输入 SMILES 后自动完成靶点预测、毒性结构警示与 PubChem GHS/Toxicity 注释，并给出结果总览。",
        "dock_module_intro": "选择受体并运行 Vina 对接，自动生成 9 个构象，支持 3D 结合模式与相互作用分析。",
        "pred_card_input": "分子输入",
        "pred_input_placeholder": "每行一个 SMILES，例如: CC(=O)OC1=CC=CC=C1C(=O)O",
        "pred_btn_example": "加载示例 SMILES",
        "pred_preview_title": "2D 结构预览",
        "pred_warn_invalid": "无效的 SMILES，无法生成预览。",
        "pred_warn_failed": "预览生成失败。",
        "pred_info_preview": "输入 SMILES 后可预览 2D 结构。",
        "pred_pubchem_title": "PubChem 关键信息",
        "pred_pubchem_loading": "正在查询 PubChem 关键信息...",
        "pred_pubchem_empty": "未检索到 PubChem 记录。",
        "pred_btn_start": "开始靶点预测 & 毒性分析",
        "pred_err_no_smiles": "请先输入有效的 SMILES。",
        "pred_spinner": "正在进行预测与分析，请稍候...",
        "pred_success": "预测完成。",
        "pred_card_result": "预测结果概览",
        "pred_warn_parse": "解析关键指标时发生错误:",
        "pred_err_format": "backend.target_prediction 返回格式异常",
        "pred_info_smiles_changed": "SMILES 已变更，请重新运行预测以更新结果。",
        "pred_select_smiles": "选择展示分子",
        "pred_decoy_info": "预测为 Decoy，表示该分子可能具有非雌激素受体靶向性。",
        "pred_no_results": "当前选择的分子没有可用预测结果。",
        "target_intro_title": "靶点介绍",
        "target_intro_fail": "靶点介绍生成失败:",
        "tox_label": "毒性风险评估",
        "tox_pass": "通过 (Pass)",
        "tox_risk": "存在风险 (Risk Alert)",
        "tox_risk_basis_title": "风险依据",
        "pred_table_title": "靶点预测详情",
        "tox_table_title": "毒性预测详情",
        "tox_info_safe": "未检测到显著的结构警示。",
        "tox_info_no_alert_risk": "未命中传统结构警示，但检测到内分泌干扰相关风险信号。",
        "tox_err_unknown": "运行预测时发生未知错误:",
        "tox_pubchem_card": "PubChem GHS/Toxicity 注释",
        "tox_pubchem_btn": "生成 PubChem GHS/Toxicity 注释",
        "tox_pubchem_spinner": "正在查询 PubChem GHS/Toxicity 注释，请稍候...",
        "tox_pubchem_hint": "点击按钮生成 PubChem GHS/Toxicity 注释。",
        "tox_pubchem_err": "生成报告失败:",
        "tox_pubchem_ghs": "GHS 注释",
        "tox_pubchem_tox": "Toxicity",
        "pubchem_translate_spinner": "正在翻译 PubChem 内容...",
        "pubchem_translate_fallback": "未检测到 API Key，PubChem 翻译不可用，已显示原文。",
        "dock_card_params": "对接参数设置",
        "dock_err_no_receptor": "请在 {dir} 目录下放置受体文件 (.pdb 或 .pdbqt)",
        "dock_label_receptor": "选择受体 (Receptor)",
        "dock_title_space": "搜索空间 (Search Space)",
        "dock_title_box": "盒子尺寸 (Box Size Å)",
        "dock_title_calc": "计算参数",
        "dock_label_exhaustiveness": "计算精度 (Exhaustiveness)",
        "dock_help_exhaustiveness": "值越大计算越精确，但耗时更长。",
        "dock_card_run": "运行控制",
        "dock_text_ready": "准备就绪后，点击按钮开始对接。",
        "dock_btn_start": "开始 AutoDock Vina 对接",
        "dock_err_select_receptor": "无法运行：请先选择受体文件。",
        "dock_info_prep": "正在准备受体和配体文件...",
        "dock_err_prep": "文件准备失败，请检查后台日志或输入分子。",
        "dock_info_vina_start": "Vina 引擎启动中，计算可能需要几分钟，请耐心等待...",
        "dock_spinner_vina": "Vina 计算进行中...",
        "dock_success": "对接完成。",
        "dock_label_best_affinity": "最佳结合能 (Best Affinity)",
        "dock_card_3d": "3D 结合模式可视化 (Molstar)",
        "dock_err_3d_load": "可视化加载失败:",
        "dock_warn_no_pdb": "未找到对接结果文件，无法进行可视化。",
        "dock_card_scores": "对接得分列表",
        "dock_info_no_scores": "无得分数据。",
        "dock_title_ai_report": "AI 分析报告",
        "dock_info_no_report": "暂无分析报告。",
        "dock_err_unknown": "对接流程发生异常:",
        "dock_err_box_calc": "无法根据完整蛋白自动计算对接盒子: {err}",
        "dock_err_deps_check": "Docking 依赖检测失败: {err}",
        "dock_label_pose_select": "选择对接构象 (用于 3D 可视化 & 相互作用分析)",
        "dock_molstar_hint": "提示：先点击 3D 视图右上角“滑杆/设置”按钮展开控制面板，再在结构树中找到 'Non-covalent Interactions' 并点击左侧眼睛图标；若未出现该项，请使用下方 PLIP 相互作用分析。",
        "dock_plip_section_title": "相互作用深度分析 (2D网络图 & 统计表)",
        "dock_plip_unavailable": "PLIP 未安装或不可用：{msg}。请先安装 plip 后再使用 2D 相互作用分析。",
        "dock_plip_card_title": "蛋白-化合物相互作用分析 (PLIP)",
        "dock_plip_no_image": "未生成 2D 相互作用图（请检查 PLIP 输出）。",
        "dock_plip_table_title": "相互作用数据明细",
        "dock_plip_no_interactions": "未解析到相互作用数据（请检查 PLIP 报告文件）。",
        "ai_title": "DeepSeek 科学助手",
        "ai_caption": "可询问药物设计、靶点背景或当前分析结果。",
        "ai_input_placeholder": "请输入您的问题...",
        "ai_err_no_key": "未检测到 API Key。请设置环境变量 'DEEPSEEK_API_KEY'。",
        "ai_spinner": "思考中...",
        "ai_err_no_response": "无法获取回复，请检查网络连接或 API Key。",
        "ai_err_fail": "AI 服务调用失败:",
        "ai_float_btn": "唤醒 AI 助手",
        "ai_send_btn": "发送",
        "ai_clear_btn": "清空",
        "export_title": "结果导出",
        "export_desc": "将当前页面输出保存为可离线打开的 HTML 报告（非 PDF）。",
        "export_btn_html": "下载完整网页报告 (.html)",
        "export_no_content": "暂无可导出的结果。请先运行预测或对接。",
        "molstar_diag_title": "Molstar / WebGL 诊断",
        "molstar_diag_impl": "当前组件",
        "molstar_diag_impl_compat": "兼容补丁组件（已启用）",
        "molstar_diag_impl_native": "原始组件（未启用兼容补丁）",
        "molstar_diag_native_warn": "当前未使用兼容补丁，可能仍触发 WebGL 初始化失败。",
        "molstar_diag_compat_error": "兼容补丁加载失败",
        "molstar_diag_status_title": "补丁状态",
        "molstar_diag_no_status": "未读取到补丁状态。",
        "molstar_diag_probe_title": "浏览器 WebGL 探针",
        "molstar_diag_probe_caption": "以下结果由当前浏览器实时检测，用于判断 WebGL1 / WebGL2 是否可用。",
    },
    "English": {
        "page_title": "QS-ER Screen",
        "header_title": "QS-ER Endocrine Disruptor Screening Platform",
        "header_desc": "Integrated target prediction, toxicity alerts, PubChem GHS/Toxicity annotations, and molecular docking analysis.",
        "sidebar_global_settings": "Global Settings",
        "sidebar_language": "Language",
        "sidebar_ai_model": "AI Model",
        "tab_pred": "Target Prediction",
        "tab_dock": "Molecular Docking",
        "module_intro_label": "Module Overview",
        "pred_module_intro": "Provide SMILES to run target prediction, toxicity alerts, and PubChem GHS/Toxicity annotations with a consolidated overview.",
        "dock_module_intro": "Choose a receptor and run Vina docking to generate 9 poses with 3D visualization and interaction analysis.",
        "pred_card_input": "Molecule Input",
        "pred_input_placeholder": "One SMILES per line, e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
        "pred_btn_example": "Load Example SMILES",
        "pred_preview_title": "2D Structure Preview",
        "pred_warn_invalid": "Invalid SMILES.",
        "pred_warn_failed": "Preview failed.",
        "pred_info_preview": "Enter SMILES to preview 2D structure.",
        "pred_pubchem_title": "PubChem Summary",
        "pred_pubchem_loading": "Fetching PubChem summary...",
        "pred_pubchem_empty": "No PubChem record found.",
        "pred_btn_start": "Start Prediction & Toxicity Analysis",
        "pred_err_no_smiles": "Please enter valid SMILES.",
        "pred_spinner": "Running prediction...",
        "pred_success": "Done.",
        "pred_card_result": "Prediction Overview",
        "pred_warn_parse": "Failed to parse metrics:",
        "pred_err_format": "Unexpected backend.target_prediction return format.",
        "pred_info_smiles_changed": "SMILES changed. Re-run prediction to refresh results.",
        "pred_select_smiles": "Select molecule",
        "pred_decoy_info": "Predicted as Decoy, suggesting the compound is unlikely to target ER.",
        "pred_no_results": "No prediction results available for the selected molecule.",
        "target_intro_title": "Target Overview",
        "target_intro_fail": "Failed to generate target overview:",
        "tox_label": "Toxicity Risk",
        "tox_pass": "Pass",
        "tox_risk": "Risk Alert",
        "tox_risk_basis_title": "Risk Basis",
        "pred_table_title": "Target Prediction Details",
        "tox_table_title": "Toxicity Alert Details",
        "tox_info_safe": "No structural alerts detected.",
        "tox_info_no_alert_risk": "No classic structural alert was hit, but endocrine-disruption risk signals were detected.",
        "tox_err_unknown": "Unknown error during prediction:",
        "tox_pubchem_card": "PubChem GHS/Toxicity Annotation",
        "tox_pubchem_btn": "Generate PubChem GHS/Toxicity Annotation",
        "tox_pubchem_spinner": "Fetching PubChem GHS/Toxicity annotation...",
        "tox_pubchem_hint": "Click to generate PubChem GHS/Toxicity annotation.",
        "tox_pubchem_err": "Report failed:",
        "tox_pubchem_ghs": "GHS Annotation",
        "tox_pubchem_tox": "Toxicity",
        "pubchem_translate_spinner": "Translating PubChem content...",
        "pubchem_translate_fallback": "API key not found. PubChem translation is unavailable; showing original content.",
        "dock_card_params": "Docking Parameters",
        "dock_err_no_receptor": "Place receptor files (.pdb/.pdbqt) in {dir}",
        "dock_label_receptor": "Select Receptor",
        "dock_title_space": "Search Space",
        "dock_title_box": "Box Size (Å)",
        "dock_title_calc": "Calculation",
        "dock_label_exhaustiveness": "Exhaustiveness",
        "dock_help_exhaustiveness": "Higher values are more accurate but slower.",
        "dock_card_run": "Run",
        "dock_text_ready": "Click the button to start docking.",
        "dock_btn_start": "Start AutoDock Vina Docking",
        "dock_err_select_receptor": "Select a receptor file first.",
        "dock_info_prep": "Preparing files...",
        "dock_err_prep": "Preparation failed.",
        "dock_info_vina_start": "Vina is running...",
        "dock_spinner_vina": "Docking...",
        "dock_success": "Docking complete.",
        "dock_label_best_affinity": "Best Affinity",
        "dock_card_3d": "3D Visualization (Molstar)",
        "dock_err_3d_load": "Visualization failed:",
        "dock_warn_no_pdb": "No docking output PDB found.",
        "dock_card_scores": "Docking Scores",
        "dock_info_no_scores": "No scores.",
        "dock_title_ai_report": "AI Report",
        "dock_info_no_report": "No report.",
        "dock_err_unknown": "Docking error:",
        "dock_err_box_calc": "Failed to compute docking box from the full receptor: {err}",
        "dock_err_deps_check": "Docking dependency check failed: {err}",
        "dock_label_pose_select": "Select docking pose (for 3D view & interaction analysis)",
        "dock_molstar_hint": "Tip: first open the controls panel from the top-right settings/sliders button in the 3D view, then find 'Non-covalent Interactions' in the structure tree and click its eye icon. If it is missing, use the PLIP interaction analysis below.",
        "dock_plip_section_title": "Interaction Deep Analysis (2D map & table)",
        "dock_plip_unavailable": "PLIP is not installed or unavailable: {msg}. Install plip to enable 2D interaction analysis.",
        "dock_plip_card_title": "Protein–Ligand Interaction Analysis (PLIP)",
        "dock_plip_no_image": "No 2D interaction image generated (check PLIP output).",
        "dock_plip_table_title": "Interaction Details",
        "dock_plip_no_interactions": "No interactions parsed (check the PLIP report).",
        "ai_title": "DeepSeek Assistant",
        "ai_caption": "Ask about targets, chemistry, or your results.",
        "ai_input_placeholder": "Type your question...",
        "ai_err_no_key": "API key not found. Set DEEPSEEK_API_KEY.",
        "ai_spinner": "Thinking...",
        "ai_err_no_response": "No response.",
        "ai_err_fail": "AI call failed:",
        "ai_float_btn": "Chat with AI",
        "ai_send_btn": "Send",
        "ai_clear_btn": "Clear",
        "export_title": "Export",
        "export_desc": "Save current page outputs as an offline HTML report (not PDF).",
        "export_btn_html": "Download Full Web Report (.html)",
        "export_no_content": "Nothing to export yet. Run prediction or docking first.",
        "molstar_diag_title": "Molstar / WebGL Diagnostics",
        "molstar_diag_impl": "Current renderer",
        "molstar_diag_impl_compat": "Compat-patched component (enabled)",
        "molstar_diag_impl_native": "Original component (compat patch disabled)",
        "molstar_diag_native_warn": "Compat patch is not active; WebGL initialization errors may persist.",
        "molstar_diag_compat_error": "Compat patch import failed",
        "molstar_diag_status_title": "Patch Status",
        "molstar_diag_no_status": "No compat status is available.",
        "molstar_diag_probe_title": "Browser WebGL Probe",
        "molstar_diag_probe_caption": "The result below is probed in your current browser context (WebGL1/WebGL2).",
    },
}


def t(key: str) -> str:
    lang = st.session_state.get("language_choice", "中文")
    base = TRANSLATIONS.get(lang, TRANSLATIONS["English"])
    if key in base:
        return base[key]
    return TRANSLATIONS["English"].get(key, key)

def _is_zh(lang_label: str) -> bool:
    if not lang_label:
        return True
    lab = str(lang_label).strip()
    return lab == "中文" or lab.lower().startswith("zh")

TOX_ALERT_LABEL_EN = {
    "alert_nitro_aromatic": "Aromatic nitro",
    "alert_nitro_aliphatic": "Aliphatic nitro",
    "alert_azide": "Azide",
    "alert_aniline": "Aniline",
    "alert_anilide": "Anilide",
    "alert_isocyanate": "Isocyanate",
    "alert_michael_acceptor": "Michael acceptor",
    "alert_epoxide": "Epoxide",
    "alert_organohalide": "Polyhalogenated",
    "alert_quinone": "Quinone",
    "alert_thioamide": "Thioamide",
    "alert_hydrazine": "Hydrazine / hydrazine-like",
}

def _localize_target_name(raw_name: str, display_name: str, lang_label: str) -> str:
    if _is_zh(lang_label):
        return display_name or raw_name or ""
    raw = (raw_name or "").strip()
    disp = (display_name or "").strip()
    if raw.lower() == "decoy" or disp == "非雌激素受体靶向性":
        return "Decoy (Non-ER target)"
    return disp or raw or ""

def _localize_tox_label(alert_name: str, label: str, lang_label: str) -> str:
    if _is_zh(lang_label):
        return label or alert_name or ""
    if alert_name in TOX_ALERT_LABEL_EN:
        return TOX_ALERT_LABEL_EN[alert_name]
    return label or alert_name or ""

def _localize_pred_results(pred_results, lang_label: str):
    if not isinstance(pred_results, list):
        return pred_results
    is_zh = _is_zh(lang_label)
    cluster_labels = (
        {
            "decoy": "非雌激素受体靶向性",
            "esr1": "ESR1靶向性",
            "esr2": "ESR2靶向性",
            "gper1": "GPER1靶向性",
            "other": "非雌激素受体靶向性",
        }
        if is_zh
        else {
            "decoy": "Decoy (Non-ER target)",
            "esr1": "ESR1 targeting",
            "esr2": "ESR2 targeting",
            "gper1": "GPER1 targeting",
            "other": "Non-ER / unknown",
        }
    )
    out = []
    for row in pred_results:
        if not isinstance(row, dict):
            out.append(row)
            continue
        raw_name = str(row.get("Raw Target", row.get("Target Name", "")) or "").strip()
        disp_name = str(row.get("Target Name", "") or "").strip()
        key = (raw_name or disp_name).lower()

        if key == "decoy" or disp_name == "非雌激素受体靶向性" or "decoy" in key:
            cluster_key = "decoy"
        elif "esr1" in key:
            cluster_key = "esr1"
        elif "esr2" in key:
            cluster_key = "esr2"
        elif "gper1" in key:
            cluster_key = "gper1"
        else:
            cluster_key = "other"
        cluster = cluster_labels.get(cluster_key, cluster_labels["other"])

        drop_cols = {"Target Name", "Raw Target", "Is Decoy"}
        r = {}
        if "Rank" in row:
            r["Rank"] = row.get("Rank")
        r["Predicted Cluster"] = cluster
        if "Probability" in row:
            r["Probability"] = row.get("Probability")
        for k, v in row.items():
            if k in drop_cols or k in r:
                continue
            r[k] = v
        out.append(r)
    return out

def _localize_toxicity_results(toxicity_results, lang_label: str):
    if not isinstance(toxicity_results, list):
        return toxicity_results
    out = []
    for row in toxicity_results:
        if not isinstance(row, dict):
            out.append(row)
            continue
        r = dict(row)
        r["Label"] = _localize_tox_label(r.get("Alert", ""), r.get("Label", ""), lang_label)
        out.append(r)
    return out


def _assess_risk(smiles, pred_results, toxicity_results, ac50_info, lang_label: str):
    try:
        return backend.assess_toxicity_risk(
            smiles=smiles,
            pred_results=pred_results or [],
            toxicity_results=toxicity_results or [],
            ac50_info=ac50_info or {},
            lang=lang_label,
        )
    except Exception:
        is_risk = bool(toxicity_results)
        return {
            "is_risk": is_risk,
            "risk_level": "concern" if is_risk else "pass",
            "status_text": t("tox_risk") if is_risk else t("tox_pass"),
            "basis": [],
        }


LANGUAGE_MAP = {
    "中文": "简体中文",
    "English": "English",
    "日本語": "日本語",
    "한국어": "한국어",
    "Español": "Español",
    "Français": "Français",
    "Deutsch": "Deutsch",
}


def _translate_pubchem_html_if_needed(html_text: str, lang_label: str, cache_key: str):
    if not html_text:
        return html_text
    if lang_label in {"English"}:
        return html_text
    cache = st.session_state.setdefault("pubchem_translate_cache", {})
    if cache_key in cache:
        return cache[cache_key]
    if not api_key:
        if not st.session_state.get("pubchem_translate_warned"):
            st.warning(t("pubchem_translate_fallback"))
            st.session_state.pubchem_translate_warned = True
        return html_text

    target_lang = LANGUAGE_MAP.get(lang_label, lang_label)
    with st.spinner(t("pubchem_translate_spinner")):
        translated = backend.translate_pubchem_html(
            html_text,
            target_lang,
            api_key=api_key,
            model_name="deepseek-chat",
            timeout=60,
        )
    if translated:
        cache[cache_key] = translated
        return translated
    return html_text


# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 定义渲染美化版聊天记录的函数
def _render_beautiful_chat_html(messages):
    rows = []
    avatar_html = (
        f"<div class='ai-avatar'>"
        f"<img src=\"data:image/jpeg;base64,{DEEPSEEK_AVATAR_B64}\" alt=\"DeepSeek\" />"
        f"</div>"
    )
    # 欢迎语
    if not messages:
        rows.append(
            "<div class='ai-row assistant'>"
            f"{avatar_html}"
            "<div class='ai-msg assistant'>"
            "<b>DeepSeek 科学助手</b><br>"
            f"{html.escape(t('ai_caption'))}"
            "</div>"
            "</div>"
        )
    for m in (messages or [])[-80:]:
        role = (m or {}).get("role", "assistant")
        content = (m or {}).get("content", "")
        safe = html.escape(str(content)).replace("\n", "<br>")

        if role == "user":
            rows.append(
                "<div class='ai-row user'>"
                f"<div class='ai-msg user'>{safe}</div>"
                f"{avatar_html}"
                "</div>"
            )
        else:
            rows.append(
                "<div class='ai-row assistant'>"
                f"{avatar_html}"
                f"<div class='ai-msg assistant'>{safe}</div>"
                "</div>"
            )

    return "<div class='ai-chat-scroll'>" + "".join(rows) + "</div>"


def _report_table_html(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "<p class='report-muted'>N/A</p>"
    try:
        return df.to_html(index=False, escape=True, border=0, classes="report-table")
    except Exception:
        return f"<pre>{html.escape(str(df))}</pre>"


def _report_text(v) -> str:
    return html.escape(str(v)).replace("\n", "<br>")


def _has_export_content() -> bool:
    if st.session_state.get("pred_batch"):
        return True
    if st.session_state.get("tox_report"):
        return True
    if st.session_state.get("dock_last_result"):
        return True
    return False


def _build_full_report_html(lang_label: str) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    title = t("header_title")
    sections = []

    # Prediction summary and details
    batch = st.session_state.get("pred_batch") or []
    pred_input_text = st.session_state.get("pred_input_text")
    current_smiles_input = st.session_state.get("smiles_input", "")
    has_current_pred = bool(batch) and pred_input_text == current_smiles_input

    pred_parts = []
    if batch:
        if not has_current_pred:
            pred_parts.append(
                "<p class='report-muted'>"
                + (_report_text(t("pred_info_smiles_changed")))
                + "</p>"
            )
        summary_rows = []
        for item in batch:
            smi = item.get("smiles", "")
            if item.get("error"):
                summary_rows.append(
                    {
                        "SMILES": smi,
                        "Top 1": "Error",
                        "Probability": "",
                        "AC50 (nM)": "",
                        "Risk": "Error",
                        "Error": item.get("error"),
                    }
                )
                continue

            pr = item.get("pred_results") or []
            top1 = pr[0] if pr else {}
            raw_t1 = str(top1.get("Raw Target", top1.get("Target Name", "")) or "")
            top1_name = _localize_target_name(raw_t1, top1.get("Target Name", ""), lang_label)
            prob = top1.get("Probability", None)
            ac50_info = item.get("ac50_info") or item.get("ic50_info") or {}
            ac50_nm = None
            if isinstance(ac50_info, dict):
                ac50_nm = ac50_info.get("AC50_nM", ac50_info.get("IC50_nM"))
            risk_info = _assess_risk(smi, pr, item.get("toxicity_results") or [], ac50_info, lang_label)
            summary_rows.append(
                {
                    "SMILES": smi,
                    "Top 1": top1_name or "Unknown",
                    "Probability": prob,
                    "AC50 (nM)": ac50_nm,
                    "Risk": risk_info.get("status_text", ""),
                    "Error": "",
                }
            )

        pred_parts.append(f"<h3>{_report_text(t('pred_card_result'))}</h3>")
        pred_parts.append(_report_table_html(pd.DataFrame(summary_rows)))

        pred_parts.append(f"<h3>{_report_text(t('pred_table_title'))}</h3>")
        for item in batch:
            smi = item.get("smiles", "")
            pred_parts.append(f"<h4>SMILES: {_report_text(smi)}</h4>")
            if item.get("error"):
                pred_parts.append(f"<p>{_report_text(item.get('error'))}</p>")
                continue

            pred_results = item.get("pred_results") or []
            toxicity_results = item.get("toxicity_results") or []
            ac50_info = item.get("ac50_info") or item.get("ic50_info") or {}
            risk_info = _assess_risk(smi, pred_results, toxicity_results, ac50_info, lang_label)
            risk_basis = risk_info.get("basis") or []

            pred_view = _localize_pred_results(pred_results, lang_label)
            pred_parts.append(_report_table_html(pd.DataFrame(pred_view)))

            ac50_nm = ""
            if isinstance(ac50_info, dict):
                ac50_nm = ac50_info.get("AC50_nM", ac50_info.get("IC50_nM", ""))
            pred_parts.append(
                "<p><b>"
                + _report_text(t("tox_label"))
                + ":</b> "
                + _report_text(risk_info.get("status_text", ""))
                + " | <b>AC50 (nM):</b> "
                + _report_text(ac50_nm if ac50_nm != "" else "N/A")
                + "</p>"
            )
            if risk_basis:
                pred_parts.append("<ul>" + "".join([f"<li>{_report_text(x)}</li>" for x in risk_basis]) + "</ul>")

            pred_parts.append(f"<h5>{_report_text(t('tox_table_title'))}</h5>")
            if toxicity_results:
                tox_view = _localize_toxicity_results(toxicity_results, lang_label)
                pred_parts.append(_report_table_html(pd.DataFrame(tox_view)))
            else:
                pred_parts.append(f"<p class='report-muted'>{_report_text(t('tox_info_safe'))}</p>")
    else:
        pred_parts.append(f"<p class='report-muted'>{_report_text(t('pred_no_results'))}</p>")
    sections.append("<section><h2>" + _report_text(t("tab_pred")) + "</h2>" + "".join(pred_parts) + "</section>")

    # PubChem report
    pubchem_parts = []
    report = st.session_state.get("tox_report")
    report_smiles = st.session_state.get("tox_report_smiles")
    if report:
        pubchem_parts.append(f"<p><b>SMILES:</b> {_report_text(report_smiles or '')}</p>")
        pubchem_parts.append(f"<h3>{_report_text(t('tox_pubchem_ghs'))}</h3>")
        pubchem_parts.append(str(report.get("ghs_data", "")))
        pubchem_parts.append(f"<h3>{_report_text(t('tox_pubchem_tox'))}</h3>")
        pubchem_parts.append(str(report.get("toxicity_data", "")))
    else:
        pubchem_parts.append(f"<p class='report-muted'>{_report_text(t('tox_pubchem_hint'))}</p>")
    sections.append(
        "<section><h2>"
        + _report_text(t("tox_pubchem_card"))
        + "</h2>"
        + "".join(pubchem_parts)
        + "</section>"
    )

    # Docking section
    docking_parts = []
    dock_result = st.session_state.get("dock_last_result")
    if isinstance(dock_result, dict) and "error" not in dock_result:
        receptor_name = st.session_state.get("dock_last_receptor") or st.session_state.get("dock_receptor")
        dock_smiles = st.session_state.get("dock_last_smiles", "")
        docking_parts.append(f"<p><b>Receptor:</b> {_report_text(receptor_name or '')}</p>")
        docking_parts.append(f"<p><b>SMILES:</b> {_report_text(dock_smiles)}</p>")

        scores = dock_result.get("docking_scores") or []
        if scores:
            best_affinity = scores[0].get("affinity", "N/A")
            docking_parts.append(f"<p><b>{_report_text(t('dock_label_best_affinity'))}:</b> {_report_text(best_affinity)} kcal/mol</p>")
            docking_parts.append(_report_table_html(pd.DataFrame(scores)))
        else:
            docking_parts.append(f"<p class='report-muted'>{_report_text(t('dock_info_no_scores'))}</p>")

        analysis_res = dock_result.get("analysis_result")
        docking_parts.append(f"<h3>{_report_text(t('dock_title_ai_report'))}</h3>")
        if analysis_res:
            docking_parts.append(f"<p>{_report_text(analysis_res)}</p>")
        else:
            docking_parts.append(f"<p class='report-muted'>{_report_text(t('dock_info_no_report'))}</p>")

        pose_results = dock_result.get("pose_results") or []
        if pose_results:
            pose_rows = []
            for p in pose_results:
                pose_rows.append(
                    {
                        "Pose": p.get("pose_index"),
                        "Affinity (kcal/mol)": p.get("affinity"),
                        "Ligand PDB": p.get("ligand_pdb"),
                        "Complex PDB": p.get("complex_pdb"),
                    }
                )
            docking_parts.append("<h3>Poses</h3>")
            docking_parts.append(_report_table_html(pd.DataFrame(pose_rows)))
    elif isinstance(dock_result, dict) and dock_result.get("error"):
        docking_parts.append(f"<p>{_report_text(dock_result.get('error'))}</p>")
    else:
        docking_parts.append(f"<p class='report-muted'>{_report_text(t('dock_info_no_report'))}</p>")

    # Optional PLIP section
    plip_res = st.session_state.get("dock_last_plip")
    if isinstance(plip_res, dict):
        docking_parts.append(f"<h3>{_report_text(t('dock_plip_section_title'))}</h3>")
        if plip_res.get("error"):
            docking_parts.append(f"<p>{_report_text(plip_res.get('error'))}</p>")
        else:
            img_path = plip_res.get("image_path")
            if img_path and Path(img_path).exists():
                p = Path(img_path)
                suffix = p.suffix.lower()
                mime = "image/png" if suffix == ".png" else "image/jpeg"
                try:
                    img_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
                    docking_parts.append(
                        f"<img src='data:{mime};base64,{img_b64}' "
                        "style='max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:6px;' />"
                    )
                except Exception:
                    pass
            interactions = plip_res.get("interactions") or []
            if interactions:
                docking_parts.append(_report_table_html(pd.DataFrame(interactions)))

    sections.append("<section><h2>" + _report_text(t("tab_dock")) + "</h2>" + "".join(docking_parts) + "</section>")

    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_report_text(title)} - Export</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f8fafc;
    }}
    h1 {{ margin: 0 0 8px 0; }}
    .meta {{ color: #64748b; margin-bottom: 16px; }}
    section {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 14px;
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin: 8px 0 12px 0;
    }}
    .report-table th, .report-table td {{
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      vertical-align: top;
      text-align: left;
    }}
    .report-table th {{
      background: #f1f5f9;
    }}
    .report-muted {{
      color: #64748b;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 8px;
    }}
  </style>
</head>
<body>
  <h1>{_report_text(title)}</h1>
  <div class="meta">Exported at: {_report_text(ts)}</div>
  {''.join(sections)}
</body>
</html>
"""


def _webgl_probe_html() -> str:
    return """
<div style="font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px;">
  <pre id="webgl-probe-output" style="margin:0; white-space:pre-wrap; word-break:break-word;"></pre>
</div>
<script>
(function () {
  function probe(type) {
    var canvas = document.createElement('canvas');
    var result = { type: type, ok: false };
    try {
      var gl = canvas.getContext(type, { failIfMajorPerformanceCaveat: false });
      if (!gl) {
        result.reason = 'context is null';
        return result;
      }
      result.ok = true;
      result.version = gl.getParameter(gl.VERSION);
      result.vendor = gl.getParameter(gl.VENDOR);
      result.renderer = gl.getParameter(gl.RENDERER);
      var dbg = gl.getExtension('WEBGL_debug_renderer_info');
      if (dbg) {
        result.unmasked_vendor = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL);
        result.unmasked_renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
      }
    } catch (e) {
      result.reason = String(e);
    }
    return result;
  }

  var out = {
    userAgent: navigator.userAgent,
    webgl2: probe('webgl2'),
    webgl1: probe('webgl')
  };
  document.getElementById('webgl-probe-output').textContent = JSON.stringify(out, null, 2);
})();
</script>
"""


def _render_molstar_diagnostics():
    st.markdown("---")
    st.markdown(f"### {t('molstar_diag_title')}")

    impl_text = (
        t("molstar_diag_impl_compat")
        if MOLSTAR_VIEWER_IMPL == "compat"
        else t("molstar_diag_impl_native")
    )
    st.caption(f"{t('molstar_diag_impl')}: {impl_text}")

    if MOLSTAR_VIEWER_IMPL != "compat":
        st.warning(t("molstar_diag_native_warn"))
    if MOLSTAR_COMPAT_ERROR:
        st.error(f"{t('molstar_diag_compat_error')}: {MOLSTAR_COMPAT_ERROR}")

    with st.expander(t("molstar_diag_status_title"), expanded=False):
        if MOLSTAR_COMPAT_STATUS:
            st.code(
                json.dumps(MOLSTAR_COMPAT_STATUS, ensure_ascii=False, indent=2),
                language="json",
            )
        else:
            st.info(t("molstar_diag_no_status"))

    st.caption(t("molstar_diag_probe_caption"))
    components.html(_webgl_probe_html(), height=240, scrolling=False)


# Sidebar settings (整合了全局设置与美化版 AI 对话)
with st.sidebar:
    st.markdown(f"### {t('sidebar_global_settings')}")
    language_choice = st.selectbox(
        t("sidebar_language"),
        ["中文", "English", "日本語", "한국어", "Español", "Français", "Deutsch"],
        index=0,
        key="language_choice",
    )
    model_name = st.selectbox(
        t("sidebar_ai_model"),
        ["deepseek-chat", "deepseek-coder"],
        index=0,
        key="ai_model_choice",
    )

    st.markdown("---")
    st.markdown(f"### {t('ai_title')}")
    
    # 聊天记录显示区域 (容器固定高度，内部滚动)
    chat_container = st.container(height=400)
    with chat_container:
        st.markdown(_render_beautiful_chat_html(st.session_state.messages), unsafe_allow_html=True)

    # 输入表单
    with st.form("ai_sidebar_form", clear_on_submit=True):
        prompt = st.text_area(
            label="Input", 
            label_visibility="collapsed",
            placeholder=t("ai_input_placeholder"),
            height=80,
            key="ai_sidebar_prompt",
        )
        c_send, c_clear = st.columns([1, 1])
        send = c_send.form_submit_button(t("ai_send_btn"), use_container_width=True)
        clear = c_clear.form_submit_button(t("ai_clear_btn"), use_container_width=True)

    if clear:
        st.session_state.messages = []
        st.rerun()

    if send:
        if not str(prompt).strip():
            st.stop()
        if not api_key:
            st.error(t("ai_err_no_key"))
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner(t("ai_spinner")):
                try:
                    response = backend.chat_with_deepseek(prompt, api_key, model_name)
                    if response:
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                    else:
                        st.error(t("ai_err_no_response"))
                except Exception as e:
                    st.error(f"{t('ai_err_fail')} {e}")
            st.rerun()

    st.markdown("---")
    st.markdown(f"### {t('export_title')}")
    st.caption(t("export_desc"))
    if _has_export_content():
        report_html = _build_full_report_html(st.session_state.get("language_choice", "中文"))
        filename = f"QSER_Report_{time.strftime('%Y%m%d_%H%M%S')}.html"
        st.download_button(
            t("export_btn_html"),
            data=report_html.encode("utf-8"),
            file_name=filename,
            mime="text/html",
            use_container_width=True,
            key="download_full_report_html_btn",
        )
    else:
        st.info(t("export_no_content"))

    # Molstar/WebGL diagnostics are intentionally hidden from the sidebar UI.


# CSS 样式 (保留气泡样式，调整滚动条适配侧边栏)
st.markdown(
    f"""
<style>
html, body, [class*="css"] {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #2c3e50;
}}
.stApp {{ background-color: #f4f7f9; }}
.custom-header {{
  background: linear-gradient(120deg, #2980b9, #2c3e50);
  padding: 1.6rem 1.6rem;
  border-radius: 10px;
  color: white;
  margin-bottom: 1.6rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}}
.custom-header h1 {{
  font-weight: 700;
  margin: 0;
  font-size: 1.9rem;
  letter-spacing: 0.2px;
}}
.custom-header p {{
  margin: 0.5rem 0 0 0;
  font-size: 1rem;
  opacity: 0.92;
}}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElement"] .stCardMarker) {{
  background-color: #FFFFFF;
  padding: 1.2rem;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  border: 1px solid #e9ecef;
  margin-bottom: 1rem;
}}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElement"] .stCardMarker--center) {{
  text-align: center;
}}
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElement"] .stCardMarker) h3 {{
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 0.75rem;
}}
.stCardMarker {{
  display: none;
}}
.module-intro {{
  background-color: #FFFFFF;
  padding: 0.85rem 1rem;
  border-radius: 10px;
  border: 1px solid #e9ecef;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  margin-bottom: 1rem;
  color: #475569;
  font-size: 0.95rem;
}}
.module-intro .label {{
  font-weight: 700;
  color: #1f2937;
  margin-right: 0.5rem;
}}
.primary-btn > button {{
  background-color: #2980b9 !important;
  color: white !important;
  border: none !important;
  font-weight: 700 !important;
}}

/* =========================================
   美化版聊天气泡 CSS (适配 Sidebar)
   ========================================= */

/* 对话区域容器 */
.ai-chat-scroll {{
    padding-right: 5px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}}

/* 气泡行布局 */
.ai-row {{ display: flex; align-items: flex-end; gap: 6px; }}
.ai-row.user {{ justify-content: flex-end; }}
.ai-row.assistant {{ justify-content: flex-start; }}

/* 头像样式 */
.ai-avatar {{
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #f1f5f9;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    flex-shrink: 0;
}}
.ai-avatar img {{
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
    display: block;
}}

/* 气泡本体 */
.ai-msg {{
    max-width: 85%;
    padding: 10px 12px;
    border-radius: 12px;
    font-size: 0.9rem;
    line-height: 1.45;
    word-break: break-word;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}}
.ai-msg.user {{
    background: #2980b9;
    color: white;
    border-bottom-right-radius: 2px;
}}
.ai-msg.assistant {{
    background: #ffffff;
    color: #334155;
    border-bottom-left-radius: 2px;
    border: 1px solid #e2e8f0;
}}

/* 隐藏 Streamlit 表单的默认边框，让它在侧边栏更自然 */
[data-testid="stForm"] {{ border: none !important; padding: 0 !important; }}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    f"""
<div class="custom-header">
  <h1>{t('header_title')}</h1>
  <p>{t('header_desc')}</p>
</div>
""",
    unsafe_allow_html=True,
)


def _parse_smiles_input(text: str) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    for line in str(text).splitlines():
        s = line.strip()
        if not s:
            continue
        lines.append(s)
    return lines


def _estimate_molstar_height(scores, analysis_text) -> int:
    score_n = len(scores or [])
    line_n = len(str(analysis_text or "").splitlines())
    # Extend viewer height when the right-side report is long, reducing blank space under the 3D canvas.
    h = 620 + max(0, score_n - 8) * 12 + max(0, line_n - 10) * 28
    return int(min(1350, max(620, h)))


def _preview_alert_highlight(mol):
    """Return (highlight_atoms, alert_labels) for toxic alerts."""
    try:
        alert_atoms = set()
        alert_labels = []
        alerts = getattr(backend, "TOXIC_ALERTS", [])
        for _name, smarts, label in alerts:
            patt = Chem.MolFromSmarts(smarts)
            if not patt:
                continue
            matches = mol.GetSubstructMatches(patt)
            if matches:
                alert_labels.append(label)
                for m in matches:
                    alert_atoms.update(m)
        return sorted(alert_atoms), sorted(set(alert_labels))
    except Exception:
        return [], []


@contextmanager
def st_card(center: bool = False):
    with st.container():
        marker_class = "stCardMarker stCardMarker--center" if center else "stCardMarker"
        st.markdown(f"<div class='{marker_class}'></div>", unsafe_allow_html=True)
        yield


st.markdown(f"## {t('tab_pred')}")
st.markdown(
    f"""
<div class="module-intro">
  <span class="label">{t('module_intro_label')}</span>
  <span>{t('pred_module_intro')}</span>
</div>
""",
    unsafe_allow_html=True,
)

if "smiles_input" not in st.session_state:
    st.session_state.smiles_input = ""
if "load_example" not in st.session_state:
    st.session_state.load_example = False
if "pred_batch" not in st.session_state:
    st.session_state.pred_batch = None
if "pred_input_text" not in st.session_state:
    st.session_state.pred_input_text = None
if "preview_smiles" not in st.session_state:
    st.session_state.preview_smiles = ""
if "detail_smiles" not in st.session_state:
    st.session_state.detail_smiles = ""
if "iupac_cache" not in st.session_state:
    st.session_state.iupac_cache = {}
if "tox_report" not in st.session_state:
    st.session_state.tox_report = None
if "tox_report_smiles" not in st.session_state:
    st.session_state.tox_report_smiles = None
if "pubchem_cache" not in st.session_state:
    st.session_state.pubchem_cache = {}
if "dock_last_result" not in st.session_state:
    st.session_state.dock_last_result = None
if "dock_last_receptor" not in st.session_state:
    st.session_state.dock_last_receptor = None
if "dock_last_smiles" not in st.session_state:
    st.session_state.dock_last_smiles = None
if "dock_last_plip" not in st.session_state:
    st.session_state.dock_last_plip = None

if st.session_state.load_example:
    st.session_state.smiles_input = "CC(=O)OC1=CC=CC=C1C(=O)O"
    st.session_state.pred_batch = None
    st.session_state.pred_input_text = None
    st.session_state.tox_report = None
    st.session_state.tox_report_smiles = None
    st.session_state.load_example = False


with st_card():
    st.subheader(t("pred_card_input"))

    col_input, col_preview = st.columns([3, 2], gap="medium")
    with col_input:
        smiles_input = st.text_area(
            t("pred_input_placeholder"),
            key="smiles_input",
            height=120,
            placeholder=t("pred_input_placeholder"),
        )
        if st.button(t("pred_btn_example"), key="load_example_btn"):
            st.session_state.load_example = True
            st.rerun()

    with col_preview:
        st.markdown(f"##### {t('pred_preview_title')}")
        smiles_list = _parse_smiles_input(smiles_input)
        preview_smiles = ""
        if len(smiles_list) > 1:
            if st.session_state.preview_smiles not in smiles_list:
                st.session_state.preview_smiles = smiles_list[0]
            st.caption(f"检测到 {len(smiles_list)} 个 SMILES，预览仅显示所选分子。")
            preview_smiles = st.selectbox(
                "选择预览分子",
                options=smiles_list,
                index=smiles_list.index(st.session_state.preview_smiles),
                key="preview_smiles",
            )
        elif len(smiles_list) == 1:
            preview_smiles = smiles_list[0]
            st.session_state.preview_smiles = preview_smiles
        else:
            st.session_state.preview_smiles = ""

        if preview_smiles:
            try:
                mol = Chem.MolFromSmiles(preview_smiles)
                if mol:
                    alert_atoms, alert_labels = _preview_alert_highlight(mol)
                    if alert_atoms:
                        color = {i: (1.0, 0.5, 0.0) for i in alert_atoms}
                        img = Draw.MolToImage(
                            mol,
                            size=(420, 240),
                            highlightAtoms=alert_atoms,
                            highlightAtomColors=color,
                        )
                    else:
                        img = Draw.MolToImage(mol, size=(420, 240))
                    st.image(img, use_container_width=True, output_format="PNG")
                    if alert_labels:
                        st.caption(f"检测到结构警示: {', '.join(alert_labels)}")

                    # IUPAC name under 2D preview (cached)
                    iupac_name = st.session_state.iupac_cache.get(preview_smiles)
                    if iupac_name is None:
                        name, _err = backend.get_iupac_name(preview_smiles)
                        if name:
                            iupac_name = name
                        else:
                            canonical = backend.canonicalize_smiles(preview_smiles)
                            if canonical:
                                iupac_name = f"未检索到 IUPAC，显示标准 SMILES: {canonical}"
                            else:
                                iupac_name = "暂无"
                        st.session_state.iupac_cache[preview_smiles] = iupac_name
                    st.caption(f"系统命名: {iupac_name}")

                else:
                    st.warning(t("pred_warn_invalid"))
            except Exception:
                st.warning(t("pred_warn_failed"))
        else:
            st.info(t("pred_info_preview"))


st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
pred_submitted = st.button(t("pred_btn_start"), use_container_width=True, key="pred_start_btn")
st.markdown("</div>", unsafe_allow_html=True)

if pred_submitted:
    smiles_list = _parse_smiles_input(smiles_input)
    if not smiles_list:
        st.error(t("pred_err_no_smiles"))
    else:
        with st.spinner(t("pred_spinner")):
            try:
                batch_results = []
                for smi in smiles_list:
                    pred_out = backend.target_prediction(smi)
                    pred_results = None
                    toxicity_results = None
                    ac50_info = None

                    if isinstance(pred_out, (list, tuple)) and len(pred_out) >= 3:
                        pred_results, toxicity_results, ac50_info = pred_out[:3]
                    elif isinstance(pred_out, (list, tuple)) and len(pred_out) == 2:
                        pred_results, toxicity_results = pred_out
                    else:
                        pred_results = {"error": t("pred_err_format")}
                        toxicity_results = []

                    if isinstance(pred_results, dict) and "error" in pred_results:
                        batch_results.append(
                            {
                                "smiles": smi,
                                "error": pred_results.get("error"),
                                "pred_results": None,
                                "toxicity_results": None,
                                "ac50_info": None,
                            }
                        )
                    else:
                        batch_results.append(
                            {
                                "smiles": smi,
                                "error": None,
                                "pred_results": pred_results,
                                "toxicity_results": toxicity_results or [],
                                "ac50_info": ac50_info or {},
                            }
                        )

                st.session_state.pred_batch = batch_results
                st.session_state.pred_input_text = smiles_input
                st.session_state.tox_report = None
                st.session_state.tox_report_smiles = None
                st.success(t("pred_success"))
            except Exception as e:
                st.error(f"{t('tox_err_unknown')} {e}")


has_results = (
    st.session_state.pred_batch is not None and st.session_state.pred_input_text == smiles_input
)
if st.session_state.pred_batch is not None and st.session_state.pred_input_text != smiles_input:
    st.info(t("pred_info_smiles_changed"))


if has_results:
    batch = st.session_state.pred_batch or []
    summary_rows = []
    lang_label = st.session_state.get("language_choice", "中文")
    for item in batch:
        smi = item.get("smiles")
        if item.get("error"):
            summary_rows.append(
                {
                    "SMILES": smi,
                    "Top 1": "Error",
                    "Probability": None,
                    "AC50 (nM)": None,
                    "Decoy": None,
                    "Risk": None,
                    "Alerts": None,
                    "Error": item.get("error"),
                }
            )
            continue

        pr = item.get("pred_results") or []
        top1 = pr[0] if pr else {}
        raw_t1 = str(top1.get("Raw Target", top1.get("Target Name", "")))
        is_decoy = "decoy" in raw_t1.lower()
        top1_name = _localize_target_name(raw_t1, top1.get("Target Name", ""), lang_label)

        prob = top1.get("Probability", None)
        ac50_info = item.get("ac50_info") or item.get("ic50_info") or {}
        if isinstance(ac50_info, dict):
            ac50_nm = ac50_info.get("AC50_nM")
            if ac50_nm is None:
                ac50_nm = ac50_info.get("IC50_nM")
        else:
            ac50_nm = None

        tox_list = item.get("toxicity_results") or []
        alert_labels = []
        for a in tox_list:
            lab = _localize_tox_label(a.get("Alert", ""), a.get("Label", ""), lang_label)
            if lab:
                alert_labels.append(str(lab))
        risk_info = _assess_risk(smi, pr, tox_list, ac50_info, lang_label)
        risk_text = risk_info.get("status_text", t("tox_risk") if risk_info.get("is_risk") else t("tox_pass"))

        summary_rows.append(
            {
                "SMILES": smi,
                "Top 1": top1_name or "Unknown",
                "Probability": prob,
                "AC50 (nM)": ac50_nm,
                "Decoy": bool(is_decoy),
                "Risk": risk_text,
                "Alerts": ", ".join(sorted(set(alert_labels))) if alert_labels else "",
                "Error": None,
            }
        )

    with st_card():
        st.subheader(t("pred_card_result"))
        summary_df = pd.DataFrame(summary_rows)
        if len(summary_df) > 0:
            row_h = 35
            header_h = 35
            height = min(600, header_h + row_h * len(summary_df))
        else:
            height = 120
        st.dataframe(summary_df, use_container_width=True, height=height)

    smi_options = [item.get("smiles") for item in batch if item.get("smiles")]
    if smi_options:
        if st.session_state.detail_smiles not in smi_options:
            st.session_state.detail_smiles = smi_options[0]
        detail_smiles = st.selectbox(
            t("pred_select_smiles"),
            options=smi_options,
            index=smi_options.index(st.session_state.detail_smiles),
            key="detail_smiles",
        )

        selected = None
        for item in batch:
            if item.get("smiles") == detail_smiles:
                selected = item
                break

        if selected is not None:
            if selected.get("error"):
                st.error(selected.get("error"))
            else:
                pred_results = selected.get("pred_results") or []
                toxicity_results = selected.get("toxicity_results") or []
                ac50_info = selected.get("ac50_info") or selected.get("ic50_info") or {}
                risk_info = _assess_risk(detail_smiles, pred_results, toxicity_results, ac50_info, lang_label)
                is_toxic = bool(risk_info.get("is_risk"))
                tox_status = risk_info.get("status_text", t("tox_risk") if is_toxic else t("tox_pass"))
                tox_basis = risk_info.get("basis") or []

                if isinstance(pred_results, list) and len(pred_results) > 0:
                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                    try:
                        target1 = pred_results[0]
                        raw_t1 = str(target1.get("Raw Target", target1.get("Target Name", "")))
                        if "decoy" in raw_t1.lower():
                            st.info(t("pred_decoy_info"))
                        target1_name = _localize_target_name(
                            raw_t1,
                            target1.get("Target Name", ""),
                            lang_label,
                        )

                        m_col1.metric(
                            label=f"Top 1: {target1_name or 'Unknown'}",
                            value=f"{float(target1.get('Probability', 0)):.4f}",
                        )
                        if len(pred_results) > 1:
                            target2 = pred_results[1]
                            target2_name = _localize_target_name(
                                target2.get("Raw Target", target2.get("Target Name", "")),
                                target2.get("Target Name", ""),
                                lang_label,
                            )
                            m_col2.metric(
                                label=f"Top 2: {target2_name or 'Unknown'}",
                                value=f"{float(target2.get('Probability', 0)):.4f}",
                            )
                        else:
                            m_col2.metric(label="Top 2", value="N/A")

                        if isinstance(ac50_info, dict) and ("AC50_nM" in ac50_info or "IC50_nM" in ac50_info):
                            ac50_nm = ac50_info.get("AC50_nM", ac50_info.get("IC50_nM"))
                            m_col3.metric(label="AC50 (nM)", value=f"{float(ac50_nm):.4g}")
                        else:
                            m_col3.metric(label="AC50 (nM)", value="N/A")
                    except Exception as e:
                        st.warning(f"{t('pred_warn_parse')} {e}")

                    tox_color = "normal" if not is_toxic else "inverse"
                    m_col4.metric(label=t("tox_label"), value=tox_status, delta_color=tox_color)
                    if tox_basis:
                        st.caption(f"{t('tox_risk_basis_title')}: " + " | ".join([str(x) for x in tox_basis]))

                # Target intro
                try:
                    top_target = ""
                    if isinstance(pred_results, list) and len(pred_results) > 0:
                        top_target = str(pred_results[0].get("Raw Target", pred_results[0].get("Target Name", "")))
                    if top_target:
                        st.markdown(f"#### {t('target_intro_title')}")
                        st.markdown(
                            backend.render_target_intro_html(top_target, is_toxic, lang_label),
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.warning(f"{t('target_intro_fail')} {e}")

                col_pred_table, col_tox_table = st.columns(2, gap="medium")
                with col_pred_table:
                    st.markdown(f"#### {t('pred_table_title')}")
                    pred_results_view = _localize_pred_results(pred_results, lang_label)
                    pred_df = pd.DataFrame(pred_results_view)
                    if len(pred_df) > 0:
                        row_h = 35
                        header_h = 35
                        pred_h = min(320, header_h + row_h * len(pred_df))
                    else:
                        pred_h = 120
                    try:
                        st.dataframe(pred_df, use_container_width=True, height=pred_h, hide_index=True)
                    except TypeError:
                        st.dataframe(pred_df, use_container_width=True, height=pred_h)
                with col_tox_table:
                    st.markdown(f"#### {t('tox_table_title')}")
                    if toxicity_results:
                        tox_view = _localize_toxicity_results(toxicity_results, lang_label)
                        tox_df = pd.DataFrame(tox_view)
                        if len(tox_df) > 0:
                            row_h = 35
                            header_h = 35
                            tox_h = min(320, header_h + row_h * len(tox_df))
                        else:
                            tox_h = 120
                        try:
                            st.dataframe(tox_df, use_container_width=True, height=tox_h, hide_index=True)
                        except TypeError:
                            st.dataframe(tox_df, use_container_width=True, height=tox_h)
                    else:
                        if is_toxic:
                            st.warning(t("tox_info_no_alert_risk"))
                            if tox_basis:
                                st.caption(f"{t('tox_risk_basis_title')}: " + " | ".join([str(x) for x in tox_basis]))
                        else:
                            st.info(t("tox_info_safe"))

    # PubChem GHS report (on demand) for selected compound
    report_smiles = st.session_state.detail_smiles or st.session_state.preview_smiles
    if report_smiles:
        with st.expander(t("tox_pubchem_card"), expanded=False):
            if st.button(t("tox_pubchem_btn"), key="tox_report_btn"):
                with st.spinner(t("tox_pubchem_spinner")):
                    try:
                        selected = None
                        for item in batch:
                            if item.get("smiles") == report_smiles:
                                selected = item
                                break
                        if selected and not selected.get("error"):
                            pr = selected.get("pred_results") or []
                            target_name = ""
                            if pr:
                                target_name = str(pr[0].get("Raw Target", pr[0].get("Target Name", "")))
                            lang_label = st.session_state.get("language_choice", "中文")
                            report = backend.comprehensive_toxicity_analysis(
                                report_smiles,
                                target_name,
                                selected.get("ac50_info") or selected.get("ic50_info"),
                                lang=lang_label,
                            )
                            st.session_state.tox_report = report
                            st.session_state.tox_report_smiles = report_smiles
                        else:
                            st.error(t("pred_no_results"))
                    except Exception as e:
                        st.error(f"{t('tox_pubchem_err')} {e}")

            report = st.session_state.get("tox_report")
            if report and st.session_state.get("tox_report_smiles") == report_smiles:
                lang_label = st.session_state.get("language_choice", "中文")
                st.markdown(f"#### {t('tox_pubchem_ghs')}")
                ghs_html = report.get("ghs_data", "暂无数据")
                ghs_html = _translate_pubchem_html_if_needed(
                    ghs_html,
                    lang_label,
                    f"{report_smiles}|ghs|{lang_label}",
                )
                st.markdown(ghs_html, unsafe_allow_html=True)
                st.markdown(f"#### {t('tox_pubchem_tox')}")
                tox_html = report.get("toxicity_data", "暂无数据")
                tox_html = _translate_pubchem_html_if_needed(
                    tox_html,
                    lang_label,
                    f"{report_smiles}|tox|{lang_label}",
                )
                st.markdown(tox_html, unsafe_allow_html=True)
            else:
                st.info(t("tox_pubchem_hint"))


st.markdown(f"## {t('tab_dock')}")
st.markdown(
    f"""
<div class="module-intro">
  <span class="label">{t('module_intro_label')}</span>
  <span>{t('dock_module_intro')}</span>
</div>
""",
    unsafe_allow_html=True,
)

lang_label = st.session_state.get("language_choice", "中文")
receptor_dir = Path(Config.RECEPTOR_DIR)
receptor_files = []
if receptor_dir.exists():
    receptor_files = sorted(
        [p.name for p in receptor_dir.iterdir() if p.is_file() and p.suffix.lower() in {".pdb", ".pdbqt"}]
    )

dock_col_params, dock_col_run = st.columns([2, 1], gap="medium")

with dock_col_params:
    with st_card():
        st.subheader(t("dock_card_params"))

        if not receptor_files:
            st.error(t("dock_err_no_receptor").format(dir=str(receptor_dir)))
            receptor_name = None
        else:
            receptor_name = st.selectbox(t("dock_label_receptor"), receptor_files, key="dock_receptor")

        # Global docking box (computed from the full receptor) and locked.
        center_x, center_y, center_z = float(Config.CENTER_X), float(Config.CENTER_Y), float(Config.CENTER_Z)
        size_x, size_y, size_z = float(Config.SIZE_X), float(Config.SIZE_Y), float(Config.SIZE_Z)
        if receptor_name:
            center, size, box_err = backend.get_receptor_box(receptor_name)
            if box_err:
                st.error(t("dock_err_box_calc").format(err=box_err))
            else:
                # 隐藏具体的数值展示，只在后台计算
                center_x, center_y, center_z = [float(x) for x in center]
                size_x, size_y, size_z = [float(x) for x in size]

        st.markdown(f"#### {t('dock_title_calc')}")
        exhaustiveness = st.slider(
            t("dock_label_exhaustiveness"),
            min_value=1,
            max_value=64,
            value=int(Config.EXHAUSTIVENESS),
            help=t("dock_help_exhaustiveness"),
        )

        tools_ok, tools_msg = True, ""
        try:
            tools_ok, tools_msg = backend.check_docking_tools(lang=lang_label)
        except Exception as e:
            tools_ok, tools_msg = False, t("dock_err_deps_check").format(err=e)
        if not tools_ok and tools_msg:
            st.warning(tools_msg)

with dock_col_run:
    with st_card(center=True):
        st.subheader(t("dock_card_run"))
        st.write(t("dock_text_ready"))

        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        dock_submitted = st.button(
            t("dock_btn_start"),
            use_container_width=True,
            disabled=(receptor_name is None or (not tools_ok)),
            key="dock_start_btn",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        progress_placeholder = st.empty()
        status_placeholder = st.empty()

if dock_submitted:
    if not receptor_name:
        st.error(t("dock_err_select_receptor"))
    else:
        status_placeholder.info(t("dock_info_prep"))
        progress_placeholder.progress(10)

        dock_smiles = st.session_state.get("preview_smiles") or ""
        if not dock_smiles:
            for line in str(st.session_state.get("smiles_input", "")).splitlines():
                s = line.strip()
                if s:
                    dock_smiles = s
                    break
        if not dock_smiles:
            dock_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

        try:
            st.session_state.dock_smiles = dock_smiles
            prep_success = backend.docking_preparation(receptor_name, dock_smiles)
            if not prep_success:
                status_placeholder.error(t("dock_err_prep"))
                progress_placeholder.empty()
            else:
                progress_placeholder.progress(30)
                status_placeholder.info(t("dock_info_vina_start"))
                with st.spinner(t("dock_spinner_vina")):
                    docking_results = backend.run_docking(
                        center_x, center_y, center_z, size_x, size_y, size_z, exhaustiveness, lang=lang_label
                    )

                progress_placeholder.progress(100)
                status_placeholder.success(t("dock_success"))
                time.sleep(0.6)
                progress_placeholder.empty()
                status_placeholder.empty()

                if isinstance(docking_results, dict) and "error" in docking_results:
                    st.session_state.dock_last_result = {"error": docking_results.get("error")}
                    st.session_state.dock_last_receptor = receptor_name
                    st.session_state.dock_last_smiles = dock_smiles
                    st.session_state.dock_last_plip = None
                    st.error(f"{t('dock_err_unknown')} {docking_results['error']}")
                else:
                    st.session_state.dock_last_result = docking_results
                    st.session_state.dock_last_receptor = receptor_name
                    st.session_state.dock_last_smiles = dock_smiles
                    st.session_state.dock_last_plip = None
                    scores = docking_results.get("docking_scores", [])
                    best_affinity = scores[0].get("affinity", "N/A") if scores else "N/A"
                    analysis_res = docking_results.get("analysis_result")
                    molstar_height = _estimate_molstar_height(scores, analysis_res)

                    with st_card():
                        st.metric(label=t("dock_label_best_affinity"), value=f"{best_affinity} kcal/mol")

                    col_3d, col_analysis = st.columns([3, 2], gap="medium")
                    
                    receptor_path = receptor_dir / receptor_name
                    ligand_pdb_path = Path(Config.OUTPUT_DIR) / "docked_ligand.pdb"
                    pose_results = docking_results.get("pose_results") or []
                    selected_ligand_pdb = ligand_pdb_path
                    selected_complex_pdb = None
                    pose_choice = None
                    if pose_results:
                        pose_labels = []
                        for p in pose_results:
                            idx = p.get("pose_index")
                            label = f"Pose {idx:03d}" if isinstance(idx, int) else f"Pose {idx}"
                            pose_labels.append(label)
                        pose_choice = st.selectbox(
                            t("dock_label_pose_select"),
                            options=pose_labels,
                            index=0,
                            key="dock_pose_select",
                        )
                        try:
                            sel_idx = pose_labels.index(pose_choice)
                            sel_pose = pose_results[sel_idx]
                            lig_pdb = sel_pose.get("ligand_pdb")
                            if lig_pdb and Path(lig_pdb).exists():
                                selected_ligand_pdb = Path(lig_pdb)
                            complex_pdb = sel_pose.get("complex_pdb")
                            if complex_pdb and Path(complex_pdb).exists():
                                selected_complex_pdb = Path(complex_pdb)
                        except Exception:
                            pass
                    
                    with col_3d:
                        with st_card():
                            st.subheader(t("dock_card_3d"))

                            if ligand_pdb_path.exists() and receptor_path.exists():
                                try:
                                    st_molstar_docking(
                                        str(receptor_path),
                                        str(selected_ligand_pdb),
                                        height=molstar_height
                                    )
                                    st.caption(t("dock_molstar_hint"))
                                except Exception as e:
                                    st.error(f"{t('dock_err_3d_load')} {e}")
                            else:
                                st.warning(t("dock_warn_no_pdb"))

                    with col_analysis:
                        with st_card():
                            st.subheader(t("dock_card_scores"))
                            if scores:
                                st.dataframe(pd.DataFrame(scores), use_container_width=True, hide_index=True, height=250)
                            else:
                                st.info(t("dock_info_no_scores"))

                            st.markdown("---")
                            st.subheader(t("dock_title_ai_report"))
                            if analysis_res:
                                st.markdown(analysis_res)
                            else:
                                st.info(t("dock_info_no_report"))
                    
                    # ==========================================
                    # 新增模块：2D 相互作用图与美化版统计表（PLIP）
                    # ==========================================
                    st.markdown("---")
                    st.markdown(f"### {t('dock_plip_section_title')}")

                    if selected_ligand_pdb.exists() and receptor_path.exists():

                        plip_ok, plip_msg = backend.check_plip_tool(lang=lang_label)
                        if not plip_ok:
                            st.warning(t("dock_plip_unavailable").format(msg=plip_msg))
                        else:
                            plip_smiles = st.session_state.get("dock_smiles") or st.session_state.get("preview_smiles") or ""
                            plip_res = backend.run_plip_2d_analysis(
                                str(receptor_path),
                                str(selected_ligand_pdb),
                                ligand_smiles=plip_smiles,
                                complex_path=str(selected_complex_pdb) if selected_complex_pdb else None,
                                pose_label=pose_choice,
                                lang=lang_label,
                            )
                            if plip_res.get("error"):
                                st.session_state.dock_last_plip = {"error": plip_res.get("error")}
                                st.warning(plip_res.get("error"))
                            else:
                                st.session_state.dock_last_plip = plip_res
                                col_2d, col_table = st.columns([5, 4], gap="large")

                                with col_2d:
                                    with st_card():
                                        st.markdown(f"#### {t('dock_plip_card_title')}")
                                        img_path = plip_res.get("image_path")
                                        if img_path and Path(img_path).exists():
                                            st.image(img_path, use_container_width=True)
                                        else:
                                            st.info(t("dock_plip_no_image"))

                                with col_table:
                                    with st_card():
                                        st.markdown(f"#### {t('dock_plip_table_title')}")
                                        interactions = plip_res.get("interactions") or []
                                        if interactions:
                                            df_ints = pd.DataFrame(interactions)
                                            prefer_cols = (
                                                [
                                                    "相互作用类型",
                                                    "化合物官能团",
                                                    "化合物原子",
                                                    "受体残基",
                                                    "氨基酸官能团",
                                                    "受体原子",
                                                    "距离 (Å)",
                                                ]
                                                if _is_zh(lang_label)
                                                else [
                                                    "Interaction Type",
                                                    "Ligand Functional Group",
                                                    "Ligand Atom",
                                                    "Receptor Residue",
                                                    "Residue Functional Group",
                                                    "Receptor Atom",
                                                    "Distance (Å)",
                                                ]
                                            )
                                            cols = [c for c in prefer_cols if c in df_ints.columns]
                                            if cols:
                                                df_ints = df_ints[cols]

                                            def highlight_interactions(row):
                                                tval = str(
                                                    row.get("相互作用类型", row.get("Interaction Type", ""))
                                                )
                                                low = tval.lower()
                                                if ("氢键" in tval) or ("hydrogen" in low):
                                                    return ['background-color: #ecfdf5; color: #065f46'] * len(row)
                                                if ("疏水" in tval) or ("hydrophobic" in low):
                                                    return ['background-color: #fffbeb; color: #92400e'] * len(row)
                                                if ("盐桥" in tval) or ("salt" in low):
                                                    return ['background-color: #eef2ff; color: #3730a3'] * len(row)
                                                return [''] * len(row)

                                            styled_df = df_ints.style.apply(highlight_interactions, axis=1)
                                            st.dataframe(
                                                styled_df,
                                                use_container_width=True,
                                                hide_index=True,
                                                height=400,
                                            )
                                        else:
                                            st.info(t("dock_plip_no_interactions"))

        except Exception as e:
            st.session_state.dock_last_result = {"error": str(e)}
            st.session_state.dock_last_plip = None
            st.error(f"{t('dock_err_unknown')} {e}")
