import io
import warnings
import zipfile
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import gaussian_kde
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "white"


@dataclass
class Settings:
    interval_min: float = 5.0
    smooth_multiplier: float = 2.0
    n_grid: int = 500
    n_quantile: int = 100
    joint_n_grid: int = 120
    gluc_min: float = 40.0
    gluc_max: float = 400.0
    vel_min: float = -5.0
    vel_max: float = 5.0
    acc_min: float = -3.0
    acc_max: float = 3.0
    rapid_rise_thr: float = 1.0
    rapid_acc_thr: float = 0.5


def make_demo_data(n_subjects: int = 12, n_days: int = 6, interval_min: int = 5) -> pd.DataFrame:
    np.random.seed(42)
    n_pts = int(24 * 60 / interval_min * n_days)

    def sim(mean_bg, sd_bg, spike_prob=0.04, spike_amp=45):
        x = np.zeros(n_pts)
        x[0] = mean_bg
        for i in range(1, n_pts):
            x[i] = 0.94 * x[i - 1] + 0.06 * mean_bg + np.random.normal(0, sd_bg * 0.18)
            if np.random.rand() < spike_prob:
                duration = np.random.randint(4, 18)
                height = spike_amp * np.random.rand()
                for d in range(min(duration, n_pts - i)):
                    x[i + d] += height * np.exp(-d / 5)
        miss_n = max(1, int(n_pts * 0.003))
        x[np.random.choice(n_pts, miss_n, replace=False)] = np.nan
        return np.clip(x, 40, 400)

    patients = {}
    configs = [
        ("Stable", 95, 12, 0.01),
        ("Mild_spike", 110, 22, 0.04),
        ("High_var", 125, 35, 0.07),
    ]
    k = 1
    for name, mu, sd, sp in configs:
        for _ in range(n_subjects // len(configs)):
            patients[f"{name}_{k:02d}"] = sim(mu + np.random.uniform(-5, 5), sd + np.random.uniform(-3, 3), sp)
            k += 1

    cols = [f"t{i:04d}" for i in range(n_pts)]
    df = pd.DataFrame.from_dict(patients, orient="index", columns=cols)
    df.index.name = "Patient_ID"
    return df


def make_template_excel() -> bytes:
    n_days = 8
    interval_min = 5
    n_points = int(24 * 60 / interval_min * n_days)
    n_per_day = int(24 * 60 / interval_min)

    time_cols = []
    for i in range(n_points):
        day = i // n_per_day + 1
        minutes = (i % n_per_day) * interval_min
        hh = minutes // 60
        mm = minutes % 60
        time_cols.append(f"day{day}_{hh:02d}:{mm:02d}")

    example = pd.DataFrame({"Patient_ID": ["CGM_01", "CGM_02", "CGM_03"]})
    for col in time_cols:
        example[col] = np.nan
    example.iloc[0, 1:11] = [100, 102, 103, 105, 104, 106, 108, 107, 109, 110]
    example.iloc[1, 1:11] = [95, 96, 98, 97, 99, 100, 101, 103, 102, 104]
    example.iloc[2, 1:11] = [120, 122, 121, 123, 125, 126, 124, 127, 129, 130]

    readme = pd.DataFrame({
        "項目": ["Patient_ID", "day1_00:00以降", "単位", "欠損値", "行", "列"],
        "説明": [
            "被検者IDを入力してください。",
            "5分ごとのCGM値を横方向に入力してください。",
            "血糖値は mg/dL を想定しています。",
            "空欄またはNAで構いません。解析時に有効点のみ使用されます。",
            "1行が1被検者です。",
            "1列が1時点です。",
        ],
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        example.to_excel(writer, index=False, sheet_name="CGM_template")
        readme.to_excel(writer, index=False, sheet_name="README")
    return output.getvalue()


def prepare_input(df: pd.DataFrame, id_column: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    if id_column and id_column in df.columns:
        df = df.set_index(id_column)
    elif df.columns[0].lower() in ["patient_id", "id", "patient", "subject", "cgm_id"]:
        df = df.set_index(df.columns[0])
    df.index = df.index.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    return df


def smooth_and_derive(values: np.ndarray, settings: Settings):
    vals = np.asarray(values, dtype=float)
    t_all = np.arange(len(vals), dtype=float) * settings.interval_min
    valid = np.isfinite(vals)
    t_v = t_all[valid]
    x_v = vals[valid]
    if len(x_v) < 10:
        return None
    s_val = len(x_v) * settings.smooth_multiplier
    try:
        spl = UnivariateSpline(t_v, x_v, k=3, s=s_val)
        xs = spl(t_all)
        vel = spl.derivative(1)(t_all)
        acc = spl.derivative(2)(t_all)
    except Exception:
        return None
    return t_all, xs, vel, acc


def kde_density(data: np.ndarray, grid: np.ndarray) -> np.ndarray:
    d = np.asarray(data, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 5 or np.nanstd(d) == 0:
        return np.zeros_like(grid, dtype=float)
    kde = gaussian_kde(d, bw_method="scott")
    dens = np.maximum(kde(grid), 0)
    norm = _trapz(dens, grid)
    if norm <= 0 or not np.isfinite(norm):
        return np.zeros_like(grid, dtype=float)
    return dens / norm


def kde_density_2d(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 10 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.zeros((len(y_grid), len(x_grid)), dtype=float)
    kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(len(y_grid), len(x_grid))
    Z = np.maximum(Z, 0)
    norm = _trapz(_trapz(Z, x_grid, axis=1), y_grid)
    if norm > 0 and np.isfinite(norm):
        Z = Z / norm
    return Z


def density_to_quantile(dens: np.ndarray, grid: np.ndarray, q_probs: np.ndarray) -> np.ndarray:
    dens = np.asarray(dens, dtype=float)
    cdf = np.cumsum(dens) * (grid[1] - grid[0])
    cdf = np.clip(cdf, 0, 1)
    cdf = np.r_[0.0, cdf, 1.0]
    grid2 = np.r_[grid[0], grid, grid[-1]]
    unique_cdf, idx = np.unique(cdf, return_index=True)
    if len(unique_cdf) < 2:
        return np.full_like(q_probs, np.nan, dtype=float)
    f = interp1d(unique_cdf, grid2[idx], kind="linear", bounds_error=False, fill_value=(grid[0], grid[-1]))
    return f(q_probs)


def density_moments(dens: np.ndarray, grid: np.ndarray) -> Tuple[float, float, float, float]:
    mean = float(_trapz(grid * dens, grid))
    var = float(_trapz((grid - mean) ** 2 * dens, grid))
    sd = float(np.sqrt(max(var, 0)))
    if sd == 0 or not np.isfinite(sd):
        return mean, sd, np.nan, np.nan
    skew = float(_trapz(((grid - mean) / sd) ** 3 * dens, grid))
    kurt = float(_trapz(((grid - mean) / sd) ** 4 * dens, grid) - 3)
    return mean, sd, skew, kurt


def area_between(dens: np.ndarray, grid: np.ndarray, lo=None, hi=None) -> float:
    mask = np.ones_like(grid, dtype=bool)
    if lo is not None:
        mask &= grid >= lo
    if hi is not None:
        mask &= grid <= hi
    if mask.sum() < 2:
        return 0.0
    return float(_trapz(dens[mask], grid[mask]) * 100)


def q_at(qf: np.ndarray, q_probs: np.ndarray, p: float) -> float:
    return float(np.interp(p, q_probs, qf))


def analyze_cgm(df_cgm: pd.DataFrame, settings: Settings):
    glucose_grid = np.linspace(settings.gluc_min, settings.gluc_max, settings.n_grid)
    vel_grid = np.linspace(settings.vel_min, settings.vel_max, settings.n_grid)
    acc_grid = np.linspace(settings.acc_min, settings.acc_max, settings.n_grid)
    q_probs = np.linspace(0.001, 0.999, settings.n_quantile)

    store: Dict[str, dict] = {}
    records = []
    skipped = []

    for pid, row in df_cgm.iterrows():
        res = smooth_and_derive(row.values, settings)
        if res is None:
            skipped.append(pid)
            continue
        t, xs, vel, acc = res
        mask_all = (
            np.isfinite(xs) & np.isfinite(vel) & np.isfinite(acc) &
            (xs >= settings.gluc_min) & (xs <= settings.gluc_max) &
            (vel >= settings.vel_min) & (vel <= settings.vel_max) &
            (acc >= settings.acc_min) & (acc <= settings.acc_max)
        )
        xs_joint = xs[mask_all]
        vel_joint = vel[mask_all]
        acc_joint = acc[mask_all]
        g_in = xs_joint
        v_in = vel_joint
        a_in = acc_joint
        if len(g_in) < 10 or len(v_in) < 10 or len(a_in) < 10:
            skipped.append(pid)
            continue
        dens_g = kde_density(g_in, glucose_grid)
        dens_v = kde_density(v_in, vel_grid)
        dens_a = kde_density(a_in, acc_grid)
        qf_g = density_to_quantile(dens_g, glucose_grid, q_probs)
        qf_v = density_to_quantile(dens_v, vel_grid, q_probs)
        qf_a = density_to_quantile(dens_a, acc_grid, q_probs)
        mg, sdg, skg, kug = density_moments(dens_g, glucose_grid)
        mv, sdv, skv, kuv = density_moments(dens_v, vel_grid)
        ma, sda, ska, kua = density_moments(dens_a, acc_grid)
        rec = {
            "Patient_ID": pid,
            "N_valid_pts": int(np.isfinite(row.values).sum()),
            "N_missing_pts": int(np.isnan(row.values.astype(float)).sum()),
            "N_joint_pts": int(mask_all.sum()),
            "Est_days": len(row.values) * settings.interval_min / 1440,
            "G_mean": mg,
            "G_SD": sdg,
            "G_Skew": skg,
            "G_Ex_Kurtosis": kug,
            "TBR_<70_%": area_between(dens_g, glucose_grid, hi=70),
            "TIR_70_180_%": area_between(dens_g, glucose_grid, lo=70, hi=180),
            "TAR_>180_%": area_between(dens_g, glucose_grid, lo=180),
            "G_Q10": q_at(qf_g, q_probs, 0.10),
            "G_Q25": q_at(qf_g, q_probs, 0.25),
            "G_Q50": q_at(qf_g, q_probs, 0.50),
            "G_Q75": q_at(qf_g, q_probs, 0.75),
            "G_Q90": q_at(qf_g, q_probs, 0.90),
            "V_mean": mv,
            "V_SD": sdv,
            "V_Skew": skv,
            "V_Ex_Kurtosis": kuv,
            "V_Q10": q_at(qf_v, q_probs, 0.10),
            "V_Q50": q_at(qf_v, q_probs, 0.50),
            "V_Q90": q_at(qf_v, q_probs, 0.90),
            "V_IQR": q_at(qf_v, q_probs, 0.75) - q_at(qf_v, q_probs, 0.25),
            "Frac_Rise_%": area_between(dens_v, vel_grid, lo=0),
            "Frac_RapidRise_%": area_between(dens_v, vel_grid, lo=settings.rapid_rise_thr),
            "Frac_RapidFall_%": area_between(dens_v, vel_grid, hi=-settings.rapid_rise_thr),
            "A_mean": ma,
            "A_SD": sda,
            "A_Skew": ska,
            "A_Ex_Kurtosis": kua,
            "A_Q10": q_at(qf_a, q_probs, 0.10),
            "A_Q50": q_at(qf_a, q_probs, 0.50),
            "A_Q90": q_at(qf_a, q_probs, 0.90),
            "A_IQR": q_at(qf_a, q_probs, 0.75) - q_at(qf_a, q_probs, 0.25),
            "Frac_Accel_%": area_between(dens_a, acc_grid, lo=0),
            "Frac_RapidAccel_%": area_between(dens_a, acc_grid, lo=settings.rapid_acc_thr),
            "Frac_RapidDecel_%": area_between(dens_a, acc_grid, hi=-settings.rapid_acc_thr),
        }
        store[pid] = {
            "t": t,
            "xs": xs_joint,
            "vel": vel_joint,
            "acc": acc_joint,
            "dens_g": dens_g,
            "dens_v": dens_v,
            "dens_a": dens_a,
            "qf_g": qf_g,
            "qf_v": qf_v,
            "qf_a": qf_a,
        }
        records.append(rec)

    df_metrics = pd.DataFrame(records)
    if not df_metrics.empty:
        df_metrics = df_metrics.set_index("Patient_ID")
    grids = {"glucose_grid": glucose_grid, "vel_grid": vel_grid, "acc_grid": acc_grid, "q_probs": q_probs}
    return df_metrics, store, skipped, grids


def make_excel_bytes(df_metrics, store, grids, settings: Settings) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_metrics.reset_index().to_excel(writer, sheet_name="Metrics_by_Patient", index=False)
        q_probs = grids["q_probs"]
        for key, sheet, prefix in [("qf_g", "Quantile_Glucose", "Q_G"), ("qf_v", "Quantile_Velocity", "Q_V"), ("qf_a", "Quantile_Acceleration", "Q_A")]:
            rows = []
            for pid, d in store.items():
                row = {"Patient_ID": pid}
                for p, val in zip(q_probs, d[key]):
                    row[f"{prefix}_{p:.3f}"] = val
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)
        settings_df = pd.DataFrame({"parameter": list(settings.__dict__.keys()), "value": list(settings.__dict__.values())})
        settings_df.to_excel(writer, sheet_name="Run_Settings", index=False)
    return output.getvalue()


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()


def render_logo_header():
    st.markdown(
        """
        <div style="
            padding:18px 22px;
            border-radius:18px;
            background: linear-gradient(135deg, #163b73 0%, #1f78b4 55%, #33a3dc 100%);
            color:white;
            box-shadow: 0 6px 18px rgba(22,59,115,0.20);
            margin-bottom: 10px;
        ">
            <div style="font-size: 33px; font-weight: 800; letter-spacing: 0.3px;">Multivariate Glucodensity Calculator 2nd</div>
            <div style="font-size: 17px; font-weight: 600; opacity: 0.95; margin-top: 4px;">神戸大学臨床糖尿病グループ</div>
            <div style="font-size: 13px; opacity: 0.9; margin-top: 8px;">利用は自己責任でお願いします。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_patient_marginals(patient_dict, grids, pid: str):
    fig = plt.figure(figsize=(16, 4.5), facecolor="white")
    axes = fig.subplots(1, 3)
    dens_info = [
        ("dens_g", grids["glucose_grid"], "Glucose density", "Glucose (mg/dL)", "#2C7FB8"),
        ("dens_v", grids["vel_grid"], "Velocity density", "Velocity (mg/dL/min)", "#F28E2B"),
        ("dens_a", grids["acc_grid"], "Acceleration density", "Acceleration (mg/dL/min²)", "#59A14F"),
    ]
    for ax, (key, grid, title, xlabel, color) in zip(axes, dens_info):
        ax.plot(grid, patient_dict[key], linewidth=2.2, color=color)
        ax.fill_between(grid, patient_dict[key], color=color, alpha=0.18)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.15)
    fig.suptitle(f"Patient-specific marginal densities: {pid}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_patient_quantiles(patient_dict, grids, pid: str):
    fig = plt.figure(figsize=(16, 4.5), facecolor="white")
    axes = fig.subplots(1, 3)
    q_info = [
        ("qf_g", "Glucose quantile function", "Glucose (mg/dL)", "#2C7FB8"),
        ("qf_v", "Velocity quantile function", "Velocity (mg/dL/min)", "#F28E2B"),
        ("qf_a", "Acceleration quantile function", "Acceleration (mg/dL/min²)", "#59A14F"),
    ]
    q_probs = grids["q_probs"]
    for ax, (key, title, ylabel, color) in zip(axes, q_info):
        ax.plot(q_probs, patient_dict[key], linewidth=2.2, color=color)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Quantile level")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.15)
    fig.suptitle(f"Patient-specific quantile functions: {pid}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_patient_joint_overlay(patient_dict, settings: Settings, pid: str):
    g, v, a = patient_dict["xs"], patient_dict["vel"], patient_dict["acc"]
    g_grid = np.linspace(settings.gluc_min, settings.gluc_max, settings.joint_n_grid)
    v_grid = np.linspace(settings.vel_min, settings.vel_max, settings.joint_n_grid)
    a_grid = np.linspace(settings.acc_min, settings.acc_max, settings.joint_n_grid)

    pairs = [
        (g, v, g_grid, v_grid, "Glucose concentration", "Velocity", "Marginal density of glucose concentration × velocity"),
        (g, a, g_grid, a_grid, "Glucose concentration", "Acceleration", "Marginal density of glucose concentration × acceleration"),
        (v, a, v_grid, a_grid, "Velocity", "Acceleration", "Marginal density of velocity × acceleration"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7), facecolor="white")
    contour_cmap = "viridis"

    for ax, (x, y, xg, yg, xl, yl, title) in zip(axes, pairs):
        Z = kde_density_2d(x, y, xg, yg)
        X, Y = np.meshgrid(xg, yg)
        cf = ax.contourf(X, Y, Z, levels=18, cmap=contour_cmap)
        ax.contour(X, Y, Z, levels=9, colors="white", linewidths=0.55, alpha=0.45)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.06)
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Existence probability density", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"Patient-specific pairwise density contour plots: {pid}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def make_single_patient_pdf(pid: str, patient_dict, grids, settings: Settings) -> bytes:
    output = io.BytesIO()
    with PdfPages(output) as pdf:
        fig1 = plot_patient_marginals(patient_dict, grids, pid)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)
        fig2 = plot_patient_quantiles(patient_dict, grids, pid)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)
        fig3 = plot_patient_joint_overlay(patient_dict, settings, pid)
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)
    output.seek(0)
    return output.getvalue()


def make_all_patients_pdf(store, grids, settings: Settings) -> bytes:
    output = io.BytesIO()
    with PdfPages(output) as pdf:
        for pid, patient_dict in store.items():
            fig1 = plot_patient_marginals(patient_dict, grids, pid)
            pdf.savefig(fig1, bbox_inches="tight")
            plt.close(fig1)

            fig2 = plot_patient_quantiles(patient_dict, grids, pid)
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

            fig3 = plot_patient_joint_overlay(patient_dict, settings, pid)
            pdf.savefig(fig3, bbox_inches="tight")
            plt.close(fig3)
    output.seek(0)
    return output.getvalue()


def make_all_patients_zip(store, grids, settings: Settings) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pid, patient_dict in store.items():
            fig1 = plot_patient_marginals(patient_dict, grids, pid)
            zf.writestr(f"{pid}/{pid}_marginal_densities.png", fig_to_png_bytes(fig1))
            plt.close(fig1)

            fig2 = plot_patient_quantiles(patient_dict, grids, pid)
            zf.writestr(f"{pid}/{pid}_quantile_functions.png", fig_to_png_bytes(fig2))
            plt.close(fig2)

            fig3 = plot_patient_joint_overlay(patient_dict, settings, pid)
            zf.writestr(f"{pid}/{pid}_pairwise_density_contours.png", fig_to_png_bytes(fig3))
            plt.close(fig3)

            pdf_bytes = make_single_patient_pdf(pid, patient_dict, grids, settings)
            zf.writestr(f"{pid}/{pid}_all_figures.pdf", pdf_bytes)

        readme = (
            "This ZIP contains patient-specific figures generated by Multivariate Glucodensity Calculator 2nd.\n"
            "Each patient folder includes marginal densities, quantile functions, pairwise density contour + scatter overlay, and a PDF report.\n"
            "Kobe University Clinical Diabetes Group / 神戸大学臨床糖尿病グループ\n"
            "Use at your own risk / 利用は自己責任でお願いします。\n"
        )
        zf.writestr("README.txt", readme)
    buffer.seek(0)
    return buffer.getvalue()


def main():
    st.set_page_config(page_title="Multivariate Glucodensity Calculator 2nd", layout="wide")
    render_logo_header()
    st.caption("CGM時系列から血糖分布、速度分布、加速度分布の指標・分位関数・2変量密度等高線図を計算します。")

    with st.expander("この計算機の位置づけ", expanded=False):
        st.markdown(
            """
            このアプリは Matabuena らの multivariate glucodensity の概念に基づく **実用的な特徴量抽出ツール** です。  
            CGM時系列を平滑化し、血糖値 G(t)、速度 dG/dt、加速度 d²G/dt² の密度・分位関数・要約指標、および  
            **周辺2変量密度（カラー等高線）** を出力します。  

            論文の Model 1〜6、long-term HbA1c/FPG予測、mgcv::gam による scalar-on-distribution regression そのものは含みません。
            """
        )

    st.sidebar.title("Multivariate Glucodensity Calculator 2nd")
    st.sidebar.markdown("**神戸大学臨床糖尿病グループ**")
    st.sidebar.caption("利用は自己責任でお願いします。")
    st.sidebar.header("入力データ")

    template_excel = make_template_excel()
    st.sidebar.download_button(
        label="入力テンプレートExcelをダウンロード",
        data=template_excel,
        file_name="glucodensity_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    uploaded = st.sidebar.file_uploader("Excel または CSV をアップロード", type=["xlsx", "xls", "csv"])
    use_demo = st.sidebar.checkbox("デモデータを使う", value=(uploaded is None))

    if uploaded is not None and not use_demo:
        if uploaded.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded)
        else:
            xls = pd.ExcelFile(uploaded)
            sheet_name = st.sidebar.selectbox("シート", xls.sheet_names)
            df_in = pd.read_excel(uploaded, sheet_name=sheet_name)
        id_column = st.sidebar.selectbox(
            "患者ID列",
            options=["先頭列または既存indexを使用しない"] + list(df_in.columns),
            index=1 if len(df_in.columns) > 0 else 0,
        )
        if id_column == "先頭列または既存indexを使用しない":
            id_column = None
        df_cgm = prepare_input(df_in, id_column)
    else:
        df_cgm = make_demo_data()
        df_cgm = prepare_input(df_cgm.reset_index(), "Patient_ID")

    st.sidebar.header("解析パラメータ")
    interval_min = st.sidebar.number_input("CGM間隔（分）", min_value=1.0, max_value=60.0, value=5.0, step=1.0)
    smooth_multiplier = st.sidebar.slider("平滑化強度：s = 有効点数 × 倍率", 0.1, 10.0, 2.0, 0.1)
    n_grid = st.sidebar.slider("1D KDEグリッド数", 100, 1000, 500, 50)
    n_quantile = st.sidebar.slider("分位関数点数", 20, 200, 100, 10)
    joint_n_grid = st.sidebar.slider("2D等高線グリッド数", 40, 200, 120, 10)

    with st.sidebar.expander("範囲と閾値"):
        gluc_min, gluc_max = st.slider("Glucose range", 40, 400, (40, 400))
        vel_min, vel_max = st.slider("Velocity range", -20.0, 20.0, (-5.0, 5.0), 0.5)
        acc_min, acc_max = st.slider("Acceleration range", -10.0, 10.0, (-3.0, 3.0), 0.5)
        rapid_rise_thr = st.number_input("急速上昇閾値：dG/dt >", value=1.0, step=0.1)
        rapid_acc_thr = st.number_input("急加速閾値：d²G/dt² >", value=0.5, step=0.1)

    settings = Settings(
        interval_min=interval_min,
        smooth_multiplier=smooth_multiplier,
        n_grid=n_grid,
        n_quantile=n_quantile,
        joint_n_grid=joint_n_grid,
        gluc_min=float(gluc_min),
        gluc_max=float(gluc_max),
        vel_min=float(vel_min),
        vel_max=float(vel_max),
        acc_min=float(acc_min),
        acc_max=float(acc_max),
        rapid_rise_thr=rapid_rise_thr,
        rapid_acc_thr=rapid_acc_thr,
    )

    st.subheader("入力データ確認")
    st.write(f"被検者数: **{df_cgm.shape[0]}**、時点数: **{df_cgm.shape[1]}**")
    st.dataframe(df_cgm.iloc[:10, :10], use_container_width=True)

    if st.button("解析を実行", type="primary"):
        with st.spinner("解析中です..."):
            df_metrics, store, skipped, grids = analyze_cgm(df_cgm, settings)

        if df_metrics.empty:
            st.error("有効な被検者がありませんでした。入力形式、欠損、平滑化条件、範囲設定を確認してください。")
            if skipped:
                st.write("スキップされたID:", skipped)
            return

        st.success(f"解析完了: {len(store)} 名を解析しました。")
        if skipped:
            st.warning(f"{len(skipped)} 名はスキップされました: {', '.join(map(str, skipped[:20]))}")

        st.subheader("指標テーブル")
        st.dataframe(df_metrics.reset_index(), use_container_width=True)

        st.subheader("患者別の詳細表示")
        selected_pid = st.selectbox("表示する患者を選択", list(store.keys()))
        patient_dict = store[selected_pid]

        fig_marginals = plot_patient_marginals(patient_dict, grids, selected_pid)
        fig_quantiles = plot_patient_quantiles(patient_dict, grids, selected_pid)
        fig_joint = plot_patient_joint_overlay(patient_dict, settings, selected_pid)

        tab1, tab2, tab3 = st.tabs(["周辺密度", "分位関数", "2変量密度等高線"])
        with tab1:
            st.pyplot(fig_marginals)
        with tab2:
            st.pyplot(fig_quantiles)
        with tab3:
            st.caption("各パネルは存在確率のカラー等高線に、実データのscatterを重ね描きしています。各パネルは2変量周辺密度関数の存在確率を、カラーの等高線で表示しています。")
            st.pyplot(fig_joint)

        png_marginals = fig_to_png_bytes(fig_marginals)
        png_quantiles = fig_to_png_bytes(fig_quantiles)
        png_joint = fig_to_png_bytes(fig_joint)
        plt.close(fig_marginals)
        plt.close(fig_quantiles)
        plt.close(fig_joint)

        st.subheader("保存")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("周辺密度PNGを保存", data=png_marginals, file_name=f"{selected_pid}_marginal_densities.png", mime="image/png")
        with c2:
            st.download_button("分位関数PNGを保存", data=png_quantiles, file_name=f"{selected_pid}_quantile_functions.png", mime="image/png")
        with c3:
            st.download_button("2変量密度等高線PNGを保存", data=png_joint, file_name=f"{selected_pid}_pairwise_density_contours.png", mime="image/png")

        pdf_bytes = make_single_patient_pdf(selected_pid, patient_dict, grids, settings)
        all_patients_pdf_bytes = make_all_patients_pdf(store, grids, settings)
        all_zip_bytes = make_all_patients_zip(store, grids, settings)
        excel_bytes = make_excel_bytes(df_metrics, store, grids, settings)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "選択患者の全図PDFを保存",
                data=pdf_bytes,
                file_name=f"{selected_pid}_all_figures.pdf",
                mime="application/pdf",
                key=f"single_pdf_{selected_pid}",
            )
        with d2:
            st.download_button(
                "解析対象者全員分の図をPDFで一括ダウンロード",
                data=all_patients_pdf_bytes,
                file_name="all_patients_glucodensity_figures.pdf",
                mime="application/pdf",
                key="all_patients_pdf",
            )

        d3, d4 = st.columns(2)
        with d3:
            st.download_button(
                "全患者の全図をZIP保存",
                data=all_zip_bytes,
                file_name="all_patients_glucodensity_figures.zip",
                mime="application/zip",
                key="all_patients_zip",
            )
        with d4:
            st.download_button(
                "Excel結果をダウンロード",
                data=excel_bytes,
                file_name="multivariate_glucodensity_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download",
            )


if __name__ == "__main__":
    main()
