
import io
import warnings
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


@dataclass
class Settings:
    interval_min: float = 5.0
    smooth_multiplier: float = 2.0
    n_grid: int = 500
    n_quantile: int = 100
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
            patients[f"{name}_{k:02d}"] = sim(mu + np.random.uniform(-5, 5),
                                               sd + np.random.uniform(-3, 3),
                                               sp)
            k += 1

    cols = [f"t{i:04d}" for i in range(n_pts)]
    df = pd.DataFrame.from_dict(patients, orient="index", columns=cols)
    df.index.name = "Patient_ID"
    return df


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

    # cubic smoothing spline
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


def density_to_quantile(dens: np.ndarray, grid: np.ndarray, q_probs: np.ndarray) -> np.ndarray:
    dens = np.asarray(dens, dtype=float)
    cdf = np.cumsum(dens) * (grid[1] - grid[0])
    cdf = np.clip(cdf, 0, 1)

    # ensure interpolation endpoints
    cdf = np.r_[0.0, cdf, 1.0]
    grid2 = np.r_[grid[0], grid, grid[-1]]
    unique_cdf, idx = np.unique(cdf, return_index=True)

    if len(unique_cdf) < 2:
        return np.full_like(q_probs, np.nan, dtype=float)

    f = interp1d(unique_cdf, grid2[idx], kind="linear",
                 bounds_error=False, fill_value=(grid[0], grid[-1]))
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

        g_in = xs[(xs >= settings.gluc_min) & (xs <= settings.gluc_max)]
        v_in = vel[(vel >= settings.vel_min) & (vel <= settings.vel_max)]
        a_in = acc[(acc >= settings.acc_min) & (acc <= settings.acc_max)]

        if len(g_in) < 5 or len(v_in) < 5 or len(a_in) < 5:
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
            "t": t, "xs": xs, "vel": vel, "acc": acc,
            "dens_g": dens_g, "dens_v": dens_v, "dens_a": dens_a,
            "qf_g": qf_g, "qf_v": qf_v, "qf_a": qf_a,
        }
        records.append(rec)

    df_metrics = pd.DataFrame(records)
    if not df_metrics.empty:
        df_metrics = df_metrics.set_index("Patient_ID")

    grids = {
        "glucose_grid": glucose_grid,
        "vel_grid": vel_grid,
        "acc_grid": acc_grid,
        "q_probs": q_probs,
    }
    return df_metrics, store, skipped, grids


def make_excel_bytes(df_metrics, store, grids, settings: Settings) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_metrics.reset_index().to_excel(writer, sheet_name="Metrics_by_Patient", index=False)

        q_probs = grids["q_probs"]
        for key, sheet, prefix in [
            ("qf_g", "Quantile_Glucose", "Q_G"),
            ("qf_v", "Quantile_Velocity", "Q_V"),
            ("qf_a", "Quantile_Acceleration", "Q_A"),
        ]:
            rows = []
            for pid, d in store.items():
                row = {"Patient_ID": pid}
                for p, val in zip(q_probs, d[key]):
                    row[f"{prefix}_{p:.3f}"] = val
                rows.append(row)
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)

        settings_df = pd.DataFrame({
            "parameter": list(settings.__dict__.keys()),
            "value": list(settings.__dict__.values())
        })
        settings_df.to_excel(writer, sheet_name="Run_Settings", index=False)

    return output.getvalue()


def make_pdf_bytes(store, grids) -> bytes:
    output = io.BytesIO()
    valid_pids = list(store.keys())
    if len(valid_pids) == 0:
        return b""

    with PdfPages(output) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle("Multivariate glucodensity: density profiles", fontsize=14, fontweight="bold")
        panels = [
            ("Glucose G(t)", grids["glucose_grid"], "dens_g", "Density"),
            ("Velocity dG/dt", grids["vel_grid"], "dens_v", "Density"),
            ("Acceleration d2G/dt2", grids["acc_grid"], "dens_a", "Density"),
        ]
        for ax, (title, grid, key, ylabel) in zip(axes, panels):
            for pid in valid_pids:
                ax.plot(grid, store[pid][key], alpha=0.45, linewidth=1)
            mean_d = np.mean([store[pid][key] for pid in valid_pids], axis=0)
            ax.plot(grid, mean_d, color="black", linestyle="--", linewidth=2.5, label="Mean")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle("Quantile functions", fontsize=14, fontweight="bold")
        q_panels = [
            ("Glucose quantiles", "qf_g", "Glucose (mg/dL)"),
            ("Velocity quantiles", "qf_v", "Velocity (mg/dL/min)"),
            ("Acceleration quantiles", "qf_a", "Acceleration (mg/dL/min2)"),
        ]
        q_probs = grids["q_probs"]
        for ax, (title, key, ylabel) in zip(axes, q_panels):
            for pid in valid_pids:
                ax.plot(q_probs, store[pid][key], alpha=0.45, linewidth=1)
            mean_q = np.nanmean([store[pid][key] for pid in valid_pids], axis=0)
            ax.plot(q_probs, mean_q, color="black", linestyle="--", linewidth=2.5, label="Mean")
            ax.set_title(title)
            ax.set_xlabel("Quantile level")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return output.getvalue()


def main():
    st.set_page_config(page_title="Multivariate Glucodensity Calculator", layout="wide")
    st.title("Multivariate Glucodensity Calculator")
    st.caption("CGM時系列から血糖分布、速度分布、加速度分布の指標と分位関数を計算します。")

    with st.expander("この計算機の位置づけ", expanded=False):
        st.markdown(
            """
            このアプリは Matabuena らの multivariate glucodensity の概念に基づく
            **実用的な特徴量抽出ツール**です。CGM時系列を平滑化し、血糖値 G(t)、
            速度 dG/dt、加速度 d²G/dt² の密度・分位関数・要約指標を出力します。

            論文の Model 1〜6、long-term HbA1c/FPG予測、mgcv::gam による
            scalar-on-distribution regression そのものは含みません。
            """
        )

    st.sidebar.header("入力データ")
    uploaded = st.sidebar.file_uploader("Excel または CSV をアップロード", type=["xlsx", "xls", "csv"])
    use_demo = st.sidebar.checkbox("デモデータを使う", value=(uploaded is None))

    sheet_name = None
    id_column = None

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
            index=1 if len(df_in.columns) > 0 else 0
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
    n_grid = st.sidebar.slider("KDEグリッド数", 100, 1000, 500, 50)
    n_quantile = st.sidebar.slider("分位関数点数", 20, 200, 100, 10)

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

    run = st.button("解析を実行", type="primary")

    if run:
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

        st.subheader("平均密度プロファイル")
        valid_pids = list(store.keys())
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        panels = [
            ("Glucose G(t)", grids["glucose_grid"], "dens_g"),
            ("Velocity dG/dt", grids["vel_grid"], "dens_v"),
            ("Acceleration d2G/dt2", grids["acc_grid"], "dens_a"),
        ]
        for ax, (title, grid, key) in zip(axes, panels):
            for pid in valid_pids:
                ax.plot(grid, store[pid][key], alpha=0.25, linewidth=1)
            mean_d = np.mean([store[pid][key] for pid in valid_pids], axis=0)
            ax.plot(grid, mean_d, color="black", linestyle="--", linewidth=2.5)
            ax.set_title(title)
            ax.set_ylabel("Density")
        plt.tight_layout()
        st.pyplot(fig)

        excel_bytes = make_excel_bytes(df_metrics, store, grids, settings)
        pdf_bytes = make_pdf_bytes(store, grids)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Excel結果をダウンロード",
                data=excel_bytes,
                file_name="multivariate_glucodensity_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with col2:
            st.download_button(
                "PDF図をダウンロード",
                data=pdf_bytes,
                file_name="multivariate_glucodensity_plots.pdf",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
