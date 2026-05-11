"""
Streamlit crypto price viewer and LSTM forecast (logic aligned with Crypto_Currency_Price_Prediction.ipynb).
Loads per-ticker or default `btc_model.h5` / `scaler.pkl`; fetches live data with yfinance.
"""

from __future__ import annotations

import io
import pickle
from contextlib import nullcontext
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.figure import Figure
from tensorflow import keras
from tensorflow.keras.models import load_model

try:
    import tensorflow as tf
except ImportError:
    tf = None

BASE_DIR = Path(__file__).resolve().parent
CURRENCY_OPTIONS = ["BTC-USD", "ETH-USD", "SOL-USD"]
VIEW_OPTIONS = ["Current Status", "Forecast"]
YAHOO_PERIODS = ["1y", "3y", "5y", "max"]
PRICE_MODES = ["Adj Close", "Close"]
CHART_THEMES = ["Light", "Dark"]
DEFAULT_HISTORY_TAIL = 500
FORECAST_MIN, FORECAST_MAX = 7, 60


def resolve_artifact_paths(symbol: str) -> tuple[Path, Path, str]:
    """
    Prefer `{ticker}_model.h5` and `{ticker}_scaler.pkl` (e.g. eth_model.h5).
    Fallback to btc_model.h5 / scaler.pkl when missing.
    """
    suffix = symbol.split("-")[0].lower()
    model_path = BASE_DIR / f"{suffix}_model.h5"
    scaler_path = BASE_DIR / f"{suffix}_scaler.pkl"
    note_parts = []

    fallback_model = BASE_DIR / "btc_model.h5"
    fallback_scaler = BASE_DIR / "scaler.pkl"

    if not model_path.is_file():
        if fallback_model.is_file():
            note_parts.append(f"Model `{model_path.name}` missing — using `{fallback_model.name}`")
            model_path = fallback_model
        else:
            note_parts.append(f"No model found for `{suffix}` or default `btc_model.h5`.")
    if not scaler_path.is_file():
        if fallback_scaler.is_file():
            note_parts.append(f"Scaler `{scaler_path.name}` missing — using `{fallback_scaler.name}`")
            scaler_path = fallback_scaler
        else:
            note_parts.append(f"No scaler found for `{suffix}` or default `scaler.pkl`.")

    if model_path.is_file() and scaler_path.is_file():
        mn, sn = model_path.name.lower(), scaler_path.name.lower()
        if mn.endswith("_model.h5") and sn.endswith("_scaler.pkl"):
            m_pre, s_pre = mn[: -len("_model.h5")], sn[: -len("_scaler.pkl")]
            if m_pre != s_pre:
                note_parts.append(
                    f"⚠️ **Naming mismatch:** `{model_path.name}` vs `{scaler_path.name}` — "
                    "use a scaler trained with that model or forecasts may be meaningless."
                )
        elif mn.endswith("_model.h5") and sn.endswith(".pkl") and not sn.endswith("_scaler.pkl"):
            note_parts.append(
                f"⚠️ **Generic scaler file** `{scaler_path.name}` with `{model_path.name}` — "
                "confirm it is the MinMaxScaler paired with this model."
            )

    return model_path, scaler_path, " ".join(note_parts)


def _file_mtime_ns(p: Path) -> int:
    return int(p.stat().st_mtime_ns) if p.is_file() else -1


def load_fitted_minmax(path: Path):
    """Load sklearn scaler saved with joblib or pickle."""
    errors = []
    for name, loader in (
        ("joblib", lambda: joblib.load(path)),
        ("pickle", lambda p=path: pickle.loads(p.read_bytes())),
    ):
        try:
            obj = loader()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        for attr in ("transform", "inverse_transform"):
            if not hasattr(obj, attr):
                errors.append(f"{name}: loaded object has no `{attr}`")
                break
        else:
            return obj
    raise RuntimeError(
        "Could not load the scaler file. Re-save from the notebook in this Python environment, e.g.\n"
        "`pickle.dump(scaler, open('scaler.pkl','wb'), protocol=4)` or `joblib.dump(sc, 'scaler.pkl')`.\n"
        f"Details: {'; '.join(errors)}"
    )


@st.cache_resource
def load_lstm_assets(model_path_str: str, scaler_path_str: str, model_mtime_ns: int, scaler_mtime_ns: int):
    """mtime args bump the cache key when checkpoints on disk change."""
    _ = model_mtime_ns, scaler_mtime_ns
    model_path = Path(model_path_str)
    scaler_path = Path(scaler_path_str)
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
    model = load_model(model_path, compile=False)
    scaler = load_fitted_minmax(scaler_path)
    return model, scaler


def infer_time_steps(model) -> int:
    shape = getattr(model, "input_shape", None)
    if shape and len(shape) >= 2 and shape[1] is not None:
        return max(1, int(shape[1]))
    return 5


def file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return "(missing)"
    st_ = path.stat()
    modified = datetime.fromtimestamp(st_.st_mtime).strftime("%Y-%m-%d %H:%M")
    size_kb = st_.st_size / 1024
    return f"{size_kb:.1f} KB, modified {modified}"


def fetch_price_history(
    symbol: str,
    period: str | None,
    range_start: date | None,
    range_end: date | None,
) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    if range_start is not None and range_end is not None:
        if range_start >= range_end:
            raise ValueError("Start date must be before end date.")
        df = ticker.history(
            start=pd.Timestamp(range_start),
            end=pd.Timestamp(range_end) + pd.Timedelta(days=1),
            auto_adjust=False,
        )
    else:
        df = ticker.history(period=period or "5y", auto_adjust=False)
    if df.empty:
        raise ValueError(f"No price data returned for {symbol}")
    return df.sort_index()


def _hashable_optional(d: date | None) -> str | None:
    return None if d is None else d.isoformat()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_history_cached(
    symbol: str,
    period: str,
    range_start_s: str | None,
    range_end_s: str | None,
) -> pd.DataFrame:
    rs = date.fromisoformat(range_start_s) if range_start_s else None
    re_ = date.fromisoformat(range_end_s) if range_end_s else None
    return fetch_price_history(symbol, period if not (rs and re_) else None, rs, re_)


def compute_daily_change(series: pd.Series) -> tuple[float, float]:
    s = series.dropna()
    if len(s) == 0:
        raise ValueError("Price series has no non-null values.")
    if len(s) < 2:
        return float(s.iloc[-1]), 0.0
    current = float(s.iloc[-1])
    prev = float(s.iloc[-2])
    pct = (current - prev) / prev * 100 if prev != 0 else 0.0
    return current, pct


def recursive_forecast(
    closes: np.ndarray,
    model,
    scaler,
    time_steps: int,
    horizon: int,
) -> np.ndarray:
    closes = np.asarray(closes, dtype=np.float64).reshape(-1, 1)
    if len(closes) < time_steps:
        raise ValueError(f"Need at least {time_steps} observations for forecasting.")

    last_block = scaler.transform(closes[-time_steps:])
    window = last_block.reshape(1, time_steps, 1).astype(np.float32)

    future_scaled: list[float] = []
    for _ in range(horizon):
        nxt = model.predict(window, verbose=0)
        future_scaled.append(float(nxt[0, 0]))
        nxt_reshaped = np.array([[nxt[0, 0]]], dtype=np.float32).reshape(1, 1, 1)
        window = np.append(window[:, 1:, :], nxt_reshaped, axis=1)

    return scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()


def _theme_colors(theme: str) -> dict:
    if theme == "Dark":
        return {
            "fig": "#1e1e1e",
            "ax": "#2d2d2d",
            "text": "#e8e8e8",
            "grid": "#444444",
            "hist": "#5fd068",
            "future": "#ff6b6b",
            "naive": "#ffd93d",
            "vol": "#74b9ff",
            "vline": "#888888",
        }
    return {
        "fig": "#ffffff",
        "ax": "#fafafa",
        "text": "#222222",
        "grid": "#cccccc",
        "hist": "#2ca02c",
        "future": "#d62728",
        "naive": "#ff7f0e",
        "vol": "#1f77b4",
        "vline": "#999999",
    }


def build_price_figure(
    close_series: pd.Series,
    volume_series: pd.Series | None,
    future_values: np.ndarray | None,
    symbol: str,
    history_tail: int,
    *,
    log_scale: bool,
    theme: str,
    show_volume: bool,
    naive_baseline: bool,
    price_label: str,
) -> Figure:
    s = close_series.dropna()
    hist = s.iloc[-history_tail:] if len(s) > history_tail else s
    vol = None
    if show_volume and volume_series is not None:
        v = volume_series.reindex(hist.index).fillna(0)
        vol = v

    colors = _theme_colors(theme)
    style_mgr = plt.style.context("dark_background") if theme == "Dark" else nullcontext()

    with style_mgr:
        if show_volume and vol is not None:
            fig, (ax_price, ax_vol) = plt.subplots(
                2,
                1,
                figsize=(12, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            )
        else:
            fig, ax_price = plt.subplots(figsize=(12, 6))
            ax_vol = None

        fig.patch.set_facecolor(colors["fig"])
        ax_price.set_facecolor(colors["ax"])
        if ax_vol is not None:
            ax_vol.set_facecolor(colors["ax"])

        x_hist = np.arange(len(hist))
        ax_price.plot(
            x_hist,
            hist.values,
            color=colors["hist"],
            linewidth=1.6,
            label=f"Historical ({price_label})",
        )

        if future_values is not None and len(future_values):
            offset = len(x_hist)
            x_future = np.arange(offset, offset + len(future_values))
            ax_price.plot(
                x_future,
                future_values,
                color=colors["future"],
                linewidth=1.8,
                label=f"LSTM forecast ({len(future_values)} days)",
            )
            if naive_baseline:
                last = float(hist.iloc[-1])
                ax_price.plot(
                    x_future,
                    np.full(len(future_values), last),
                    color=colors["naive"],
                    linestyle="--",
                    linewidth=1.2,
                    label="Naive baseline (flat last close)",
                )
            ax_price.axvline(x=offset - 0.5, color=colors["vline"], linestyle=":", linewidth=0.9)

        if log_scale:
            ax_price.set_yscale("log")

        ax_price.set_title(f"{symbol} — price & forecast", fontsize=15, color=colors["text"], pad=12)
        ax_price.set_ylabel("Price (USD)", color=colors["text"])
        ax_price.tick_params(colors=colors["text"])
        ax_price.legend(loc="upper left", framealpha=0.92, labelcolor=colors["text"])
        ax_price.grid(True, alpha=0.35, color=colors["grid"])

        if ax_vol is not None and vol is not None:
            ax_vol.bar(x_hist, vol.values, color=colors["vol"], alpha=0.65, width=1.0)
            ax_vol.set_ylabel("Volume", color=colors["text"], fontsize=9)
            ax_vol.tick_params(colors=colors["text"])
            ax_vol.grid(True, axis="y", alpha=0.25, color=colors["grid"])
            ax_vol.set_xlabel("Trading days (recent window)", color=colors["text"])
        else:
            ax_price.set_xlabel("Trading days (recent window)", color=colors["text"])

        fig.tight_layout()

    return fig


def figure_to_png_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()


def render_dashboard(
    *,
    symbol: str,
    forecast_view: str,
    yahoo_period: str,
    use_custom_range: bool,
    range_start: date | None,
    range_end: date | None,
    price_mode: str,
    history_span: int,
    forecast_horizon: int,
    chart_theme: str,
    log_scale: bool,
    show_volume: bool,
    naive_baseline: bool,
    table_expanded: bool,
    show_interactive_chart: bool,
    show_debug_panel: bool,
    model_path: Path,
    scaler_path: Path,
    artifact_note: str,
) -> None:
    try:
        model, scaler = load_lstm_assets(
            str(model_path),
            str(scaler_path),
            _file_mtime_ns(model_path),
            _file_mtime_ns(scaler_path),
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.exception(exc)
        return

    expected_steps = infer_time_steps(model)

    with st.spinner("Fetching market data …"):
        try:
            hist = fetch_price_history_cached(
                symbol,
                yahoo_period,
                _hashable_optional(range_start) if use_custom_range else None,
                _hashable_optional(range_end) if use_custom_range else None,
            )
        except Exception as exc:
            st.error(f"Could not download data for **{symbol}**: {exc}")
            return

    # yfinance occasionally returns MultiIndex columns (feature, ticker)
    if isinstance(hist.columns, pd.MultiIndex) and hist.columns.nlevels >= 2:
        try:
            hist = hist.droplevel(1, axis=1).copy()
        except (ValueError, IndexError):
            hist = hist.copy()
            hist.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in hist.columns]

    if price_mode == "Adj Close" and "Adj Close" in hist.columns:
        close_col = hist["Adj Close"]
    else:
        close_col = hist["Close"] if "Close" in hist.columns else hist["Adj Close"]

    vol_col = hist["Volume"] if "Volume" in hist.columns else None

    if len(close_col.dropna()) == 0:
        st.error(
            "**No usable price bars** for this range and price mode. Broaden dates or verify the ticker loads in Yahoo Finance."
        )
        return

    try:
        current, pct_chg = compute_daily_change(close_col)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.title("Cryptocurrency price dashboard")
    st.caption("Data: Yahoo Finance · LSTM inference only (not financial advice).")

    m1, m2, m3 = st.columns([2, 1, 1])
    with m1:
        arrow = "+" if pct_chg >= 0 else ""
        st.metric(
            label=f"{symbol} — last {price_mode.lower()}",
            value=f"${current:,.2f}",
            delta=f"{arrow}{pct_chg:.2f}% vs prior session",
            delta_color="normal" if pct_chg >= 0 else "inverse",
        )
    with m2:
        st.metric("Bars loaded", len(close_col.dropna()))
    with m3:
        st.metric("LSTM window (model)", f"{expected_steps} days")

    if len(close_col.dropna()) < expected_steps:
        st.warning(
            f"Only {len(close_col.dropna())} rows — need at least **{expected_steps}** for this model’s input window."
        )

    if artifact_note:
        st.info(artifact_note)

    if symbol != "BTC-USD" and model_path.stem.lower() == "btc_model":
        st.warning(
            "**Disclaimer:** Default BTC checkpoint is in use. Train and add `eth_model.h5` / `eth_scaler.pkl` "
            "(and similarly for SOL) for pair-matched forecasts."
        )

    show_forecast = forecast_view == "Forecast"
    future_arr: np.ndarray | None = None
    if show_forecast:
        try:
            future_arr = recursive_forecast(
                close_col.values.astype(float),
                model,
                scaler,
                time_steps=expected_steps,
                horizon=forecast_horizon,
            )
        except Exception as exc:
            st.warning(f"Forecast failed: {exc}")
            future_arr = None

    log_eff = log_scale
    if log_scale:
        hist_tail_np = close_col.dropna().iloc[-history_span:].to_numpy(dtype=float, copy=False)
        if hist_tail_np.size == 0 or np.nanmin(hist_tail_np) <= 0:
            st.warning("Log scale disabled: historical window contains non-positive values.")
            log_eff = False
        elif future_arr is not None and len(future_arr):
            fu = np.asarray(future_arr, dtype=float)
            if np.nanmin(fu) <= 0:
                st.warning("Log scale disabled: forecast contains non-positive values.")
                log_eff = False

    st.divider()
    st.subheader("Price chart")

    fig = build_price_figure(
        close_col,
        vol_col,
        future_arr if show_forecast else None,
        symbol,
        history_span,
        log_scale=log_eff,
        theme=chart_theme,
        show_volume=show_volume,
        naive_baseline=naive_baseline and show_forecast and future_arr is not None,
        price_label=price_mode,
    )
    st.pyplot(fig)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="Download chart (PNG)",
            data=figure_to_png_bytes(fig),
            file_name=f"{symbol.replace('-', '_')}_chart.png",
            mime="image/png",
            use_container_width=True,
            key="download_chart_png",
        )

    plt.close(fig)

    if show_forecast and future_arr is not None:
        fwd = pd.DataFrame(
            {
                "forecast_day_index": np.arange(1, len(future_arr) + 1),
                "predicted_price_usd": future_arr.round(4),
            }
        )
        csv_bytes = fwd.to_csv(index=False).encode("utf-8")
        with col_dl2:
            st.download_button(
                label="Download forecast (CSV)",
                data=csv_bytes,
                file_name=f"{symbol.replace('-', '_')}_forecast.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_forecast_csv",
            )
        expander_kw = {"expanded": table_expanded}
        with st.expander("Forecast table", **expander_kw):
            st.dataframe(fwd, use_container_width=True, hide_index=True)
    elif show_forecast:
        st.markdown("Forecast unavailable — adjust inputs or artifacts.")

    if show_interactive_chart:
        st.subheader("Interactive chart")
        st.caption("Native Streamlit chart (pan/zoom). Same price window as the slider above.")
        df_line = pd.DataFrame({price_mode: close_col}).iloc[-history_span:]
        st.line_chart(df_line, height=320)

    st.caption("Demo / research only — not financial advice.")

    if show_debug_panel:
        st.divider()
        with st.expander("Technical: model files & library versions", expanded=False):
            st.markdown(
                f"| File | Detail |\n|------|--------|\n| `{model_path.name}` | {file_fingerprint(model_path)} |\n"
                f"| `{scaler_path.name}` | {file_fingerprint(scaler_path)} |\n"
            )
            k_ver = keras.__version__
            if tf is not None:
                st.markdown(f"**TensorFlow** `{tf.__version__}` · **Keras** `{k_ver}`")
            else:
                st.markdown(f"**Keras** `{k_ver}`")
            st.markdown(
                "**Disclaimer (full):** Outputs are experimental. "
                "Past and predicted prices are not trading, legal, or tax advice."
            )


def main():
    st.set_page_config(page_title="Crypto Price & Forecast", layout="wide", initial_sidebar_state="expanded")

    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg,#f8f9fb 0%,#eef1f6 100%);
            border-radius: 12px;
            padding: 16px 20px;
            border: 1px solid #e5e9f0;
        }
        /* Prefer pointer/hand on controls instead of text caret (I-beam) over dropdowns, radios, toggles */
        .stApp [data-baseweb="select"] > div:first-child {
            cursor: pointer !important;
        }
        .stApp div[data-testid="stSelectbox"] [data-baseweb="select"] {
            cursor: pointer !important;
        }
        .stApp [data-testid="stRadio"] label,
        .stApp [data-testid="stMarkdownContainer"] div[data-testid*="radio"] ~ div label {
            cursor: pointer !important;
        }
        .stApp div[data-testid="stRadio"] div[role="radiogroup"] label {
            cursor: pointer !important;
        }
        .stApp button[kind],
        .stApp button[data-testid*="baseButton"] {
            cursor: pointer !important;
        }
        .stApp div[data-testid="stCheckbox"] label,
        .stApp div[data-testid="column"] div[data-testid="stMarkdownContainer"] summary {
            cursor: pointer !important;
        }
        .stApp [data-testid="stSlider"] div[role="slider"] {
            cursor: grab !important;
        }
        .stApp [data-testid="stSlider"]:active div[role="slider"] {
            cursor: grabbing !important;
        }
        .stApp div[data-testid="stDateInput"] input {
            cursor: pointer !important;
        }
        .stApp [data-testid="stVerticalBlockBorderWrapper"] details > summary {
            cursor: pointer !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Data")
        symbol = st.selectbox("Currency", CURRENCY_OPTIONS, index=0)
        data_mode = st.radio("Historical range source", ["Yahoo period", "Custom date range"])
        use_custom_range = data_mode == "Custom date range"
        yahoo_period = st.selectbox(
            "Yahoo Finance period",
            YAHOO_PERIODS,
            index=2,
            disabled=use_custom_range,
            help="Used only when “Historical range source” is Yahoo period.",
        )
        range_start: date | None = None
        range_end: date | None = None
        if use_custom_range:
            range_end = st.date_input("End date", value=date.today())
            range_start = st.date_input(
                "Start date",
                value=date.today() - timedelta(days=365 * 3),
            )

        price_mode = st.radio("Price series", PRICE_MODES, index=0, horizontal=True)

        st.divider()
        st.header("View")
        forecast_view = st.radio("Mode", VIEW_OPTIONS, index=0)
        forecast_horizon = st.slider(
            "Forecast horizon (days)",
            FORECAST_MIN,
            FORECAST_MAX,
            30,
            disabled=(forecast_view != "Forecast"),
        )
        history_span = st.slider("Chart history tail (sessions)", 60, 2000, DEFAULT_HISTORY_TAIL)
        st.caption(
            "LSTM input window (# of past days fed to the model) is taken from **model.input_shape**, not editable here "
            "(see dashboard metric)."
        )

        st.divider()
        st.header("Chart styling")
        chart_theme = st.selectbox("Chart theme", CHART_THEMES)
        log_scale = st.toggle("Log price axis", value=False)
        show_volume = st.toggle("Volume panel (matplotlib)", value=False)
        naive_baseline = st.toggle(
            "Naive flat baseline (forecast only)",
            value=False,
            disabled=(forecast_view != "Forecast"),
        )

        st.divider()
        st.header("Outputs")
        table_expanded = st.toggle("Forecast table starts expanded", value=False)
        show_interactive_chart = st.toggle(
            "Show interactive chart (below main figure)",
            value=True,
            help="Streamlit line chart for quick pan/zoom; uses the same history window.",
        )

        st.divider()
        st.header("Advanced")
        show_debug_panel = st.toggle(
            "Show technical panel (files, TF/Keras)",
            value=False,
            help="Adds an expandable footer with artifact paths and library versions.",
        )

        st.divider()
        st.header("Refresh")
        auto_refresh_min = st.slider("Auto refresh (minutes, 0 = off)", 0, 120, 0)
        model_path, scaler_path, artifact_note = resolve_artifact_paths(symbol)

    dash_kwargs = dict(
        symbol=symbol,
        forecast_view=forecast_view,
        yahoo_period=yahoo_period,
        use_custom_range=use_custom_range,
        range_start=range_start,
        range_end=range_end,
        price_mode=price_mode,
        history_span=history_span,
        forecast_horizon=int(forecast_horizon),
        chart_theme=chart_theme,
        log_scale=log_scale,
        show_volume=show_volume,
        naive_baseline=naive_baseline,
        table_expanded=table_expanded,
        show_interactive_chart=show_interactive_chart,
        show_debug_panel=show_debug_panel,
        model_path=model_path,
        scaler_path=scaler_path,
        artifact_note=artifact_note,
    )

    if auto_refresh_min <= 0:
        render_dashboard(**dash_kwargs)
    else:
        @st.fragment(run_every=timedelta(minutes=int(auto_refresh_min)))
        def _body():
            render_dashboard(**dash_kwargs)

        _body()


if __name__ == "__main__":
    main()
