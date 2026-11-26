import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="POTATECH - Cortadora y Peladora de Papas",
    page_icon="⚙️",
    layout="wide",
)

DEFAULT_FILE = "Optimización en el Procesamiento de Papa (respuestas).xlsx"
EST_PROFILE_LABEL = "Trabajo o soy propietario/a de un negocio de alimentos (En el pasado o presente)"

BAR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
]


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def colored_bar_chart(series: pd.Series, ylabel: str = "", horizontal: bool = False, suffix: str = ""):
    """Dibuja un gráfico de barras donde cada barra tiene un color distinto y etiqueta de valor."""
    s = series.dropna()
    if s.empty:
        st.info("Sin datos disponibles para graficar.")
        return

    labels = [str(idx) for idx in s.index]
    values = s.values.astype(float)
    colors = [BAR_PALETTE[i % len(BAR_PALETTE)] for i in range(len(labels))]

    if horizontal:
        fig_h = max(3.0, 0.45 * len(labels) + 1.0)
        fig, ax = plt.subplots(figsize=(8.0, fig_h))
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.62)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel, fontsize=9)
        max_v = max(values) if len(values) > 0 else 1.0
        ax.set_xlim(0, max_v * 1.18 if max_v > 0 else 1.0)
        for bar, val in zip(bars, values):
            txt = f"{val:.1f}{suffix}" if suffix == "%" else f"{int(val)}{suffix}"
            ax.text(
                val + max_v * 0.02,
                bar.get_y() + bar.get_height() / 2,
                txt,
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color="#222222",
            )
    else:
        fig, ax = plt.subplots(figsize=(8.5, 4.2))
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, values, color=colors, width=0.55)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9)
        max_v = max(values) if len(values) > 0 else 1.0
        ax.set_ylim(0, max_v * 1.18 if max_v > 0 else 1.0)
        for bar, val in zip(bars, values):
            txt = f"{val:.1f}{suffix}" if suffix == "%" else f"{int(val)}{suffix}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max_v * 0.02,
                txt,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#222222",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y" if not horizontal else "x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


@st.cache_data
def load_data(file_source) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    xls = pd.ExcelFile(file_source)
    sheet_name = xls.sheet_names[0]
    raw_df = pd.read_excel(xls, sheet_name=sheet_name)

    rename_map = {
        raw_df.columns[0]: "marca_temporal",
        raw_df.columns[1]: "perfil",
    }
    est_names = [
        "est_p2_kg_dia",
        "est_p3_realiza_proceso",
        "est_p4_conocia_equipo",
        "est_p5_adquirio_equipo",
        "est_p6_utilidad_1_5",
        "est_p7_disposicion_uso",
        "est_p8_plazo",
        "est_p9_factores_decision",
        "est_p10_barreras_compra",
        "est_p11_tiempo_diario",
        "est_p12_precio_max",
        "est_p13_precio_costoso",
    ]
    for idx, col in enumerate(raw_df.columns[2:14]):
        rename_map[col] = est_names[idx]

    df = raw_df.rename(columns=rename_map).copy()
    df["marca_temporal"] = pd.to_datetime(df["marca_temporal"], errors="coerce")
    df["perfil"] = df["perfil"].astype(str).map(clean_text)

    est_df = df[df["perfil"] == EST_PROFILE_LABEL].copy()
    est_df["utilidad_num"] = pd.to_numeric(
        est_df["est_p6_utilidad_1_5"].astype(str).str.extract(r"^(\d)")[0],
        errors="coerce",
    )

    meta = {
        "sheet_name": sheet_name,
        "total": len(df),
        "n_negocios": len(est_df),
        "n_hogar": int(df["perfil"].str.contains("Consumidor general", na=False).sum()),
        "n_no_aplica": int(df["perfil"].str.contains("No suelo cocinar", na=False).sum()),
        "fecha_min": df["marca_temporal"].min(),
        "fecha_max": df["marca_temporal"].max(),
    }
    return df, est_df, meta


def explode_multi(series: pd.Series, replacements: dict[str, str] | None = None) -> pd.Series:
    items = []
    for val in series.dropna():
        txt = str(val)
        if replacements:
            for old, new in replacements.items():
                txt = txt.replace(old, new)
        for part in txt.split(","):
            cleaned = clean_text(part)
            if cleaned:
                items.append(cleaned)
    return pd.Series(items).value_counts() if items else pd.Series(dtype="int64")


def build_qfd_inputs(est_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = max(len(est_df), 1)
    p9 = explode_multi(
        est_df["est_p9_factores_decision"],
        {"Era propietaria, actualmente no tengo el negocio": "Expropietario/a"},
    )
    p10 = explode_multi(est_df["est_p10_barreras_compra"])

    ques_data = [
        ("Q1", "Reducir el tiempo de preparación diaria", int(p9.get("Reducción del tiempo de preparación", 0))),
        ("Q2", "Adquirir el equipo a precio accesible (< USD 400)", int(p10.get("Precio demasiado alto", 0))),
        ("Q3", "Minimizar el desperdicio (merma) de papa", int(p9.get("Reducción del desperdicio de papa", 0))),
        ("Q4", "Ocupar poco espacio en cocina comercial", int(p10.get("Falta de espacio", 0))),
        ("Q5", "Reducir esfuerzo y mano de obra operativa", int(max(p9.get("Ahorro de mano de obra", 0), p10.get("Prefiero realizar el proceso manualmente", 0)))),
        ("Q6", "Obtener cortes uniformes y consistentes", int(p9.get("Uniformidad del corte", 0))),
        ("Q7", "Garantizar seguridad durante la operación", int(p9.get("Seguridad", 0))),
        ("Q8", "Facilitar la limpieza e higienización", int(max(p9.get("Facilidad de limpieza", 0), p10.get("Dificultad de limpieza", 0)))),
        ("Q9", "Mantener bajos costos y facilidad de mantenimiento", int(max(p9.get("Facilidad de mantenimiento", 0), p10.get("Costos de mantenimiento", 0)))),
        ("Q10", "Optimizar consumo eléctrico e hídrico", int(max(p9.get("Bajo consumo de energía", 0), p9.get("Bajo consumo de agua", 0), p10.get("Alto consumo de energía", 0)))),
    ]

    q_df = pd.DataFrame(ques_data, columns=["ID", "Requisito del Cliente (QUÉ)", "Menciones"])
    q_df["Frecuencia (%)"] = (q_df["Menciones"] / n * 100.0).round(1)
    max_m = max(q_df["Menciones"].max(), 1)
    q_df["Importancia (1-5)"] = (1.0 + 4.0 * (q_df["Menciones"] / max_m)).round(2)
    q_df["Peso Cliente (%)"] = (q_df["Importancia (1-5)"] / q_df["Importancia (1-5)"].sum() * 100.0).round(1)

    comos_meta = pd.DataFrame(
        [
            ("C1", "Capacidad de procesamiento (kg/h)", "↑ Maximizar", "≥ 20–35 kg/h (lote 2–3 kg en ≤ 4 min)"),
            ("C2", "Porcentaje de merma en pelado y corte (%)", "↓ Minimizar", "≤ 10–12% de pérdida de masa"),
            ("C3", "Costo de adaptacion de la maquina/Valor objetivo (USD)", "↓ Minimizar", "≤ USD 200–399 costo/valor objetivo"),
            ("C4", "Huella en planta y masa total (mm / kg)", "↓ Minimizar", "≤ 450 × 350 × 550 mm; masa ≤ 22 kg"),
            ("C5", "Tolerancia dimensional de corte (mm)", "↓ Minimizar", "Bastón 9 × 9 mm (± 1.0 mm)"),
            ("C6", "Potencia eléctrica instalada (W)", "↓ Minimizar", "370–550 W (0.5–0.75 HP, 110 V)"),
            ("C7", "Tiempo de desensamble y limpieza (min)", "↓ Minimizar", "≤ 3 min sin herramientas; AISI 304"),
            ("C8", "Nivel de protección y seguridad mecánica", "↑ Maximizar", "Microswitch en tapa + paro de emergencia"),
        ],
        columns=["Código", "Característica Técnica (CÓMO)", "Dirección", "Valor Objetivo Inicial"],
    )

    rel_init = pd.DataFrame(
        [
            [9, 1, 1, 0, 1, 3, 1, 0],
            [3, 1, 9, 3, 1, 3, 1, 1],
            [1, 9, 1, 0, 3, 0, 0, 0],
            [3, 0, 3, 9, 0, 1, 1, 0],
            [9, 1, 1, 1, 1, 3, 3, 1],
            [1, 3, 1, 0, 9, 1, 0, 0],
            [0, 0, 3, 1, 0, 1, 1, 9],
            [0, 1, 1, 1, 0, 0, 9, 3],
            [0, 1, 3, 1, 1, 1, 3, 1],
            [3, 3, 1, 0, 0, 9, 1, 0],
        ],
        index=q_df["ID"],
        columns=comos_meta["Código"],
    )

    roof_df = pd.DataFrame(
        [
            ["—", "-", "--", "--", "0", "--", "0", "0"],
            ["-", "—", "-", "0", "+", "0", "0", "0"],
            ["--", "-", "—", "+", "-", "--", "-", "--"],
            ["--", "0", "+", "—", "0", "+", "+", "-"],
            ["0", "+", "-", "0", "—", "0", "-", "0"],
            ["--", "0", "--", "+", "0", "—", "0", "0"],
            ["0", "0", "-", "+", "-", "0", "—", "+"],
            ["0", "0", "--", "-", "0", "0", "+", "—"],
        ],
        index=comos_meta["Código"],
        columns=comos_meta["Código"],
    )

    return q_df, comos_meta, rel_init, roof_df


# --- BARRA LATERAL ---
st.sidebar.title("Configuración POTATECH")
uploaded = st.sidebar.file_uploader("Cargar Excel/CSV de respuestas", type=["xlsx", "xls", "csv"])
file_source = uploaded if uploaded is not None else DEFAULT_FILE

try:
    _, est_df, meta = load_data(file_source)
except Exception as exc:
    st.error(f"No se pudo leer el archivo: {exc}")
    st.stop()

# --- ENCABEZADO VERTICAL (ÍCONO ARRIBA, TÍTULO ABAJO) ---
st.markdown(
    f"""
    <div style="text-align: center; padding: 0.4rem 0 1.0rem 0;">
        <div style="font-size: 2.8rem; line-height: 1.15;">⚙️</div>
        <h1 style="margin: 0.25rem 0; font-size: 2.0rem;">POTATECH - Cortadora y Peladora de Papas</h1>
        <p style="margin: 0; color: #666666; font-size: 0.95rem;">
            Casa de la Calidad (QFD) | Segmento objetivo: <b>Negocios de alimentos (n = {meta['n_negocios']})</b>
            de <b>{meta['total']}</b> encuestas ({meta['n_hogar']} hogar, {meta['n_no_aplica']} no aplica)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 4 MÉTRICAS EN DISPOSICIÓN 2 Y 2 EN VERTICAL ---
u_series = est_df["utilidad_num"].dropna()
disp_pct = est_df["est_p7_disposicion_uso"].isin(
    ["Definitivamente la utilizaría", "Probablemente la utilizaría"]
).mean() * 100
pct_negocios = (meta["n_negocios"] / max(meta["total"], 1)) * 100

with st.container():
    st.metric(
        "Negocios encuestados",
        f"{meta['n_negocios']} ({pct_negocios:.1f}%)",
        f"De {meta['total']} respuestas totales",
    )
    st.metric(
        "Utilidad media (1–5)",
        f"{u_series.mean():.2f}",
        f"Mediana: {u_series.median():.1f}",
    )

with st.container():
    st.metric(
        "Disposición favorable",
        f"{disp_pct:.1f}%",
        "Definitivamente / Probablemente",
    )
    st.metric(
        "Rango precio viable",
        "≤ USD 399",
        "13/16 negocios (81.3%)",
    )

st.markdown("---")

# --- PESTAÑAS ---
tab_est, tab_ques, tab_qfd, tab_tech = st.tabs([
    "1. Estadística (Negocios)",
    "2. Necesidades Jerarquizadas (QUÉS)",
    "3. Casa de la Calidad (Matriz QFD)",
    "4. Requisitos Técnicos (CÓMOS)",
])

with tab_est:
    st.subheader("Volumen diario de papa (`P2`)")
    order_p2 = ["Menos de 5 kg", "5–10 kg", "11–20 kg", "21–40 kg", "Más de 100 kg"]
    colored_bar_chart(
        est_df["est_p2_kg_dia"].value_counts().reindex(order_p2).dropna(),
        ylabel="Número de negocios",
        horizontal=False,
    )

    st.subheader("Precio máximo aceptable (`P12`)")
    colored_bar_chart(
        est_df["est_p12_precio_max"].value_counts(),
        ylabel="Número de negocios",
        horizontal=False,
    )

    st.subheader("Factores decisivos de compra (`P9`)")
    p9_vc = explode_multi(
        est_df["est_p9_factores_decision"],
        {"Era propietaria, actualmente no tengo el negocio": "Expropietario/a"},
    )
    colored_bar_chart(p9_vc, ylabel="Menciones", horizontal=False)

    st.subheader("Barreras para no adquirir (`P10`)")
    p10_vc = explode_multi(est_df["est_p10_barreras_compra"])
    colored_bar_chart(p10_vc, ylabel="Menciones", horizontal=False)

q_df, comos_meta, rel_init, roof_df = build_qfd_inputs(est_df)

with tab_ques:
    st.subheader("Voz del Cliente Jerarquizada (QUÉS)")
    st.write("Puedes ajustar la importancia (1 a 5) directamente en la tabla si deseas recalcular la Casa de la Calidad:")
    edited_q = st.data_editor(
        q_df,
        column_config={
            "Importancia (1-5)": st.column_config.NumberColumn(min_value=1.0, max_value=5.0, step=0.1),
        },
        disabled=["ID", "Requisito del Cliente (QUÉ)", "Menciones", "Frecuencia (%)", "Peso Cliente (%)"],
        width="stretch",
        hide_index=True,
    )
    edited_q["Peso Cliente (%)"] = (
        edited_q["Importancia (1-5)"] / edited_q["Importancia (1-5)"].sum() * 100.0
    ).round(1)

    st.subheader("Peso relativo de cada necesidad del cliente (%)")
    colored_bar_chart(
        edited_q.set_index("Requisito del Cliente (QUÉ)")["Peso Cliente (%)"],
        ylabel="Peso relativo (%)",
        horizontal=False,
        suffix="%",
    )

with tab_qfd:
    st.subheader("Matriz de Relaciones (QUÉS vs. CÓMOS) y Techo de Correlaciones")
    st.caption("Convención de relación: **0** = Nula | **1** = Débil | **3** = Media | **9** = Fuerte (celdas editables).")

    with st.expander("Ver Techo de Correlaciones Técnicas (Interacción entre CÓMOS)", expanded=False):
        st.caption("`++` Sinergia fuerte | `+` Sinergia | `0` Neutra | `-` Compromiso (trade-off) | `--` Compromiso fuerte")
        st.dataframe(roof_df, width="stretch")

    rel_display = rel_init.copy()
    rel_display.insert(0, "Importancia", edited_q["Importancia (1-5)"].values)
    rel_display.insert(0, "Requisito (QUÉ)", edited_q["Requisito del Cliente (QUÉ)"].values)

    edited_rel_full = st.data_editor(
        rel_display,
        disabled=["Requisito (QUÉ)", "Importancia"],
        width="stretch",
    )

    rel_numeric = edited_rel_full[comos_meta["Código"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    imp_vec = edited_q["Importancia (1-5)"].values
    abs_scores = imp_vec @ rel_numeric
    rel_scores = (abs_scores / max(abs_scores.sum(), 1e-9) * 100.0).round(1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(rel_numeric, cmap="YlGnBu", aspect="auto", vmin=0, vmax=9)
    ax.set_xticks(range(len(comos_meta)))
    ax.set_xticklabels(comos_meta["Código"], fontweight="bold")
    ax.set_yticks(range(len(edited_q)))
    ax.set_yticklabels([f"{r['ID']} ({r['Importancia (1-5)']})" for _, r in edited_q.iterrows()])
    for i in range(rel_numeric.shape[0]):
        for j in range(rel_numeric.shape[1]):
            val = int(rel_numeric[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", color="white" if val >= 6 else "black", fontweight="bold")
    ax.set_title("Mapa de Relaciones QFD (QUÉS vs. CÓMOS)")
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with tab_tech:
    st.subheader("Requisitos Técnicos Priorizados por la Casa de la Calidad")
    tech_out = comos_meta.copy()
    tech_out["Puntuación Absoluta QFD"] = abs_scores.round(1)
    tech_out["Importancia Técnica (%)"] = rel_scores
    tech_out = tech_out.sort_values("Puntuación Absoluta QFD", ascending=False).reset_index(drop=True)
    tech_out.insert(0, "Prioridad", tech_out.index + 1)

    st.dataframe(tech_out, width="stretch", hide_index=True)

    st.subheader("Importancia técnica relativa de los requisitos (%)")
    colored_bar_chart(
        tech_out.set_index("Característica Técnica (CÓMO)")["Importancia Técnica (%)"],
        ylabel="Importancia técnica (%)",
        horizontal=False,
        suffix="%",
    )

    csv_qfd = tech_out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Descargar Matriz de Requisitos Técnicos QFD (.csv)",
        data=csv_qfd,
        file_name="potatech_qfd_requisitos_tecnicos.csv",
        mime="text/csv",
    )
