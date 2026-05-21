"""
WebCorp Business - Sistema de Control Logistico e Inteligencia Operativa v3.1
Area de Inteligencia de Negocios
Desarrollado en Streamlit | Guatemala

FIXES APLICADOS (v3.0):
- Efectividad por cohorte de antiguedad (no sobre ordenes de hoy)
- "Entregado sin Liquidar" como ALERTA ROJA de cartera vencida
- datetime truncado a medianoche para EDAD_DIAS
- Parser de fechas robusto con reporte de calidad de datos
- Modo permisivo: carga todo, analiza lo limpio, reporta lo sucio
- Deduplicacion automatica por ORDEN
- VALOR_NUM limpia formatos Q, $, comas antes de convertir
- Ranking de asesores con filtro minimo de volumen
- Lead Time Deposito -> Liquidacion
- Alertas basadas en % del volumen, no montos fijos
- White Spaces (volumen vs efectividad) [MKT #1]
- Riesgo Competencia desde Sub-Status [MKT #2]
- Segmentacion por Perfil Logistico de Cliente [MKT #4]

RETRO JEFA (v3.1):
- Efectividad por cohortes de tiempo: 24h, 48h, 72h, 72h+
- Desglose de efectividad por canal (CONTROL INTERNO / mensajeria, cargo, forza, MEG)
- Heatmap canal x cohorte
- Antiguedad por STATUS con explicacion clara y tabla resumen
- Geografico ordenado por cantidad de ordenes (impacto), no por efectividad
- Departamentos con dual-axis: volumen + efectividad
- Retornos como tasa proporcional (%), no conteo absoluto
- Texto en retornos muestra retornos/total por producto
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from typing import Tuple, List, Dict, Optional
import uuid
import re

# ============================================================================
# CONFIGURACION Y BRANDING WEBCORP
# ============================================================================

# Paleta WebCorp Business - Inteligencia de Negocios
WC = {
    'primary':     '#1B4F8A',   # Azul oscuro corporativo
    'secondary':   '#2E86DE',   # Azul WebCorp (del logo)
    'accent':      '#0ABDE3',   # Cian claro
    'success':     '#10AC84',   # Verde exito
    'warning':     '#F39C12',   # Ambar alerta
    'danger':      '#EE5A24',   # Rojo peligro
    'critical':    '#B71540',   # Rojo critico
    'bg':          '#FFFFFF',   # Fondo blanco
    'bg_light':    '#F8F9FA',   # Gris muy claro
    'bg_card':     '#F1F2F6',   # Gris tarjeta
    'text':        '#2D3436',   # Texto principal
    'text_muted':  '#636E72',   # Texto secundario
    'border':      '#DFE6E9',   # Bordes
}

st.set_page_config(
    page_title="WebCorp BI | Control Logistico",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>W</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {WC['bg']};
    }}

    /* Header brand bar */
    .brand-bar {{
        background: linear-gradient(135deg, {WC['primary']} 0%, {WC['secondary']} 100%);
        padding: 16px 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .brand-bar h1 {{
        margin: 0; font-size: 1.5rem; font-weight: 700; color: white;
    }}
    .brand-bar p {{
        margin: 0; font-size: 0.8rem; opacity: 0.85; color: white;
    }}

    /* KPI Cards */
    .kpi-card {{
        background: {WC['bg']};
        border: 1px solid {WC['border']};
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .kpi-card h4 {{
        margin: 0; font-size: 0.75rem; font-weight: 500;
        color: {WC['text_muted']}; text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .kpi-card .kpi-value {{
        font-size: 1.8rem; font-weight: 700; margin: 6px 0 2px 0;
    }}
    .kpi-card .kpi-sub {{
        font-size: 0.72rem; color: {WC['text_muted']}; margin: 0;
    }}
    .kpi-blue .kpi-value {{ color: {WC['secondary']}; }}
    .kpi-green .kpi-value {{ color: {WC['success']}; }}
    .kpi-orange .kpi-value {{ color: {WC['warning']}; }}
    .kpi-red .kpi-value {{ color: {WC['danger']}; }}
    .kpi-critical .kpi-value {{ color: {WC['critical']}; }}

    /* Alert boxes */
    .alert-wc {{
        padding: 14px 18px;
        border-radius: 8px;
        margin: 8px 0;
        font-size: 0.85rem;
        border-left: 4px solid;
    }}
    .alert-danger {{
        background-color: #FFF5F5; border-color: {WC['danger']};
        color: {WC['danger']};
    }}
    .alert-warning {{
        background-color: #FFFBF0; border-color: {WC['warning']};
        color: #946200;
    }}
    .alert-success {{
        background-color: #F0FFF4; border-color: {WC['success']};
        color: #0B6E4F;
    }}
    .alert-info {{
        background-color: #EBF5FF; border-color: {WC['secondary']};
        color: {WC['primary']};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: {WC['bg_light']};
        border-radius: 8px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 40px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.85rem;
    }}

    /* Section dividers */
    .section-header {{
        font-size: 1.1rem;
        font-weight: 600;
        color: {WC['primary']};
        border-bottom: 2px solid {WC['secondary']};
        padding-bottom: 8px;
        margin: 20px 0 12px 0;
    }}

    /* Data quality badge */
    .dq-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
    }}
    .dq-good {{ background: #D4EDDA; color: #155724; }}
    .dq-warn {{ background: #FFF3CD; color: #856404; }}
    .dq-bad {{ background: #F8D7DA; color: #721C24; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {WC['bg_light']};
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, {WC['primary']}, {WC['secondary']});
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CATALOGOS
# ============================================================================

COLUMNAS_OBLIGATORIAS = [
    'CLIENTE', 'ASESOR', 'GUIA', 'FECHA', 'REMITENTE', 'DESTINATARIO',
    'DIRECCION', 'TELEFONO', 'COD', 'VALOR', 'ORDEN', 'REFERENCIA',
    'FECHA DEPOSITO', 'STATUS', 'CONTROL INTERNO', 'SUB STATUS',
    'INTENTOS DE ENTREGA', 'REPROGRAMADO'
]

STATUS_VALIDOS = [
    'ENTREGADO LIQUIDADO', 'ENTREGADO', 'EN RUTA', 'EN GESTION',
    'REPROGRAMADO', 'ILOCALIZABLE', 'RECHAZADO', 'RECLAMO',
    'FUERA DE COBERTURA', 'EN RUTA PARA DEVOLUCION', 'RETORNADO A WEBCORP'
]

# Sub-status que indican riesgo competitivo (MKT #2)
RIESGO_COMPETENCIA = [
    'NO HIZO PEDIDO', 'PRECIO INCORRECTO',
    'COMPRA OTRO PRODUCTO', 'ERROR EN PRODUCTO'
]

# ============================================================================
# FUNCIONES DE VALIDACION (MODO PERMISIVO)
# ============================================================================

def validar_estructura_csv(df: pd.DataFrame, nombre_archivo: str) -> Tuple[bool, str, List[str]]:
    """Valida columnas. Retorna (ok, mensaje, columnas_faltantes)."""
    columnas_base = []
    for col in df.columns:
        col_limpio = col.strip()
        if '.' in col_limpio:
            partes = col_limpio.rsplit('.', 1)
            if partes[1].isdigit():
                col_limpio = partes[0]
        columnas_base.append(col_limpio.upper())

    columnas_unicas = set(columnas_base)
    faltantes = [c for c in COLUMNAS_OBLIGATORIAS if c.upper() not in columnas_unicas]

    if faltantes:
        return False, f"{nombre_archivo}: Columnas faltantes: {', '.join(faltantes)}", faltantes
    return True, "", []


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas, elimina duplicadas y Unnamed."""
    df = df.copy()
    nuevos = []
    for col in df.columns:
        nombre = str(col).strip() if col is not None else ''
        nombre = nombre.replace('\xa0', ' ')
        nombre = ' '.join(nombre.split())
        nuevos.append(nombre)
    df.columns = nuevos

    columnas_ok = []
    vistos = set()
    for col in df.columns:
        base = col
        if '.' in col:
            partes = col.rsplit('.', 1)
            if partes[1].isdigit():
                base = partes[0]
        if base not in vistos and base != '':
            columnas_ok.append(col)
            vistos.add(base)
    df = df[columnas_ok]

    renames = {}
    for col in df.columns:
        if '.' in col:
            partes = col.rsplit('.', 1)
            if partes[1].isdigit():
                renames[col] = partes[0]
    if renames:
        df = df.rename(columns=renames)

    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df.loc[:, df.columns != '']
    return df


def parsear_fecha_robusto(serie: pd.Series) -> Tuple[pd.Series, float]:
    """Parsea fechas con pd.to_datetime, retorna (serie_dt, pct_fallos)."""
    resultado = pd.to_datetime(serie, format='%d/%m/%y', errors='coerce')
    mask_na = resultado.isna() & serie.notna() & (serie.astype(str).str.strip() != '') & (~serie.astype(str).str.strip().isin(['#N/A', 'N/A', 'nan', 'None', '']))
    if mask_na.any():
        for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']:
            pendientes = mask_na & resultado.isna()
            if not pendientes.any():
                break
            resultado[pendientes] = pd.to_datetime(serie[pendientes], format=fmt, errors='coerce')

    total_no_vacio = serie.notna() & (serie.astype(str).str.strip() != '') & (~serie.astype(str).str.strip().isin(['#N/A', 'N/A', 'nan', 'None']))
    n_total = total_no_vacio.sum()
    n_fallos = (resultado.isna() & total_no_vacio).sum()
    pct_fallos = (n_fallos / n_total * 100) if n_total > 0 else 0.0
    return resultado, pct_fallos


def limpiar_valor(serie: pd.Series) -> Tuple[pd.Series, float]:
    """Limpia Q, $, comas, espacios antes de convertir a numerico."""
    limpio = serie.astype(str).str.replace(r'[Q$\s,]', '', regex=True)
    resultado = pd.to_numeric(limpio, errors='coerce')
    total_no_vacio = serie.notna() & (serie.astype(str).str.strip() != '')
    n_total = total_no_vacio.sum()
    n_nulos = (resultado.isna() & total_no_vacio).sum()
    pct_fallos = (n_nulos / n_total * 100) if n_total > 0 else 0.0
    resultado = resultado.fillna(0)
    return resultado, pct_fallos


def extraer_intentos(valor) -> int:
    if pd.isna(valor) or str(valor).strip() == '':
        return 0
    match = re.search(r'(\d+)\s*INTENTO', str(valor).upper().strip())
    return int(match.group(1)) if match else 0


def extraer_componente_direccion(direccion, idx: int) -> str:
    """Extrae componente por indice del formato Region,Depto,Zona,..."""
    if pd.isna(direccion) or str(direccion).strip() == '':
        return 'SIN DATO'
    partes = str(direccion).split(',')
    if len(partes) > idx:
        val = partes[idx].strip()
        return val if val else 'SIN DATO'
    return 'SIN DATO'


def procesar_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Procesa el DataFrame. Retorna (df_procesado, reporte_calidad)."""
    df = df.copy()
    calidad = {}

    # Normalizar STATUS
    if 'STATUS' in df.columns:
        df['STATUS'] = df['STATUS'].astype(str).str.strip().str.upper()
        df.loc[df['STATUS'].isin(['NAN', '', 'NONE']), 'STATUS'] = 'SIN STATUS'

    if 'SUB STATUS' in df.columns:
        df['SUB STATUS'] = df['SUB STATUS'].astype(str).str.strip().str.upper()
        df.loc[df['SUB STATUS'].isin(['NAN', '', 'NONE']), 'SUB STATUS'] = ''

    # Fechas
    df['FECHA_DT'], pct_fecha = parsear_fecha_robusto(df['FECHA'])
    calidad['pct_fecha_invalida'] = pct_fecha

    df['FECHA_DEPOSITO_DT'] = pd.NaT
    if 'FECHA DEPOSITO' in df.columns:
        df['FECHA_DEPOSITO_DT'], pct_dep = parsear_fecha_robusto(df['FECHA DEPOSITO'])
        calidad['pct_fecha_deposito_invalida'] = pct_dep

    # FIX: EDAD_DIAS con medianoche, no datetime.now()
    hoy = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    df['EDAD_DIAS'] = df['FECHA_DT'].apply(lambda x: (hoy - x).days if pd.notna(x) else None)

    # Lead Time: Deposito -> hoy (o liquidacion)
    df['LEAD_TIME'] = None
    mask_dep = df['FECHA_DEPOSITO_DT'].notna() & df['FECHA_DT'].notna()
    df.loc[mask_dep, 'LEAD_TIME'] = (df.loc[mask_dep, 'FECHA_DEPOSITO_DT'] - df.loc[mask_dep, 'FECHA_DT']).dt.days

    # Geograficos
    df['REGION'] = df['DIRECCION'].apply(lambda x: extraer_componente_direccion(x, 0))
    df['DEPARTAMENTO'] = df['DIRECCION'].apply(lambda x: extraer_componente_direccion(x, 1))
    df['GEO_KEY'] = df['DIRECCION'].apply(lambda x: extraer_componente_direccion(x, 2))

    calidad['pct_geo_sin_dato'] = (df['GEO_KEY'] == 'SIN DATO').mean() * 100

    # Intentos
    df['NUM_INTENTOS'] = df['INTENTOS DE ENTREGA'].apply(extraer_intentos)

    # Flags de estado
    df['ES_LIQUIDADO'] = df['STATUS'] == 'ENTREGADO LIQUIDADO'
    df['ES_ENTREGADO'] = df['STATUS'] == 'ENTREGADO'
    df['ES_ENTREGADO_O_LIQUIDADO'] = df['ES_LIQUIDADO'] | df['ES_ENTREGADO']
    df['ES_METRO_GUATEMALA'] = df['DIRECCION'].astype(str).str.upper().str.startswith('REGION METROPOLITANA,GUATEMALA,')

    # FIX: VALOR_NUM con limpieza de formato
    df['VALOR_NUM'], pct_valor = limpiar_valor(df['VALOR'])
    calidad['pct_valor_invalido'] = pct_valor

    # Producto (primer componente de REFERENCIA)
    df['PRODUCTO'] = df['REFERENCIA'].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'SIN REFERENCIA'
    )

    # Riesgo competencia (MKT #2)
    df['ES_RIESGO_COMPETENCIA'] = df['SUB STATUS'].isin(RIESGO_COMPETENCIA)

    return df, calidad


# ============================================================================
# METRICAS (CORREGIDAS)
# ============================================================================

def calcular_efectividad_cohorte(df: pd.DataFrame, dias_minimo: int = 3) -> float:
    """FIX: Efectividad = Liquidadas / Ordenes con >= N dias de antiguedad.
    Excluye ordenes jovenes del denominador."""
    df_cohorte = df[df['EDAD_DIAS'].notna() & (df['EDAD_DIAS'] >= dias_minimo)]
    if len(df_cohorte) == 0:
        return 0.0
    return (df_cohorte['ES_LIQUIDADO'].sum() / len(df_cohorte)) * 100


def calcular_efectividad_simple(df: pd.DataFrame) -> float:
    """Efectividad bruta (para compatibilidad)."""
    if len(df) == 0:
        return 0.0
    return (df['ES_LIQUIDADO'].sum() / len(df)) * 100


def obtener_kpis_generales(df: pd.DataFrame) -> Dict:
    total = len(df)
    liquidados = int(df['ES_LIQUIDADO'].sum())
    entregados_sin_liq = int(df['ES_ENTREGADO'].sum())
    efectividad_cohorte = calcular_efectividad_cohorte(df, dias_minimo=3)
    efectividad_bruta = calcular_efectividad_simple(df)

    return {
        'total_ordenes': total,
        'total_liquidadas': liquidados,
        'total_entregadas_sin_liquidar': entregados_sin_liq,
        'total_pendientes': total - liquidados,
        'efectividad_cohorte': efectividad_cohorte,
        'efectividad_bruta': efectividad_bruta,
        'valor_total': df['VALOR_NUM'].sum(),
        'valor_liquidado': df.loc[df['ES_LIQUIDADO'], 'VALOR_NUM'].sum(),
        'valor_entregado_sin_liquidar': df.loc[df['ES_ENTREGADO'], 'VALOR_NUM'].sum(),
        'valor_en_ruta': df.loc[df['STATUS'] == 'EN RUTA', 'VALOR_NUM'].sum(),
        'valor_pendiente': df.loc[~df['ES_LIQUIDADO'], 'VALOR_NUM'].sum(),
    }


# ============================================================================
# COMPONENTES UI
# ============================================================================

def brand_header():
    st.markdown(f"""
    <div class="brand-bar">
        <div>
            <h1>WEBCORP Business</h1>
            <p>Area de Inteligencia de Negocios | Control Logistico e Inteligencia Operativa</p>
        </div>
        <div style="text-align:right;">
            <p style="font-size:0.7rem; opacity:0.7;">v3.1</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(titulo, valor, subtitulo="", tipo="blue"):
    st.markdown(f"""
    <div class="kpi-card kpi-{tipo}">
        <h4>{titulo}</h4>
        <p class="kpi-value">{valor}</p>
        <p class="kpi-sub">{subtitulo}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(texto):
    st.markdown(f'<div class="section-header">{texto}</div>', unsafe_allow_html=True)


def alert_box(texto, tipo="info"):
    st.markdown(f'<div class="alert-wc alert-{tipo}">{texto}</div>', unsafe_allow_html=True)


def data_quality_badge(pct_fallo):
    if pct_fallo == 0:
        return '<span class="dq-badge dq-good">OK</span>'
    elif pct_fallo < 5:
        return f'<span class="dq-badge dq-warn">{pct_fallo:.1f}% con error</span>'
    else:
        return f'<span class="dq-badge dq-bad">{pct_fallo:.1f}% con error</span>'


def convertir_df_a_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


def descargar_csv(df, nombre, texto="Descargar CSV", key=None):
    csv = convertir_df_a_csv(df)
    if key is None:
        key = f"dl_{nombre}_{uuid.uuid4().hex[:6]}"
    st.download_button(label=texto, data=csv, file_name=nombre, mime='text/csv', key=key)


# ============================================================================
# GRAFICOS (con template WebCorp)
# ============================================================================

WC_TEMPLATE = dict(
    layout=dict(
        font=dict(family="Inter, sans-serif", color=WC['text']),
        paper_bgcolor=WC['bg'],
        plot_bgcolor=WC['bg'],
        title_font=dict(size=14, color=WC['primary']),
        margin=dict(t=50, b=40, l=40, r=20),
    )
)

def aplicar_tema(fig):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=WC['text'], size=11),
        paper_bgcolor=WC['bg'],
        plot_bgcolor=WC['bg_light'],
        title_font=dict(size=14, color=WC['primary']),
    )
    return fig


def crear_grafico_status(df):
    status_counts = df['STATUS'].value_counts()
    colores = [WC['success'], WC['secondary'], WC['accent'], WC['warning'],
               WC['danger'], WC['critical'], '#636E72', '#B2BEC3', '#DFE6E9',
               '#74B9FF', '#A29BFE']
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index, values=status_counts.values,
        hole=0.5, marker_colors=colores[:len(status_counts)],
        textinfo='percent+label', textposition='outside',
        textfont_size=10
    )])
    fig.update_layout(title="Distribucion por STATUS", showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.3), height=420)
    return aplicar_tema(fig)


def crear_grafico_tendencia(df):
    df_f = df[df['FECHA_DT'].notna()].copy()
    if len(df_f) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos de fecha validos", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    df_f['FECHA_SOLO'] = df_f['FECHA_DT'].dt.date
    tend = df_f.groupby('FECHA_SOLO').agg(total=('ORDEN', 'count'), liquidados=('ES_LIQUIDADO', 'sum')).reset_index()
    tend['efectividad'] = (tend['liquidados'] / tend['total']) * 100
    tend = tend.sort_values('FECHA_SOLO')

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name='Ordenes', x=tend['FECHA_SOLO'], y=tend['total'],
                         marker_color=WC['secondary'], opacity=0.7, text=tend['total'],
                         textposition='outside'), secondary_y=False)
    fig.add_trace(go.Scatter(name='Efectividad %', x=tend['FECHA_SOLO'], y=tend['efectividad'],
                             mode='lines+markers', line=dict(color=WC['danger'], width=3),
                             text=[f"{e:.0f}%" for e in tend['efectividad']],
                             textposition='top center'), secondary_y=True)
    fig.add_hline(y=65, line_dash="dash", line_color=WC['success'], annotation_text="Meta 65%", secondary_y=True)
    fig.update_layout(title=f"Tendencia ({len(tend)} dias)", height=420,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis=dict(tickformat='%d/%m', tickangle=-45))
    fig.update_yaxes(title_text="Ordenes", secondary_y=False)
    fig.update_yaxes(title_text="Efectividad (%)", secondary_y=True, range=[0, 100])
    return aplicar_tema(fig)


# ============================================================================
# SECCIONES DEL DASHBOARD
# ============================================================================

def seccion_calidad_datos(calidad: Dict, n_duplicados: int, n_filas_invalidas: int, n_total: int):
    """Reporte de calidad de datos (modo permisivo)."""
    section_header("Calidad de Datos")
    cols = st.columns(5)
    items = [
        ("Fechas", calidad.get('pct_fecha_invalida', 0)),
        ("Fecha Deposito", calidad.get('pct_fecha_deposito_invalida', 0)),
        ("Valores (Q)", calidad.get('pct_valor_invalido', 0)),
        ("Geo (sin dato)", calidad.get('pct_geo_sin_dato', 0)),
    ]
    for i, (label, pct) in enumerate(items):
        with cols[i]:
            badge = data_quality_badge(pct)
            st.markdown(f"**{label}**: {badge}", unsafe_allow_html=True)
    with cols[4]:
        if n_duplicados > 0:
            st.markdown(f'**Duplicados**: <span class="dq-badge dq-warn">{n_duplicados} removidos</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'**Duplicados**: <span class="dq-badge dq-good">0</span>', unsafe_allow_html=True)


def seccion_dashboard_principal(df):
    kpis = obtener_kpis_generales(df)

    section_header("KPIs Generales")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Total Ordenes", f"{kpis['total_ordenes']:,}", "En sistema", "blue")
    with c2:
        kpi_card("Liquidadas", f"{kpis['total_liquidadas']:,}",
                 f"Pendientes: {kpis['total_pendientes']:,}", "green")
    with c3:
        # FIX: Efectividad por cohorte
        efect = kpis['efectividad_cohorte']
        tipo_e = "green" if efect >= 65 else "red"
        kpi_card("Efectividad (3+ dias)", f"{efect:.1f}%",
                 f"Bruta: {kpis['efectividad_bruta']:.1f}% | Meta: 65%", tipo_e)
    with c4:
        kpi_card("Valor Liquidado", f"Q{kpis['valor_liquidado']:,.0f}",
                 f"En ruta: Q{kpis['valor_en_ruta']:,.0f}", "blue")
    with c5:
        # FIX: "Entregado sin Liquidar" como ALERTA ROJA
        n_esl = kpis['total_entregadas_sin_liquidar']
        v_esl = kpis['valor_entregado_sin_liquidar']
        tipo_esl = "critical" if v_esl > 5000 else "orange"
        kpi_card("CARTERA VENCIDA", f"{n_esl} ordenes",
                 f"Q{v_esl:,.0f} sin cobrar", tipo_esl)

    # Alerta cartera vencida
    if kpis['valor_entregado_sin_liquidar'] > 0:
        alert_box(
            f"CARTERA VENCIDA: {kpis['total_entregadas_sin_liquidar']} ordenes entregadas "
            f"sin liquidar por Q{kpis['valor_entregado_sin_liquidar']:,.0f}. "
            f"Esto NO es un estado intermedio, es dinero en la calle sin cobrar.",
            "danger"
        )

    # ================================================================
    # EFECTIVIDAD POR COHORTES DE TIEMPO (retro jefa)
    # ================================================================
    section_header("Efectividad por Cohorte de Tiempo")
    st.caption("Que tan rapido se liquidan las ordenes. Cada cohorte muestra ordenes que YA tienen esa antiguedad minima.")

    # Calcular cohortes
    df_con_edad = df[df['EDAD_DIAS'].notna()].copy()
    cohortes = [
        ('24 hrs', 1, 1),
        ('48 hrs', 2, 2),
        ('72 hrs', 3, 3),
        ('72+ hrs', 4, 9999),
    ]

    # -- Tabla general de cohortes --
    cohorte_data = []
    for label, min_d, max_d in cohortes:
        mask = (df_con_edad['EDAD_DIAS'] >= min_d) & (df_con_edad['EDAD_DIAS'] <= max_d) if max_d < 9999 else (df_con_edad['EDAD_DIAS'] >= min_d)
        # Para cohorte "exacta" (ej 24h = ordenes con exactamente 1 dia)
        # usamos rango para la columna, pero para efectividad acumulada usamos >= min_d
        df_cohorte_exacta = df_con_edad[(df_con_edad['EDAD_DIAS'] >= min_d) & (df_con_edad['EDAD_DIAS'] <= max_d)] if max_d < 9999 else df_con_edad[df_con_edad['EDAD_DIAS'] >= min_d]
        df_cohorte_acum = df_con_edad[df_con_edad['EDAD_DIAS'] >= min_d]
        n_acum = len(df_cohorte_acum)
        liq_acum = df_cohorte_acum['ES_LIQUIDADO'].sum()
        efect_acum = (liq_acum / n_acum * 100) if n_acum > 0 else 0
        cohorte_data.append({
            'Cohorte': label,
            'Ordenes (acumulado)': n_acum,
            'Liquidadas': int(liq_acum),
            'Efectividad %': round(efect_acum, 1)
        })

    df_cohortes = pd.DataFrame(cohorte_data)

    c1, c2 = st.columns([1, 2])
    with c1:
        # Tabla de cohortes general
        st.markdown("**General (todas las ordenes)**")
        st.dataframe(df_cohortes, use_container_width=True, hide_index=True)

    with c2:
        # Grafico de barras de efectividad por cohorte
        fig_coh = go.Figure()
        colors_coh = [WC['danger'], WC['warning'], WC['accent'], WC['success']]
        for i, row in df_cohortes.iterrows():
            fig_coh.add_trace(go.Bar(
                x=[row['Cohorte']], y=[row['Efectividad %']],
                marker_color=colors_coh[i] if i < len(colors_coh) else WC['secondary'],
                text=f"{row['Efectividad %']:.1f}%", textposition='outside',
                name=row['Cohorte'], showlegend=False
            ))
        fig_coh.add_hline(y=65, line_dash="dash", line_color=WC['success'], annotation_text="Meta 65%")
        fig_coh.update_layout(title="Efectividad Acumulada por Cohorte de Tiempo",
                              yaxis_title="Efectividad %", yaxis_range=[0, 100], height=350)
        st.plotly_chart(aplicar_tema(fig_coh), use_container_width=True, key="coh_general")

    # -- Desglose por CANAL (CONTROL INTERNO) --
    # NOTA: El campo CONTROL INTERNO mapea a los canales operativos.
    # Si tu data tiene un campo CANAL separado (mensajeria, cargo, forza, MEG),
    # cambia 'CONTROL INTERNO' por ese campo aqui.
    section_header("Efectividad por Cohorte y Canal")
    st.caption("Desglose por canal operativo (campo CONTROL INTERNO). Si los canales son mensajeria/cargo/forza/MEG, mapear en el CSV.")

    canal_col = 'CONTROL INTERNO'
    if canal_col in df.columns:
        canales = sorted(df_con_edad[canal_col].dropna().unique().tolist())
        canal_cohorte_rows = []
        for canal in canales:
            df_canal = df_con_edad[df_con_edad[canal_col] == canal]
            for label, min_d, max_d in cohortes:
                df_c = df_canal[df_canal['EDAD_DIAS'] >= min_d]
                n = len(df_c)
                liq = df_c['ES_LIQUIDADO'].sum()
                efect = (liq / n * 100) if n > 0 else 0
                canal_cohorte_rows.append({
                    'Canal': canal, 'Cohorte': label,
                    'Ordenes': n, 'Liquidadas': int(liq),
                    'Efectividad %': round(efect, 1)
                })
        df_canal_coh = pd.DataFrame(canal_cohorte_rows)

        # Heatmap-style table
        pivot = df_canal_coh.pivot_table(index='Canal', columns='Cohorte', values='Efectividad %', aggfunc='first')
        pivot = pivot.reindex(columns=[c[0] for c in cohortes])

        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(pivot.round(1), use_container_width=True)
        with c2:
            fig_hm = px.imshow(pivot, text_auto='.1f', color_continuous_scale='RdYlGn',
                               range_color=[0, 100], title="Efectividad % por Canal y Cohorte",
                               labels=dict(x="Cohorte", y="Canal", color="Efectividad %"))
            fig_hm.update_layout(height=350)
            st.plotly_chart(aplicar_tema(fig_hm), use_container_width=True, key="coh_canal_hm")

    st.markdown("---")

    # Alertas operativas contextuales
    section_header("Alertas Operativas")
    c1, c2, c3 = st.columns(3)

    with c1:
        df_metro = df[(df['ES_METRO_GUATEMALA']) & (~df['ES_LIQUIDADO'])]
        df_metro_alerta = df_metro[df_metro['EDAD_DIAS'].notna() & (df_metro['EDAD_DIAS'] >= 2)]
        pct = (len(df_metro_alerta) / len(df_metro) * 100) if len(df_metro) > 0 else 0
        st.markdown("**Metro Guatemala 2+ dias**")
        st.metric("Ordenes", f"{len(df_metro_alerta)} ({pct:.0f}%)")
        if len(df_metro_alerta) > 0:
            with st.expander("Detalle"):
                st.dataframe(df_metro_alerta[['ORDEN','FECHA','STATUS','SUB STATUS','EDAD_DIAS']].head(50), height=200)

    with c2:
        df_gen = df[~df['ES_LIQUIDADO']]
        df_gen_alerta = df_gen[df_gen['EDAD_DIAS'].notna() & (df_gen['EDAD_DIAS'] >= 3)]
        pct_g = (len(df_gen_alerta) / len(df_gen) * 100) if len(df_gen) > 0 else 0
        st.markdown("**General 3+ dias sin liquidar**")
        st.metric("Ordenes", f"{len(df_gen_alerta)} ({pct_g:.0f}%)")

    with c3:
        # FIX: Alerta basada en % del total, no monto fijo
        pct_pendiente = (kpis['valor_pendiente'] / kpis['valor_total'] * 100) if kpis['valor_total'] > 0 else 0
        st.markdown("**Valor no liquidado**")
        st.metric("% del total", f"{pct_pendiente:.1f}%", f"Q{kpis['valor_pendiente']:,.0f}")
        if pct_pendiente > 60:
            alert_box(f"Valor pendiente es {pct_pendiente:.0f}% del total. Revisar operacion.", "warning")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(crear_grafico_status(df), use_container_width=True, key="main_status")
    with c2:
        st.plotly_chart(crear_grafico_tendencia(df), use_container_width=True, key="main_tend")


def seccion_reportes(df):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Cliente/Asesor", "Geografico", "Productos", "Intentos y Fallos", "Valor Economico"
    ])

    with tab1:
        reporte_cliente_asesor(df)
    with tab2:
        reporte_geografico(df)
    with tab3:
        reporte_productos(df)
    with tab4:
        reporte_intentos(df)
    with tab5:
        reporte_valor(df)


def reporte_cliente_asesor(df):
    section_header("Efectividad por Cliente")
    ec = df.groupby('CLIENTE').agg(
        total=('ORDEN','count'), liquidados=('ES_LIQUIDADO','sum'),
        valor_total=('VALOR_NUM','sum')
    ).reset_index()
    ec['efectividad'] = (ec['liquidados'] / ec['total'] * 100)
    ec = ec.sort_values('total', ascending=False)

    c1, c2 = st.columns([2,1])
    with c1:
        fig = px.bar(ec, x='CLIENTE', y='efectividad', color='efectividad',
                     color_continuous_scale=[[0, WC['danger']], [0.65, WC['warning']], [1, WC['success']]],
                     range_color=[0,100], title="Efectividad por Cliente",
                     text=[f"{e:.1f}%" for e in ec['efectividad']])
        fig.add_hline(y=65, line_dash="dash", line_color=WC['warning'], annotation_text="Meta 65%")
        st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="ec_bar")
    with c2:
        st.dataframe(ec[['CLIENTE','total','liquidados','efectividad']].round(1), use_container_width=True)
        descargar_csv(ec, "efectividad_cliente.csv", key="dl_ec")

    st.markdown("---")
    section_header("Rendimiento por Asesor (min. 10 ordenes)")

    # FIX: Filtro minimo de volumen para ranking de asesores
    ea = df.groupby('ASESOR').agg(
        total=('ORDEN','count'), liquidados=('ES_LIQUIDADO','sum'),
        promedio_intentos=('NUM_INTENTOS','mean')
    ).reset_index()
    ea['efectividad'] = (ea['liquidados'] / ea['total'] * 100)
    ea['pendientes'] = ea['total'] - ea['liquidados']

    min_ordenes = 10
    ea_filtrado = ea[ea['total'] >= min_ordenes].sort_values('efectividad', ascending=False)
    ea_bajo_volumen = ea[ea['total'] < min_ordenes]

    c1, c2 = st.columns([2,1])
    with c1:
        if len(ea_filtrado) > 0:
            fig = px.bar(ea_filtrado.head(20), x='ASESOR', y=['liquidados','pendientes'],
                         title=f"Ordenes por Asesor (min {min_ordenes} ordenes, Top 20)",
                         barmode='stack', color_discrete_map={'liquidados': WC['success'], 'pendientes': WC['danger']})
            st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="ea_bar")
        else:
            st.info("Ningun asesor alcanza el minimo de ordenes para ranking.")
    with c2:
        st.dataframe(ea_filtrado[['ASESOR','total','efectividad','promedio_intentos']].round(2), use_container_width=True)
        if len(ea_bajo_volumen) > 0:
            st.caption(f"{len(ea_bajo_volumen)} asesores con <{min_ordenes} ordenes excluidos del ranking.")
        descargar_csv(ea, "efectividad_asesor.csv", key="dl_ea")


def reporte_geografico(df):
    section_header("Analisis Geografico")
    c1, c2 = st.columns(2)
    with c1:
        deptos = sorted(df['DEPARTAMENTO'].unique().tolist())
        deptos_sel = st.multiselect("Departamento(s):", deptos, default=[], key="geo_depto")
    with c2:
        if deptos_sel:
            zonas_disp = sorted(df[df['DEPARTAMENTO'].isin(deptos_sel)]['GEO_KEY'].unique().tolist())
        else:
            zonas_disp = sorted(df['GEO_KEY'].unique().tolist())
        zonas_sel = st.multiselect("Zona(s):", zonas_disp, default=[], key="geo_zona")

    df_f = df.copy()
    if deptos_sel:
        df_f = df_f[df_f['DEPARTAMENTO'].isin(deptos_sel)]
    if zonas_sel:
        df_f = df_f[df_f['GEO_KEY'].isin(zonas_sel)]

    if len(df_f) == 0:
        st.warning("Sin datos con los filtros seleccionados.")
        return

    ez = df_f.groupby('GEO_KEY').agg(
        total=('ORDEN','count'), liquidados=('ES_LIQUIDADO','sum'),
        valor_total=('VALOR_NUM','sum')
    ).reset_index()
    ez['efectividad'] = (ez['liquidados'] / ez['total'] * 100)
    ez = ez.sort_values('total', ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        # FIX retro jefa: ordenar por cantidad de ordenes (impacto)
        top = ez.sort_values('total', ascending=True).tail(15)
        colors = [WC['danger'] if e < 65 else WC['success'] for e in top['efectividad']]
        fig = go.Figure(data=[go.Bar(
            y=top['GEO_KEY'], x=top['total'], orientation='h', marker_color=colors,
            text=[f"{int(t)} ordenes | {e:.1f}%" for t, e in zip(top['total'], top['efectividad'])],
            textposition='outside')])
        fig.update_layout(title="Zonas por Volumen de Ordenes (Top 15, color = efectividad)",
                          xaxis_title="Cantidad de Ordenes", height=500, margin=dict(l=180))
        st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="geo_zona_bar")
    with c2:
        # Scatter: volumen vs efectividad
        fig = px.scatter(ez, x='total', y='efectividad', size='valor_total', color='efectividad',
                         color_continuous_scale=[[0, WC['danger']], [0.65, WC['warning']], [1, WC['success']]],
                         range_color=[0,100], hover_name='GEO_KEY',
                         title="Volumen vs Efectividad por Zona",
                         labels={'total':'Ordenes','efectividad':'Efectividad %'})
        fig.add_hline(y=65, line_dash="dash", line_color=WC['warning'])
        st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="geo_scatter")

    # FIX retro jefa: Departamentos ordenados por volumen
    section_header("Efectividad por Departamento (ordenado por volumen)")
    ed = df_f.groupby('DEPARTAMENTO').agg(
        total=('ORDEN','count'), liquidados=('ES_LIQUIDADO','sum'),
        valor_total=('VALOR_NUM','sum'),
        retornos=('STATUS', lambda x: x.isin(['RETORNADO A WEBCORP','RECHAZADO','EN RUTA PARA DEVOLUCION']).sum()),
    ).reset_index()
    ed['efectividad'] = (ed['liquidados'] / ed['total'] * 100)
    # FIX retro jefa: tasa de retorno proporcional, no absoluto
    ed['tasa_retorno'] = (ed['retornos'] / ed['total'] * 100)
    ed = ed.sort_values('total', ascending=False)

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ed.head(15)['DEPARTAMENTO'], y=ed.head(15)['total'],
            name='Ordenes', marker_color=WC['secondary'], opacity=0.7,
            text=ed.head(15)['total'], textposition='outside'
        ))
        fig.add_trace(go.Scatter(
            x=ed.head(15)['DEPARTAMENTO'], y=ed.head(15)['efectividad'],
            name='Efectividad %', mode='lines+markers+text',
            line=dict(color=WC['success'], width=2),
            text=[f"{e:.0f}%" for e in ed.head(15)['efectividad']],
            textposition='top center', yaxis='y2'
        ))
        fig.update_layout(
            title="Departamentos por Volumen + Efectividad",
            yaxis=dict(title="Ordenes"),
            yaxis2=dict(title="Efectividad %", overlaying='y', side='right', range=[0, 100]),
            xaxis_tickangle=-45, height=420,
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="geo_depto_vol")
    with c2:
        st.dataframe(ed[['DEPARTAMENTO','total','efectividad','tasa_retorno']].round(1), use_container_width=True)
        descargar_csv(ed, "efectividad_departamento.csv", key="dl_geo_depto")

    st.dataframe(ez.round(1), use_container_width=True)
    descargar_csv(ez, "efectividad_geografica.csv", key="dl_geo")


def reporte_productos(df):
    section_header("Analisis de Productos (REFERENCIA)")
    ps = df.groupby('PRODUCTO').agg(
        total=('ORDEN','count'), liquidados=('ES_LIQUIDADO','sum'), valor_total=('VALOR_NUM','sum')
    ).reset_index()
    ps['efectividad'] = (ps['liquidados'] / ps['total'] * 100)
    ps = ps.sort_values('total', ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(ps.head(10), x='PRODUCTO', y='efectividad', color='efectividad',
                     color_continuous_scale=[[0, WC['danger']], [0.65, WC['warning']], [1, WC['success']]],
                     range_color=[0,100], title="Efectividad por Producto (Top 10)",
                     text=[f"{e:.1f}%" for e in ps.head(10)['efectividad']])
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="prod_efect")
    with c2:
        # FIX retro jefa: tasa de retorno PROPORCIONAL (% sobre ordenes de ese producto)
        # "30% de mil es igual a 30% de 100" - el que mas ordenes tiene no necesariamente
        # tiene mas retorno proporcionalmente
        section_header("Tasa de Retorno Proporcional")
        st.caption("Porcentaje de retorno sobre las ordenes de CADA producto, no conteo absoluto.")
        fr = df.groupby('PRODUCTO').agg(
            total=('ORDEN','count'),
            retornos=('STATUS', lambda x: x.isin(['RETORNADO A WEBCORP','RECHAZADO','EN RUTA PARA DEVOLUCION']).sum()),
        ).reset_index()
        fr['tasa_retorno'] = (fr['retornos'] / fr['total'] * 100)
        fr = fr[fr['retornos'] > 0].sort_values('tasa_retorno', ascending=False).head(10)
        if len(fr) > 0:
            fig = px.bar(fr, x='PRODUCTO', y='tasa_retorno',
                         title="Tasa de Retorno % (proporcional a ordenes del producto)",
                         color='tasa_retorno', color_continuous_scale='Reds',
                         text=[f"{t:.1f}% ({int(r)}/{int(tot)})" for t, r, tot in zip(fr['tasa_retorno'], fr['retornos'], fr['total'])])
            fig.update_layout(xaxis_tickangle=-45, yaxis_title="Tasa de Retorno %")
            st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="prod_retorno")

    st.dataframe(ps.round(1), use_container_width=True)
    descargar_csv(ps, "analisis_productos.csv", key="dl_prod")


def reporte_intentos(df):
    section_header("Analisis de Intentos de Entrega")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Promedio Intentos", f"{df['NUM_INTENTOS'].mean():.2f}")
    with c2:
        pct_rep = (df['STATUS'] == 'REPROGRAMADO').mean() * 100
        st.metric("% Reprogramadas", f"{pct_rep:.1f}%")
    with c3:
        st.metric("Ordenes >2 intentos", len(df[df['NUM_INTENTOS'] > 2]))
    with c4:
        n_sin = len(df[(df['NUM_INTENTOS'] == 0) & (df['EDAD_DIAS'].notna()) & (df['EDAD_DIAS'] >= 2) & (~df['ES_ENTREGADO_O_LIQUIDADO'])])
        st.metric("0 intentos + 2 dias", n_sin)

    # Curva de conversion por intento (FIX: reemplaza boxplot tautologico)
    section_header("Curva de Conversion por Intento")
    conv = df.groupby('NUM_INTENTOS').agg(
        total=('ORDEN','count'), liquidados=('ES_LIQUIDADO','sum')
    ).reset_index()
    conv['tasa_exito'] = (conv['liquidados'] / conv['total'] * 100)
    conv = conv[conv['NUM_INTENTOS'] <= 5]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=conv['NUM_INTENTOS'], y=conv['total'], name='Ordenes',
                         marker_color=WC['secondary'], opacity=0.6, yaxis='y'))
    fig.add_trace(go.Scatter(x=conv['NUM_INTENTOS'], y=conv['tasa_exito'], name='% Liquidacion',
                             mode='lines+markers+text', line=dict(color=WC['success'], width=3),
                             text=[f"{t:.0f}%" for t in conv['tasa_exito']], textposition='top center', yaxis='y2'))
    fig.update_layout(
        title="Tasa de Liquidacion por Numero de Intentos",
        xaxis_title="Intentos", yaxis=dict(title="Ordenes"),
        yaxis2=dict(title="% Liquidacion", overlaying='y', side='right', range=[0,100]),
        height=400, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="conv_curva")

    # Ordenes 0 intentos y 2+ dias
    df_fantasma = df[(df['NUM_INTENTOS'] == 0) & (df['EDAD_DIAS'].notna()) & (df['EDAD_DIAS'] >= 2) & (~df['ES_ENTREGADO_O_LIQUIDADO'])]
    if len(df_fantasma) > 0:
        alert_box(f"{len(df_fantasma)} ordenes con 0 intentos y 2+ dias en bodega. Estas ordenes no se estan trabajando.", "danger")
        st.dataframe(df_fantasma[['ORDEN','CLIENTE','STATUS','SUB STATUS','EDAD_DIAS','DIRECCION']].head(50), height=250)
        descargar_csv(df_fantasma, "ordenes_0_intentos_2dias.csv", key="dl_fantasma")

    # ================================================================
    # ANTIGUEDAD POR STATUS (retro jefa: "que se entienda")
    # ================================================================
    section_header("Antiguedad de Ordenes por STATUS")
    st.caption(
        "Cada barra/caja muestra cuantos DIAS llevan las ordenes que estan en ese estado. "
        "Si una orden esta en 'EN GESTION' y la caja marca 5-10 dias, significa que esas ordenes "
        "llevan entre 5 y 10 dias sin resolverse. A mayor antiguedad, mayor urgencia."
    )

    df_no_liq = df[~df['ES_LIQUIDADO'] & df['EDAD_DIAS'].notna()].copy()
    if len(df_no_liq) > 0:
        # Tabla resumen clara
        antig = df_no_liq.groupby('STATUS').agg(
            ordenes=('ORDEN','count'),
            dias_promedio=('EDAD_DIAS','mean'),
            dias_minimo=('EDAD_DIAS','min'),
            dias_maximo=('EDAD_DIAS','max'),
            dias_mediana=('EDAD_DIAS','median'),
        ).reset_index().sort_values('dias_promedio', ascending=False)
        antig = antig.round(1)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Resumen: dias que llevan las ordenes en cada estado**")
            st.dataframe(antig, use_container_width=True, hide_index=True)
            prom_general = df_no_liq['EDAD_DIAS'].mean()
            st.metric("Antiguedad Promedio General (no liquidadas)", f"{prom_general:.1f} dias")

        with c2:
            # Bar chart horizontal: promedio de dias por status (mas claro que boxplot)
            antig_sorted = antig.sort_values('dias_promedio', ascending=True)
            colors = [WC['danger'] if d > 7 else WC['warning'] if d > 3 else WC['success']
                      for d in antig_sorted['dias_promedio']]
            fig = go.Figure(data=[go.Bar(
                y=antig_sorted['STATUS'],
                x=antig_sorted['dias_promedio'],
                orientation='h',
                marker_color=colors,
                text=[f"{d:.1f} dias ({int(n)} ordenes)" for d, n in zip(antig_sorted['dias_promedio'], antig_sorted['ordenes'])],
                textposition='outside'
            )])
            fig.update_layout(
                title="Dias promedio que llevan las ordenes en cada estado (no liquidadas)",
                xaxis_title="Dias promedio en el estado",
                height=400, margin=dict(l=200)
            )
            st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="antig_status_bar")
    else:
        st.success("Todas las ordenes estan liquidadas.")


def reporte_valor(df):
    section_header("Analisis de Valor Economico")
    kpis = obtener_kpis_generales(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Valor Total", f"Q{kpis['valor_total']:,.0f}")
    with c2:
        st.metric("Liquidado", f"Q{kpis['valor_liquidado']:,.0f}")
    with c3:
        st.metric("Entregado s/Liq", f"Q{kpis['valor_entregado_sin_liquidar']:,.0f}")
    with c4:
        st.metric("En Ruta", f"Q{kpis['valor_en_ruta']:,.0f}")
    with c5:
        st.metric("Pendiente", f"Q{kpis['valor_pendiente']:,.0f}")

    # Lead Time (FIX: FECHA DEPOSITO ahora se usa)
    section_header("Lead Time: Deposito a Liquidacion")
    df_lt = df[df['LEAD_TIME'].notna() & (df['LEAD_TIME'] >= 0)]
    if len(df_lt) > 0:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Lead Time Promedio", f"{df_lt['LEAD_TIME'].mean():.1f} dias")
            fig = px.histogram(df_lt, x='LEAD_TIME', nbins=15, title="Distribucion de Lead Time (dias)",
                               color_discrete_sequence=[WC['secondary']])
            st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="lt_hist")
        with c2:
            # Lead time vs tasa de rechazo (MKT #2)
            df_lt_bucket = df[df['LEAD_TIME'].notna()].copy()
            df_lt_bucket['LT_BUCKET'] = pd.cut(df_lt_bucket['LEAD_TIME'], bins=[-1, 2, 4, 7, 14, 100],
                                                labels=['0-2d', '3-4d', '5-7d', '8-14d', '15+d'])
            if df_lt_bucket['LT_BUCKET'].notna().any():
                lt_rej = df_lt_bucket.groupby('LT_BUCKET', observed=True).agg(
                    ordenes=('ORDEN','count'),
                    rechazos=('STATUS', lambda x: (x.isin(['RECHAZADO','RETORNADO A WEBCORP'])).sum())
                ).reset_index()
                lt_rej['tasa_rechazo'] = (lt_rej['rechazos'] / lt_rej['ordenes'] * 100)
                fig = px.bar(lt_rej, x='LT_BUCKET', y='tasa_rechazo',
                             title="Tasa de Rechazo por Lead Time",
                             text=[f"{t:.1f}%" for t in lt_rej['tasa_rechazo']],
                             color='tasa_rechazo', color_continuous_scale='Reds')
                st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="lt_rechazo")

    # Resumen ejecutivo
    section_header("Resumen Ejecutivo de Valor")
    df_retornos = df[df['STATUS'].isin(['RETORNADO A WEBCORP','EN RUTA PARA DEVOLUCION','RECHAZADO'])]
    resumen = pd.DataFrame({
        'Concepto': ['Valor Total', 'Liquidado', 'Entregado sin Liquidar', 'En Ruta',
                     'En Gestion', 'Reprogramado', 'Ilocalizable', 'Retorno/Rechazo'],
        'Monto (Q)': [
            kpis['valor_total'], kpis['valor_liquidado'], kpis['valor_entregado_sin_liquidar'],
            kpis['valor_en_ruta'],
            df.loc[df['STATUS']=='EN GESTION','VALOR_NUM'].sum(),
            df.loc[df['STATUS']=='REPROGRAMADO','VALOR_NUM'].sum(),
            df.loc[df['STATUS']=='ILOCALIZABLE','VALOR_NUM'].sum(),
            df_retornos['VALOR_NUM'].sum()
        ]
    })
    resumen['% del Total'] = (resumen['Monto (Q)'] / kpis['valor_total'] * 100).round(1) if kpis['valor_total'] > 0 else 0
    resumen['Monto (Q)'] = resumen['Monto (Q)'].apply(lambda x: f"Q{x:,.0f}")
    st.dataframe(resumen, use_container_width=True)
    descargar_csv(resumen, "resumen_valor.csv", key="dl_resumen_valor")


# ============================================================================
# INTELIGENCIA DE MERCADO (PROPUESTAS MKT VIABLES)
# ============================================================================

def seccion_inteligencia_mercado(df):
    """Modulos de Inteligencia de Mercado implementables con datos actuales."""
    tab1, tab2, tab3, tab4 = st.tabs([
        "White Spaces", "Riesgo Competencia", "Perfil Logistico Cliente", "Penetracion por Asesor"
    ])

    with tab1:
        # MKT #1: White Spaces
        section_header("Mapa de White Spaces: Volumen vs Efectividad")
        st.caption("Zonas con alto volumen + baja efectividad = mercado desatendido. Oportunidad de infraestructura logistica.")
        ws = df.groupby('GEO_KEY').agg(
            volumen=('ORDEN','count'),
            efectividad=('ES_LIQUIDADO','mean'),
            valor_total=('VALOR_NUM','sum')
        ).reset_index()
        ws['valor_perdido'] = df[~df['ES_LIQUIDADO']].groupby('GEO_KEY')['VALOR_NUM'].sum().reindex(ws['GEO_KEY']).fillna(0).values
        ws['efectividad_pct'] = ws['efectividad'] * 100
        ws['score_oportunidad'] = (ws['volumen'].rank(pct=True) * 0.6 + (1 - ws['efectividad']).rank(pct=True) * 0.4)
        ws = ws.sort_values('score_oportunidad', ascending=False)

        fig = px.scatter(ws, x='volumen', y='efectividad_pct', size='valor_perdido',
                         color='score_oportunidad', hover_name='GEO_KEY',
                         color_continuous_scale=[[0, WC['success']], [0.5, WC['warning']], [1, WC['danger']]],
                         title="White Spaces: Zonas con Oportunidad Logistica",
                         labels={'volumen':'Volumen de Ordenes', 'efectividad_pct':'Efectividad %', 'valor_perdido':'Valor Perdido (Q)'})
        fig.add_hline(y=65, line_dash="dash", line_color=WC['success'], annotation_text="Meta 65%")
        st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="ws_scatter")

        top_ws = ws.head(10)
        if len(top_ws) > 0:
            alert_box(
                f"Top oportunidad: {top_ws.iloc[0]['GEO_KEY']} con {int(top_ws.iloc[0]['volumen'])} ordenes, "
                f"{top_ws.iloc[0]['efectividad_pct']:.0f}% efectividad y Q{top_ws.iloc[0]['valor_perdido']:,.0f} en valor perdido.",
                "warning"
            )
        st.dataframe(top_ws[['GEO_KEY','volumen','efectividad_pct','valor_perdido','score_oportunidad']].round(2), use_container_width=True)
        descargar_csv(ws, "white_spaces.csv", key="dl_ws")

    with tab2:
        # MKT #2: Riesgo Competencia
        section_header("Senales de Riesgo Competitivo desde Sub-Status")
        st.caption("Sub-status como 'NO HIZO PEDIDO', 'PRECIO INCORRECTO' o 'COMPRA OTRO PRODUCTO' son senales de mercado, no solo fallos operativos.")
        df_rc = df[df['ES_RIESGO_COMPETENCIA']].copy()
        n_total_rc = len(df_rc)
        pct_rc = (n_total_rc / len(df) * 100) if len(df) > 0 else 0

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Ordenes con senal competitiva", f"{n_total_rc} ({pct_rc:.1f}%)")
            rc_sub = df_rc['SUB STATUS'].value_counts().reset_index()
            rc_sub.columns = ['SUB STATUS', 'Cantidad']
            fig = px.bar(rc_sub, x='SUB STATUS', y='Cantidad', title="Distribucion de Senales Competitivas",
                         color='Cantidad', color_continuous_scale='Reds')
            st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="rc_bar")
        with c2:
            rc_zona = df_rc.groupby('GEO_KEY').size().reset_index(name='incidencias')
            rc_zona = rc_zona.sort_values('incidencias', ascending=False).head(10)
            fig = px.bar(rc_zona, y='GEO_KEY', x='incidencias', orientation='h',
                         title="Top 10 Zonas con Riesgo Competitivo",
                         color='incidencias', color_continuous_scale='Reds')
            st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="rc_zona")

        # Pricing signals (MKT #9)
        df_precio = df[df['SUB STATUS'] == 'PRECIO INCORRECTO']
        if len(df_precio) > 0:
            alert_box(f"{len(df_precio)} ordenes rechazadas por PRECIO INCORRECTO. Revisar pricing en zonas afectadas.", "warning")

    with tab3:
        # MKT #4: Segmentacion por Perfil Logistico
        section_header("Segmentacion de Clientes por Perfil Logistico")
        st.caption("Clasifica clientes por costo de servicio: VIP (bajo costo, alta efectividad), Alto Costo (muchos reintentos), Estandar.")
        pc = df.groupby('CLIENTE').agg(
            ordenes=('ORDEN','count'),
            efectividad=('ES_LIQUIDADO','mean'),
            reprogramaciones=('STATUS', lambda x: (x=='REPROGRAMADO').mean()),
            intentos_promedio=('NUM_INTENTOS','mean'),
            valor_promedio=('VALOR_NUM','mean')
        ).reset_index()

        def clasificar(row):
            if row['efectividad'] > 0.85 and row['intentos_promedio'] < 1.5:
                return 'VIP LOGISTICO'
            elif row['reprogramaciones'] > 0.15 or row['intentos_promedio'] > 2.5:
                return 'ALTO COSTO'
            else:
                return 'ESTANDAR'

        pc['PERFIL'] = pc.apply(clasificar, axis=1)
        color_map = {'VIP LOGISTICO': WC['success'], 'ESTANDAR': WC['secondary'], 'ALTO COSTO': WC['danger']}

        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(pc, x='ordenes', y='efectividad', color='PERFIL', size='valor_promedio',
                             hover_name='CLIENTE', color_discrete_map=color_map,
                             title="Clientes por Perfil Logistico",
                             labels={'ordenes':'Volumen','efectividad':'Efectividad'})
            st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="perfil_scatter")
        with c2:
            perfil_resumen = pc.groupby('PERFIL').agg(
                n_clientes=('CLIENTE','count'),
                ordenes_total=('ordenes','sum'),
                efectividad_media=('efectividad','mean'),
                intentos_media=('intentos_promedio','mean')
            ).reset_index()
            perfil_resumen['efectividad_media'] = (perfil_resumen['efectividad_media'] * 100).round(1)
            st.dataframe(perfil_resumen.round(2), use_container_width=True)

        st.dataframe(pc.round(3), use_container_width=True)
        descargar_csv(pc, "perfil_logistico_clientes.csv", key="dl_perfil")

    with tab4:
        # MKT #6: Penetracion por Asesor/Zona
        section_header("Indice de Penetracion por Zona")
        st.caption("Ordenes por asesor activo. Zonas con baja penetracion = fuerza de ventas ociosa o mercado sin explotar.")
        pen = df.groupby('GEO_KEY').agg(
            ordenes=('ORDEN','count'),
            asesores=('ASESOR','nunique'),
            valor=('VALOR_NUM','sum')
        ).reset_index()
        pen['ordenes_por_asesor'] = (pen['ordenes'] / pen['asesores']).round(1)
        pen['valor_por_asesor'] = (pen['valor'] / pen['asesores']).round(0)
        pen = pen.sort_values('ordenes_por_asesor', ascending=False)

        fig = px.bar(pen.head(20), x='GEO_KEY', y='ordenes_por_asesor',
                     title="Ordenes por Asesor por Zona (Top 20)",
                     color='ordenes_por_asesor',
                     color_continuous_scale=[[0, WC['danger']], [0.5, WC['warning']], [1, WC['success']]],
                     text=[f"{o:.0f}" for o in pen.head(20)['ordenes_por_asesor']])
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="pen_bar")
        st.dataframe(pen.round(1), use_container_width=True)
        descargar_csv(pen, "penetracion_zona.csv", key="dl_pen")


def seccion_filtros_personalizados(df):
    section_header("Dashboard Personalizado")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        clientes = sorted(df['CLIENTE'].dropna().unique().tolist())
        cl_sel = st.multiselect("Cliente:", clientes, default=clientes, key="fp_cl")
    with c2:
        asesores = ['Todos'] + sorted(df['ASESOR'].astype(str).unique().tolist())
        as_sel = st.selectbox("Asesor:", asesores, key="fp_as")
    with c3:
        status_list = sorted(df['STATUS'].dropna().unique().tolist())
        st_sel = st.multiselect("Status:", status_list, default=status_list, key="fp_st")
    with c4:
        deptos = sorted(df['DEPARTAMENTO'].dropna().unique().tolist())
        dp_sel = st.multiselect("Departamento:", deptos, default=deptos, key="fp_dp")

    df_f = df.copy()
    if cl_sel:
        df_f = df_f[df_f['CLIENTE'].isin(cl_sel)]
    if as_sel != 'Todos':
        df_f = df_f[df_f['ASESOR'].astype(str) == as_sel]
    if st_sel:
        df_f = df_f[df_f['STATUS'].isin(st_sel)]
    if dp_sel:
        df_f = df_f[df_f['DEPARTAMENTO'].isin(dp_sel)]

    if len(df_f) == 0:
        st.warning("Sin datos con los filtros seleccionados.")
        return

    kpis = obtener_kpis_generales(df_f)
    st.write(f"Mostrando **{len(df_f):,}** de **{len(df):,}** ordenes")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Ordenes", f"{kpis['total_ordenes']:,}")
    with c2:
        st.metric("Liquidadas", f"{kpis['total_liquidadas']:,}")
    with c3:
        st.metric("Efectividad (3+d)", f"{kpis['efectividad_cohorte']:.1f}%")
    with c4:
        st.metric("Valor", f"Q{kpis['valor_total']:,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(crear_grafico_status(df_f), use_container_width=True, key="fp_status")
    with c2:
        sub = df_f[df_f['SUB STATUS'].notna() & (df_f['SUB STATUS'] != '')]
        if len(sub) > 0:
            sc = sub['SUB STATUS'].value_counts().head(10)
            fig = go.Figure(data=[go.Bar(y=sc.index, x=sc.values, orientation='h',
                                         marker_color=WC['secondary'], text=sc.values, textposition='outside')])
            fig.update_layout(title="Top 10 Sub-Status", height=400, margin=dict(l=220))
            st.plotly_chart(aplicar_tema(fig), use_container_width=True, key="fp_sub")

    st.dataframe(df_f[['ORDEN','CLIENTE','ASESOR','STATUS','SUB STATUS','EDAD_DIAS','VALOR_NUM','DEPARTAMENTO','GEO_KEY']].head(500),
                 use_container_width=True, height=400)
    descargar_csv(df_f, "datos_filtrados.csv", key="dl_fp")


# ============================================================================
# MAIN
# ============================================================================

def main():
    brand_header()

    if 'datos_cargados' not in st.session_state:
        st.session_state.datos_cargados = False
        st.session_state.df = None
        st.session_state.calidad = {}
        st.session_state.n_duplicados = 0

    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding: 8px 0;">
            <span style="font-size:1.3rem; font-weight:700; color:{WC['primary']};">WEBCORP</span>
            <span style="font-size:0.9rem; color:{WC['text_muted']};"> Business</span>
            <br><span style="font-size:0.65rem; color:{WC['text_muted']};">Inteligencia de Negocios</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Carga de Archivos**")
        archivos = st.file_uploader("CSV de seguimiento", type=['csv'], accept_multiple_files=True,
                                     help="Sube uno o mas archivos CSV. Se consolidaran automaticamente.")
        if archivos:
            st.info(f"{len(archivos)} archivo(s)")

        procesar = st.button("Procesar Archivos", type="primary", disabled=not archivos)

        if procesar and archivos:
            errores_estructura = []
            dfs_validos = []
            progress = st.progress(0)

            for i, archivo in enumerate(archivos):
                progress.progress((i+1) / len(archivos))
                try:
                    df_temp = pd.read_csv(archivo, encoding='utf-8-sig')
                    df_temp = normalizar_columnas(df_temp)
                    ok, msg, _ = validar_estructura_csv(df_temp, archivo.name)
                    if not ok:
                        errores_estructura.append(msg)
                        continue
                    dfs_validos.append(df_temp)
                except Exception as e:
                    errores_estructura.append(f"{archivo.name}: Error de lectura - {str(e)}")

            progress.empty()

            if errores_estructura:
                for err in errores_estructura:
                    st.error(err)

            if dfs_validos:
                df_consolidado = pd.concat(dfs_validos, ignore_index=True)

                # FIX: Deduplicacion automatica
                n_antes = len(df_consolidado)
                dupes = df_consolidado.duplicated(subset=['ORDEN'], keep=False)
                if dupes.any():
                    df_consolidado = df_consolidado.drop_duplicates(subset=['ORDEN'], keep='last')
                n_duplicados = n_antes - len(df_consolidado)

                # Modo permisivo: procesar todo
                df_procesado, calidad = procesar_dataframe(df_consolidado)

                st.session_state.df = df_procesado
                st.session_state.datos_cargados = True
                st.session_state.calidad = calidad
                st.session_state.n_duplicados = n_duplicados
                st.session_state.archivos = len(dfs_validos)

                st.success(f"{len(dfs_validos)} archivo(s) | {len(df_procesado):,} registros")
                if n_duplicados > 0:
                    st.warning(f"{n_duplicados} ordenes duplicadas removidas (se conservo la ultima version).")

        # Filtro global de fechas
        if st.session_state.datos_cargados:
            st.markdown("---")
            st.markdown("**Filtros Globales**")
            df = st.session_state.df
            if df['FECHA_DT'].notna().any():
                fmin = df['FECHA_DT'].min().date()
                fmax = df['FECHA_DT'].max().date()
                rango = st.date_input("Rango de fechas:", value=(fmin, fmax),
                                       min_value=fmin, max_value=fmax, key="fg_fecha")
                if len(rango) == 2:
                    st.session_state.filtro_fecha = rango

        st.markdown("---")
        st.markdown(f"""
        <div style="text-align:center; padding: 8px 0;">
            <p style="color:{WC['text_muted']}; font-size:0.72rem;">
                WebCorp Business BI<br>Guatemala
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Contenido principal
    if st.session_state.datos_cargados:
        df = st.session_state.df.copy()

        if hasattr(st.session_state, 'filtro_fecha') and len(st.session_state.filtro_fecha) == 2:
            fi, ff = st.session_state.filtro_fecha
            mask = df['FECHA_DT'].notna() & (df['FECHA_DT'].dt.date >= fi) & (df['FECHA_DT'].dt.date <= ff)
            df = df[mask]

        if len(df) == 0:
            st.warning("Sin datos en el rango seleccionado.")
            return

        # Reporte de calidad
        seccion_calidad_datos(st.session_state.calidad, st.session_state.n_duplicados, 0, len(df))

        # Tabs principales
        t1, t2, t3, t4 = st.tabs([
            "Dashboard Principal", "Reportes Operativos",
            "Inteligencia de Mercado", "Personalizado"
        ])
        with t1:
            seccion_dashboard_principal(df)
        with t2:
            seccion_reportes(df)
        with t3:
            seccion_inteligencia_mercado(df)
        with t4:
            seccion_filtros_personalizados(df)
    else:
        st.markdown(f"""
        <div style="text-align:center; padding: 60px 20px;">
            <h2 style="color:{WC['primary']};">Sistema de Control Logistico e Inteligencia Operativa</h2>
            <p style="color:{WC['text_muted']}; max-width:600px; margin:auto;">
                Sube archivos CSV de seguimiento en el panel lateral para comenzar el analisis.
                El sistema valida, limpia y analiza automaticamente tus datos.
            </p>
        </div>
        """, unsafe_allow_html=True)

        section_header("Columnas requeridas en el CSV")
        cols = st.columns(3)
        for i, col_name in enumerate(COLUMNAS_OBLIGATORIAS):
            cols[i % 3].code(col_name)


if __name__ == "__main__":
    main()
