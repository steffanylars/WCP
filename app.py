"""
Aplicación de Reportería Logística y Control Operativo v2.0
Desarrollada en Streamlit para análisis de datos de seguimiento de entregas
CORREGIDO: Duplicate element IDs, cálculos de efectividad, filtros múltiples
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

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

st.set_page_config(
    page_title="📦 Control Logístico",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card-red {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card-green {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .alert-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CATÁLOGOS OFICIALES
# ============================================================================

COLUMNAS_OBLIGATORIAS = [
    'CLIENTE', 'ASESOR', 'GUIA', 'FECHA', 'REMITENTE', 'DESTINATARIO',
    'DIRECCION', 'TELEFONO', 'COD', 'VALOR', 'ORDEN', 'REFERENCIA',
    'FECHA DEPOSITO', 'STATUS', 'CONTROL INTERNO', 'SUB STATUS',
    'INTENTOS DE ENTREGA', 'REPROGRAMADO'
]

STATUS_VALIDOS = [
    'ENTREGADO LIQUIDADO',
    'ENTREGADO',  # AGREGADO: Status de entregado pero sin liquidar
    'EN RUTA',
    'EN GESTION',
    'REPROGRAMADO',
    'ILOCALIZABLE',
    'RECHAZADO',
    'RECLAMO',
    'FUERA DE COBERTURA',
    'EN RUTA PARA DEVOLUCION',
    'RETORNADO A WEBCORP'
]

SUB_STATUS_VALIDOS = [
    'PUNTO DE ENCUENTRO',
    'ENTREGADO',
    'CONFIRMADO POR CLIENTE',
    'EN GESTION',
    'CONFIRMADO NUEVA FECHA',
    'ALMACENADO',
    'REPROGRAMADO CC',
    'DIRECCIÓN Y TELEFONO ERRONEO',
    'FUERA DE COBERTURA',
    'NO TIENE DINERO',
    'DUPLICADO',
    'NO HIZO PEDIDO',
    'PRECIO INCORRECTO',
    'CAMBIO DE DIRECCIÓN',
    'FUERA DE TIEMPO',
    'NUMERO INCORRECTO',
    'ERROR EN PRODUCTO',
    'RECHAZADO CC',
    'TIEMPO DE ESPERA',
    'COMPRA OTRO PRODUCTO',
    'NADIE EN CASA',
    'ESPERA DE PAGO CARGO',
    'AGENCIA FUERA DE COBERTURA',
    'AGENCIA A PETICION DEL CLIENTE',
    'DIRECTO A AGENCIA',
    'CUMPLIO INTENTOS DE ENTREGA',
    'RETORNO A SOLICITUD DE CC',
    'RETORNADO A WEBCORP'
]

# ============================================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================================

def validar_estructura_csv(df: pd.DataFrame, nombre_archivo: str) -> Tuple[bool, str]:
    """Valida que el DataFrame tenga todas las columnas obligatorias."""
    columnas_base = []
    for col in df.columns:
        col_limpio = col.strip()
        if '.' in col_limpio:
            partes = col_limpio.rsplit('.', 1)
            if partes[1].isdigit():
                col_limpio = partes[0]
        columnas_base.append(col_limpio.upper())
    
    columnas_unicas = set(columnas_base)
    columnas_faltantes = []
    
    for col in COLUMNAS_OBLIGATORIAS:
        if col.upper() not in columnas_unicas:
            columnas_faltantes.append(col)
    
    if columnas_faltantes:
        return False, f"❌ **{nombre_archivo}**: Columnas faltantes: {', '.join(columnas_faltantes)}"
    
    return True, ""


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los nombres de las columnas."""
    df = df.copy()
    
    nuevos_nombres = []
    for col in df.columns:
        nombre = str(col).strip() if col is not None else ''
        nombre = nombre.replace('\xa0', ' ')
        nombre = ' '.join(nombre.split())
        nuevos_nombres.append(nombre)
    
    df.columns = nuevos_nombres
    
    columnas_a_mantener = []
    nombres_vistos = set()
    
    for col in df.columns:
        nombre_base = col
        if '.' in col:
            partes = col.rsplit('.', 1)
            if partes[1].isdigit():
                nombre_base = partes[0]
        
        if nombre_base not in nombres_vistos and nombre_base != '':
            columnas_a_mantener.append(col)
            nombres_vistos.add(nombre_base)
    
    df = df[columnas_a_mantener]
    
    nuevos_nombres = {}
    for col in df.columns:
        if '.' in col:
            partes = col.rsplit('.', 1)
            if partes[1].isdigit():
                nuevos_nombres[col] = partes[0]
    
    if nuevos_nombres:
        df = df.rename(columns=nuevos_nombres)
    
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df.loc[:, df.columns != '']
    
    return df


def parsear_fecha(valor) -> Optional[datetime]:
    """Intenta parsear una fecha en varios formatos comunes"""
    if pd.isna(valor) or str(valor).strip() in ['', '#N/A', 'N/A', 'nan', 'None']:
        return None
    
    valor_str = str(valor).strip()
    formatos = ['%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d-%m-%y']
    
    for fmt in formatos:
        try:
            return datetime.strptime(valor_str, fmt)
        except ValueError:
            continue
    
    return None


def extraer_intentos(valor) -> int:
    """Extrae el número de intentos de entrega del texto"""
    if pd.isna(valor) or str(valor).strip() == '':
        return 0
    
    valor_str = str(valor).upper().strip()
    
    import re
    match = re.search(r'(\d+)\s*INTENTO', valor_str)
    if match:
        return int(match.group(1))
    
    return 0


def extraer_geo_key(direccion: str) -> str:
    """Extrae el tercer componente de la dirección (municipio/zona)"""
    if pd.isna(direccion) or str(direccion).strip() == '':
        return 'DESCONOCIDO'
    
    partes = str(direccion).split(',')
    if len(partes) >= 3:
        return partes[2].strip()
    return 'DESCONOCIDO'


def extraer_departamento(direccion: str) -> str:
    """Extrae el segundo componente de la dirección (departamento)"""
    if pd.isna(direccion) or str(direccion).strip() == '':
        return 'DESCONOCIDO'
    
    partes = str(direccion).split(',')
    if len(partes) >= 2:
        return partes[1].strip()
    return 'DESCONOCIDO'


def extraer_region(direccion: str) -> str:
    """Extrae el primer componente de la dirección (región)"""
    if pd.isna(direccion) or str(direccion).strip() == '':
        return 'DESCONOCIDO'
    
    partes = str(direccion).split(',')
    if len(partes) >= 1:
        return partes[0].strip()
    return 'DESCONOCIDO'


def validar_contenido_fila(row: pd.Series, idx: int) -> List[Dict]:
    """Valida el contenido de una fila y retorna lista de errores encontrados."""
    errores = []
    orden = row.get('ORDEN', '')
    
    if pd.isna(row.get('ORDEN')) or str(row.get('ORDEN', '')).strip() == '':
        errores.append({
            'fila': idx + 2,
            'orden': str(orden),
            'campo': 'ORDEN',
            'razon': 'El campo ORDEN está vacío'
        })
    
    if pd.isna(row.get('DIRECCION')) or str(row.get('DIRECCION', '')).strip() == '':
        errores.append({
            'fila': idx + 2,
            'orden': str(orden),
            'campo': 'DIRECCION',
            'razon': 'El campo DIRECCION está vacío'
        })
    
    status = row.get('STATUS', '')
    if pd.isna(status) or str(status).strip() == '':
        errores.append({
            'fila': idx + 2,
            'orden': str(orden),
            'campo': 'STATUS',
            'razon': 'El campo STATUS está vacío'
        })
    elif str(status).strip().upper() not in [s.upper() for s in STATUS_VALIDOS]:
        errores.append({
            'fila': idx + 2,
            'orden': str(orden),
            'campo': 'STATUS',
            'razon': f'STATUS "{status}" no está en el catálogo válido'
        })
    
    sub_status = row.get('SUB STATUS', '')
    if not pd.isna(sub_status) and str(sub_status).strip() != '':
        if str(sub_status).strip().upper() not in [s.upper() for s in SUB_STATUS_VALIDOS]:
            errores.append({
                'fila': idx + 2,
                'orden': str(orden),
                'campo': 'SUB STATUS',
                'razon': f'SUB STATUS "{sub_status}" no está en el catálogo válido'
            })
    
    fecha = row.get('FECHA', '')
    if pd.isna(fecha) or str(fecha).strip() == '':
        errores.append({
            'fila': idx + 2,
            'orden': str(orden),
            'campo': 'FECHA',
            'razon': 'El campo FECHA está vacío'
        })
    elif parsear_fecha(fecha) is None:
        errores.append({
            'fila': idx + 2,
            'orden': str(orden),
            'campo': 'FECHA',
            'razon': f'FECHA "{fecha}" no es una fecha válida'
        })
    
    fecha_deposito = row.get('FECHA DEPOSITO', '')
    if not pd.isna(fecha_deposito) and str(fecha_deposito).strip() not in ['', '#N/A', 'N/A']:
        if parsear_fecha(fecha_deposito) is None:
            errores.append({
                'fila': idx + 2,
                'orden': str(orden),
                'campo': 'FECHA DEPOSITO',
                'razon': f'FECHA DEPOSITO "{fecha_deposito}" no es una fecha válida'
            })
    
    return errores


def procesar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Procesa el DataFrame añadiendo campos calculados"""
    df = df.copy()
    
    df['FECHA_DT'] = df['FECHA'].apply(parsear_fecha)
    
    hoy = datetime.now()
    df['EDAD_DIAS'] = df['FECHA_DT'].apply(
        lambda x: (hoy - x).days if x is not None else None
    )
    
    # Campos geográficos
    df['GEO_KEY'] = df['DIRECCION'].apply(extraer_geo_key)
    df['DEPARTAMENTO'] = df['DIRECCION'].apply(extraer_departamento)
    df['REGION'] = df['DIRECCION'].apply(extraer_region)
    
    df['NUM_INTENTOS'] = df['INTENTOS DE ENTREGA'].apply(extraer_intentos)
    
    # Flags de estado
    df['ES_LIQUIDADO'] = df['STATUS'].str.upper().str.strip() == 'ENTREGADO LIQUIDADO'
    df['ES_ENTREGADO'] = df['STATUS'].str.upper().str.strip() == 'ENTREGADO'
    df['ES_ENTREGADO_O_LIQUIDADO'] = df['ES_LIQUIDADO'] | df['ES_ENTREGADO']
    
    df['ES_METRO_GUATEMALA'] = df['DIRECCION'].str.startswith('Region Metropolitana,Guatemala,', na=False)
    
    df['VALOR_NUM'] = pd.to_numeric(df['VALOR'], errors='coerce').fillna(0)
    
    return df


# ============================================================================
# FUNCIONES DE MÉTRICAS
# ============================================================================

def calcular_efectividad(df: pd.DataFrame) -> float:
    """Calcula el porcentaje de efectividad (solo ENTREGADO LIQUIDADO)"""
    if len(df) == 0:
        return 0.0
    liquidados = df['ES_LIQUIDADO'].sum()
    return (liquidados / len(df)) * 100


def obtener_kpis_generales(df: pd.DataFrame) -> Dict:
    """Calcula los KPIs generales del dashboard"""
    total = len(df)
    liquidados = int(df['ES_LIQUIDADO'].sum())
    entregados_sin_liquidar = int(df['ES_ENTREGADO'].sum())
    efectividad = calcular_efectividad(df)
    
    valor_liquidado = df[df['ES_LIQUIDADO']]['VALOR_NUM'].sum()
    valor_entregado_sin_liquidar = df[df['ES_ENTREGADO']]['VALOR_NUM'].sum()
    valor_en_ruta = df[df['STATUS'].str.upper() == 'EN RUTA']['VALOR_NUM'].sum()
    valor_pendiente = df[~df['ES_LIQUIDADO']]['VALOR_NUM'].sum()
    
    return {
        'total_ordenes': total,
        'total_liquidadas': liquidados,
        'total_entregadas_sin_liquidar': entregados_sin_liquidar,
        'total_pendientes': total - liquidados,
        'efectividad': efectividad,
        'valor_total': df['VALOR_NUM'].sum(),
        'valor_liquidado': valor_liquidado,
        'valor_entregado_sin_liquidar': valor_entregado_sin_liquidar,
        'valor_en_ruta': valor_en_ruta,
        'valor_pendiente': valor_pendiente
    }


# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def crear_grafico_status(df: pd.DataFrame, key_suffix: str = "") -> go.Figure:
    """Crea gráfico de donut para distribución de STATUS"""
    status_counts = df['STATUS'].value_counts()
    
    colors = px.colors.qualitative.Set3
    
    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.5,
        marker_colors=colors[:len(status_counts)],
        textinfo='percent+label',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Distribución por STATUS",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        height=450
    )
    
    return fig


def crear_grafico_efectividad_zona(df: pd.DataFrame, key_suffix: str = "") -> go.Figure:
    """Crea gráfico de barras de efectividad por zona con barra deslizante"""

    efectividad_zona = df.groupby('GEO_KEY').agg(
        total=('ORDEN', 'count'),
        liquidados=('ES_LIQUIDADO', 'sum')
    ).reset_index()

    efectividad_zona['efectividad'] = (
        efectividad_zona['liquidados'] / efectividad_zona['total']
    ) * 100

    efectividad_zona = efectividad_zona.sort_values(
        'efectividad', ascending=True
    ).reset_index(drop=True)

    n_zonas = len(efectividad_zona)
    window_size = 15  # cuántas barras se muestran a la vez

    start_idx = st.slider(
        "Desliza para ver más zonas",
        min_value=0,
        max_value=max(0, n_zonas - window_size),
        value=0,
        step=1,
        key=f"slider_zona_{key_suffix}"
    )

    zona_view = efectividad_zona.iloc[start_idx:start_idx + window_size]

    colors = [
        '#dc3545' if e < 65 else '#28a745'
        for e in zona_view['efectividad']
    ]

    fig = go.Figure(data=[go.Bar(
        y=zona_view['GEO_KEY'],
        x=zona_view['efectividad'],
        orientation='h',
        marker_color=colors,
        text=[
            f"{e:.1f}% ({int(t)})"
            for e, t in zip(zona_view['efectividad'], zona_view['total'])
        ],
        textposition='outside'
    )])

    fig.add_vline(
        x=65,
        line_dash="dash",
        line_color="orange",
        annotation_text="Meta 65%",
        annotation_position="top"
    )

    fig.update_layout(
        title=f"Efectividad por Zona (mostrando {start_idx + 1}–{min(start_idx + window_size, n_zonas)} de {n_zonas})",
        xaxis_title="Efectividad (%)",
        yaxis_title="Zona",
        height=500,
        margin=dict(l=200)
    )

    return fig


def crear_grafico_tendencia(df: pd.DataFrame, key_suffix: str = "") -> go.Figure:
    """Crea gráfico de tendencia temporal - CORREGIDO para mostrar todos los días"""
    df_fechas = df[df['FECHA_DT'].notna()].copy()
    
    if len(df_fechas) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No hay datos de fecha válidos", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df_fechas['FECHA_SOLO'] = df_fechas['FECHA_DT'].dt.date
    
    # Agrupar por fecha
    tendencia = df_fechas.groupby('FECHA_SOLO').agg(
        total=('ORDEN', 'count'),
        liquidados=('ES_LIQUIDADO', 'sum')
    ).reset_index()
    
    tendencia['efectividad'] = (tendencia['liquidados'] / tendencia['total']) * 100
    tendencia = tendencia.sort_values('FECHA_SOLO')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            name='Total Órdenes', 
            x=tendencia['FECHA_SOLO'], 
            y=tendencia['total'],
            marker_color='#667eea', 
            opacity=0.7,
            text=tendencia['total'],
            textposition='outside'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            name='Efectividad %', 
            x=tendencia['FECHA_SOLO'], 
            y=tendencia['efectividad'],
            mode='lines+markers+text', 
            line=dict(color='#f5576c', width=3),
            text=[f"{e:.0f}%" for e in tendencia['efectividad']],
            textposition='top center'
        ),
        secondary_y=True
    )
    
    fig.add_hline(y=65, line_dash="dash", line_color="green", 
                  annotation_text="Meta 65%", secondary_y=True)
    
    fig.update_layout(
        title=f"Tendencia de Órdenes y Efectividad ({len(tendencia)} días)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat='%d/%m', tickangle=-45)
    )
    
    fig.update_yaxes(title_text="Cantidad de Órdenes", secondary_y=False)
    fig.update_yaxes(title_text="Efectividad (%)", secondary_y=True, range=[0, 100])
    
    return fig


def crear_grafico_sub_status(df: pd.DataFrame, key_suffix: str = "") -> go.Figure:
    """Crea gráfico de barras horizontales para SUB STATUS"""
    df_sub = df[df['SUB STATUS'].notna() & (df['SUB STATUS'] != '')]
    if len(df_sub) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No hay datos de SUB STATUS", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    sub_status_counts = df_sub['SUB STATUS'].value_counts().head(10)
    
    fig = go.Figure(data=[go.Bar(
        y=sub_status_counts.index,
        x=sub_status_counts.values,
        orientation='h',
        marker_color=px.colors.qualitative.Pastel[:len(sub_status_counts)],
        text=sub_status_counts.values,
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Top 10 SUB STATUS",
        xaxis_title="Cantidad",
        yaxis_title="Sub Status",
        height=400,
        margin=dict(l=250)
    )
    
    return fig


def crear_grafico_intentos(df: pd.DataFrame, key_suffix: str = "") -> go.Figure:
    """Crea boxplot de intentos por STATUS"""
    fig = px.box(
        df, 
        x='STATUS', 
        y='NUM_INTENTOS',
        color='STATUS',
        title="Distribución de Intentos por STATUS"
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=450,
        showlegend=False
    )
    
    return fig


def crear_grafico_valor_status(df: pd.DataFrame, key_suffix: str = "") -> go.Figure:
    """Crea gráfico de donut para valor económico por STATUS"""
    valor_status = df.groupby('STATUS')['VALOR_NUM'].sum().reset_index()
    valor_status = valor_status.sort_values('VALOR_NUM', ascending=False)
    
    fig = go.Figure(data=[go.Pie(
        labels=valor_status['STATUS'],
        values=valor_status['VALOR_NUM'],
        hole=0.4,
        textinfo='percent+label',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Valor Económico por STATUS",
        height=450
    )
    
    return fig


# ============================================================================
# FUNCIONES DE DESCARGA
# ============================================================================

def convertir_df_a_csv(df: pd.DataFrame) -> bytes:
    """Convierte DataFrame a bytes CSV para descarga"""
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


def descargar_csv(df: pd.DataFrame, nombre_archivo: str, texto_boton: str = "📥 Descargar CSV", key: str = None):
    """Crea botón de descarga para DataFrame"""
    csv = convertir_df_a_csv(df)
    if key is None:
        key = f"download_{nombre_archivo}_{uuid.uuid4().hex[:8]}"
    st.download_button(
        label=texto_boton,
        data=csv,
        file_name=nombre_archivo,
        mime='text/csv',
        key=key
    )


# ============================================================================
# COMPONENTES DE UI
# ============================================================================

def mostrar_kpi_card(titulo: str, valor: str, subtitulo: str = "", es_efectividad: bool = False, valor_num: float = 0, tipo: str = "normal"):
    """Muestra una tarjeta KPI estilizada"""
    if es_efectividad:
        color_class = "metric-card-green" if valor_num >= 65 else "metric-card-red"
    elif tipo == "warning":
        color_class = "metric-card-orange"
    elif tipo == "success":
        color_class = "metric-card-green"
    elif tipo == "danger":
        color_class = "metric-card-red"
    else:
        color_class = "metric-card"
    
    st.markdown(f"""
    <div class="{color_class}">
        <h3 style="margin:0;font-size:0.9rem;">{titulo}</h3>
        <h1 style="margin:10px 0;font-size:2rem;">{valor}</h1>
        <p style="margin:0;font-size:0.8rem;opacity:0.9;">{subtitulo}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SECCIONES DEL DASHBOARD
# ============================================================================

def seccion_errores(errores: List[Dict]):
    """Muestra la sección de errores de validación"""
    st.error("⚠️ Se encontraron errores de validación. Corrija los datos antes de continuar.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total de errores", len(errores))
    
    with col2:
        tipos_error = {}
        for e in errores:
            campo = e['campo']
            tipos_error[campo] = tipos_error.get(campo, 0) + 1
        
        st.write("**Errores por campo:**")
        for campo, count in sorted(tipos_error.items(), key=lambda x: -x[1]):
            st.write(f"- {campo}: {count}")
    
    st.subheader("Detalle de errores")
    
    df_errores = pd.DataFrame(errores)
    st.dataframe(df_errores, use_container_width=True, height=400)
    
    descargar_csv(df_errores, "errores_de_validacion.csv", key="download_errores")


def seccion_dashboard_principal(df: pd.DataFrame):
    """Muestra el dashboard principal con KPIs"""
    kpis = obtener_kpis_generales(df)
    
    st.subheader("📊 KPIs Generales")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        mostrar_kpi_card("Total Órdenes", f"{kpis['total_ordenes']:,}", "Órdenes cargadas")
    
    with col2:
        mostrar_kpi_card("Liquidadas", f"{kpis['total_liquidadas']:,}", 
                        f"Pendientes: {kpis['total_pendientes']:,}", tipo="success")
    
    with col3:
        mostrar_kpi_card("Entregadas s/Liquidar", f"{kpis['total_entregadas_sin_liquidar']:,}", 
                        f"Q{kpis['valor_entregado_sin_liquidar']:,.0f} por cobrar", tipo="warning")
    
    with col4:
        mostrar_kpi_card("Efectividad", f"{kpis['efectividad']:.1f}%", 
                        "Meta: ≥65%", es_efectividad=True, valor_num=kpis['efectividad'])
    
    with col5:
        mostrar_kpi_card("Valor Liquidado", f"Q{kpis['valor_liquidado']:,.0f}", 
                        f"En ruta: Q{kpis['valor_en_ruta']:,.0f}")
    
    st.markdown("---")
    
    # Alertas operativas
    st.subheader("🚨 Alertas Operativas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Alerta Región Metropolitana Guatemala ≥2 días
        df_metro = df[(df['ES_METRO_GUATEMALA']) & (~df['ES_LIQUIDADO'])]
        df_metro_alerta = df_metro[df_metro['EDAD_DIAS'] >= 2]
        
        total_metro_no_entregado = len(df_metro)
        alerta_metro = len(df_metro_alerta)
        pct_metro = (alerta_metro / total_metro_no_entregado * 100) if total_metro_no_entregado > 0 else 0
        
        st.markdown("#### 🏙️ Metro Guatemala ≥2 días")
        st.metric("Órdenes", f"{alerta_metro} ({pct_metro:.1f}%)")
        
        if alerta_metro > 0:
            with st.expander("Ver detalle"):
                st.dataframe(df_metro_alerta[['ORDEN', 'FECHA', 'STATUS', 'SUB STATUS', 'DIRECCION', 'EDAD_DIAS']], 
                           use_container_width=True, height=200)
                descargar_csv(
                    df_metro_alerta[['ORDEN', 'FECHA', 'STATUS', 'SUB STATUS', 'DIRECCION']],
                    "alerta_metro_guatemala.csv",
                    key="download_metro_alerta"
                )
    
    with col2:
        # Alerta General ≥3 días
        df_general = df[~df['ES_LIQUIDADO']]
        df_general_alerta = df_general[df_general['EDAD_DIAS'] >= 3]
        
        total_no_entregado = len(df_general)
        alerta_general = len(df_general_alerta)
        pct_general = (alerta_general / total_no_entregado * 100) if total_no_entregado > 0 else 0
        
        st.markdown("#### 🌎 General ≥3 días")
        st.metric("Órdenes", f"{alerta_general} ({pct_general:.1f}%)")
        
        if alerta_general > 0:
            with st.expander("Ver detalle"):
                st.dataframe(df_general_alerta[['ORDEN', 'FECHA', 'STATUS', 'SUB STATUS', 'DIRECCION', 'EDAD_DIAS']], 
                           use_container_width=True, height=200)
                descargar_csv(
                    df_general_alerta[['ORDEN', 'FECHA', 'STATUS', 'SUB STATUS', 'DIRECCION']],
                    "alerta_general.csv",
                    key="download_general_alerta"
                )
    
    with col3:
        # Entregados sin liquidar
        df_entregados_sin_liq = df[df['ES_ENTREGADO']]
        valor_sin_liq = df_entregados_sin_liq['VALOR_NUM'].sum()
        
        st.markdown("#### 💰 Entregados sin Liquidar")
        st.metric("Órdenes", f"{len(df_entregados_sin_liq)}")
        st.metric("Valor", f"Q{valor_sin_liq:,.0f}")
        
        if len(df_entregados_sin_liq) > 0:
            with st.expander("Ver detalle"):
                st.dataframe(df_entregados_sin_liq[['ORDEN', 'FECHA', 'CLIENTE', 'VALOR_NUM', 'DIRECCION']], 
                           use_container_width=True, height=200)
                descargar_csv(
                    df_entregados_sin_liq[['ORDEN', 'FECHA', 'CLIENTE', 'VALOR_NUM', 'DIRECCION', 'STATUS']],
                    "entregados_sin_liquidar.csv",
                    key="download_entregados_sin_liq"
                )
    
    st.markdown("---")
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        fig_status = crear_grafico_status(df)
        st.plotly_chart(fig_status, use_container_width=True, key="chart_status_main")
    
    with col2:
        fig_tendencia = crear_grafico_tendencia(df)
        st.plotly_chart(fig_tendencia, use_container_width=True, key="chart_tendencia_main")


def seccion_reportes(df: pd.DataFrame):
    """Muestra los reportes operativos y estratégicos"""
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Por Cliente/Asesor",
        "🗺️ Geográfico", 
        "📦 Productos",
        "🔄 Intentos y Fallos",
        "💰 Valor Económico",
        "📅 Tendencias"
    ])
    
    with tab1:
        reporte_cliente_asesor(df)
    
    with tab2:
        reporte_geografico(df)
    
    with tab3:
        reporte_productos(df)
    
    with tab4:
        reporte_intentos_fallos(df)
    
    with tab5:
        reporte_valor_economico(df)
    
    with tab6:
        reporte_tendencias(df)


def reporte_cliente_asesor(df: pd.DataFrame):
    """Reporte de rendimiento por cliente y asesor"""
    
    st.subheader("📊 Efectividad por Cliente")
    

    #solo liquidados
    efectividad_cliente = df.groupby('CLIENTE').agg(
        total=('ORDEN', 'count'),
        liquidados=('ES_LIQUIDADO', 'sum'),
        valor_total=('VALOR_NUM', 'sum')
    ).reset_index()
    
    efectividad_cliente['efectividad'] = (efectividad_cliente['liquidados'] / efectividad_cliente['total']) * 100
    efectividad_cliente['pendientes'] = efectividad_cliente['total'] - efectividad_cliente['liquidados']
    efectividad_cliente = efectividad_cliente.sort_values('total', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            efectividad_cliente,
            x='CLIENTE',
            y='efectividad',
            color='efectividad',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[0, 100],
            title="Efectividad por Cliente",
            text=[f"{e:.1f}%" for e in efectividad_cliente['efectividad']]
        )
        fig.add_hline(y=65, line_dash="dash", line_color="orange", annotation_text="Meta 65%")
        st.plotly_chart(fig, use_container_width=True, key="chart_efect_cliente")
    
    with col2:
        st.dataframe(efectividad_cliente[['CLIENTE', 'total', 'liquidados', 'efectividad']], 
                    use_container_width=True)
        descargar_csv(efectividad_cliente, "efectividad_cliente.csv", key="download_efect_cliente")
    
    st.markdown("---")
    
    st.subheader("👤 Rendimiento por Asesor")
    
    
    efectividad_asesor = df.groupby('ASESOR').agg(
        total=('ORDEN', 'count'),
        liquidados=('ES_LIQUIDADO', 'sum'),
        promedio_intentos=('NUM_INTENTOS', 'mean')
    ).reset_index()
    
    efectividad_asesor['efectividad'] = (efectividad_asesor['liquidados'] / efectividad_asesor['total']) * 100
    efectividad_asesor['pendientes'] = efectividad_asesor['total'] - efectividad_asesor['liquidados']
    efectividad_asesor = efectividad_asesor.sort_values('efectividad', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            efectividad_asesor.head(20),
            x='ASESOR',
            y=['liquidados', 'pendientes'],
            title="Órdenes por Asesor (Top 20)",
            barmode='stack',
            color_discrete_map={'liquidados': '#28a745', 'pendientes': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_ordenes_asesor")
    
    with col2:
        st.dataframe(efectividad_asesor[['ASESOR', 'total', 'efectividad', 'promedio_intentos']], 
                    use_container_width=True)
        descargar_csv(efectividad_asesor, "efectividad_asesor.csv", key="download_efect_asesor")


def reporte_geografico(df: pd.DataFrame):
    """Reporte geográfico por zona - CORREGIDO con filtros múltiples"""
    
    st.subheader("🗺️ Análisis Geográfico")
    
    # Filtros múltiples
    col1, col2 = st.columns(2)
    
    with col1:
        departamentos = sorted(df['DEPARTAMENTO'].unique().tolist())
        deptos_seleccionados = st.multiselect(
            "Filtrar por Departamento(s):",
            options=departamentos,
            default=[],
            key="filtro_departamentos"
        )
    
    with col2:
        if deptos_seleccionados:
            zonas_disponibles = sorted(df[df['DEPARTAMENTO'].isin(deptos_seleccionados)]['GEO_KEY'].unique().tolist())
        else:
            zonas_disponibles = sorted(df['GEO_KEY'].unique().tolist())
        
        zonas_seleccionadas = st.multiselect(
            "Filtrar por Zona(s):",
            options=zonas_disponibles,
            default=[],
            key="filtro_zonas"
        )
    
    # Aplicar filtros
    df_filtrado = df.copy()
    if deptos_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['DEPARTAMENTO'].isin(deptos_seleccionados)]
    if zonas_seleccionadas:
        df_filtrado = df_filtrado[df_filtrado['GEO_KEY'].isin(zonas_seleccionadas)]
    
    if len(df_filtrado) == 0:
        st.warning("No hay datos con los filtros seleccionados.")
        return
    
    # Calcular efectividad por zona - CORREGIDO
    efectividad_zona = df_filtrado.groupby('GEO_KEY').agg(
        total=('ORDEN', 'count'),
        liquidados=('ES_LIQUIDADO', 'sum'),
        valor_total=('VALOR_NUM', 'sum'),
        promedio_intentos=('NUM_INTENTOS', 'mean')
    ).reset_index()
    
    efectividad_zona['efectividad'] = (efectividad_zona['liquidados'] / efectividad_zona['total']) * 100
    efectividad_zona['pendientes'] = efectividad_zona['total'] - efectividad_zona['liquidados']
    efectividad_zona = efectividad_zona.sort_values('total', ascending=False)
    
    # Efectividad por departamento
    efectividad_depto = df_filtrado.groupby('DEPARTAMENTO').agg(
        total=('ORDEN', 'count'),
        liquidados=('ES_LIQUIDADO', 'sum'),
        valor_total=('VALOR_NUM', 'sum')
    ).reset_index()
    
    efectividad_depto['efectividad'] = (efectividad_depto['liquidados'] / efectividad_depto['total']) * 100
    efectividad_depto = efectividad_depto.sort_values('total', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de efectividad por zona
        top_zonas = efectividad_zona.sort_values('efectividad', ascending=True).tail(15)
        colors = ['#dc3545' if e < 65 else '#28a745' for e in top_zonas['efectividad']]
        
        fig = go.Figure(data=[go.Bar(
            y=top_zonas['GEO_KEY'],
            x=top_zonas['efectividad'],
            orientation='h',
            marker_color=colors,
            text=[f"{e:.1f}% (n={int(t)})" for e, t in zip(top_zonas['efectividad'], top_zonas['total'])],
            textposition='outside'
        )])
        
        fig.add_vline(x=65, line_dash="dash", line_color="orange", annotation_text="Meta 65%")
        fig.update_layout(
            title="Efectividad por Zona",
            xaxis_title="Efectividad (%)",
            height=500,
            margin=dict(l=200)
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_efect_zona")
    
    with col2:
        # Treemap
        fig = px.treemap(
            efectividad_zona,
            path=['GEO_KEY'],
            values='total',
            color='efectividad',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[0, 100],
            title="Mapa de Calor por Zona"
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_treemap_zona")
    
    # Efectividad por departamento
    st.subheader("📍 Efectividad por Departamento")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(
            efectividad_depto.head(15),
            x='DEPARTAMENTO',
            y='efectividad',
            color='efectividad',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[0, 100],
            title="Efectividad por Departamento",
            text=[f"{e:.1f}%" for e in efectividad_depto.head(15)['efectividad']]
        )
        fig.add_hline(y=65, line_dash="dash", line_color="orange")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key="chart_efect_depto")
    
    with col2:
        st.dataframe(efectividad_depto, use_container_width=True)
        descargar_csv(efectividad_depto, "efectividad_departamento.csv", key="download_efect_depto")
    

def reporte_productos(df: pd.DataFrame):
    """Reporte de análisis de productos"""
    
    st.subheader("📦 Análisis de Productos (REFERENCIA)")
    
    df_productos = df.copy()
    df_productos['PRODUCTO'] = df_productos['REFERENCIA'].apply(
        lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'SIN REFERENCIA'
    )
    
    productos_stats = df_productos.groupby('PRODUCTO').agg(
        total=('ORDEN', 'count'),
        liquidados=('ES_LIQUIDADO', 'sum'),
        valor_total=('VALOR_NUM', 'sum')
    ).reset_index()
    
    productos_stats['efectividad'] = (productos_stats['liquidados'] / productos_stats['total']) * 100
    productos_stats['valor_liquidado'] = df_productos[df_productos['ES_LIQUIDADO']].groupby(
        df_productos[df_productos['ES_LIQUIDADO']]['PRODUCTO']
    )['VALOR_NUM'].sum().reindex(productos_stats['PRODUCTO']).fillna(0).values
    productos_stats['valor_pendiente'] = productos_stats['valor_total'] - productos_stats['valor_liquidado']
    productos_stats = productos_stats.sort_values('total', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            productos_stats.head(10),
            values='total',
            names='PRODUCTO',
            title="Top 10 Productos por Cantidad",
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_pie_productos")
    
    with col2:
        fig = px.bar(
            productos_stats.head(10),
            x='PRODUCTO',
            y='efectividad',
            color='efectividad',
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[0, 100],
            title="Efectividad por Producto (Top 10)",
            text=[f"{e:.1f}%" for e in productos_stats.head(10)['efectividad']]
        )
        fig.add_hline(y=65, line_dash="dash", line_color="orange")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key="chart_efect_productos")
    
    # Productos con más retornos
    st.subheader("⚠️ Productos con más Retornos/Rechazos")
    
    df_retornos = df_productos[df_productos['STATUS'].isin(['RETORNADO A WEBCORP', 'EN RUTA PARA DEVOLUCION', 'RECHAZADO'])]
    
    if len(df_retornos) > 0:
        retornos_producto = df_retornos.groupby('PRODUCTO').size().reset_index(name='retornos')
        retornos_producto = retornos_producto.sort_values('retornos', ascending=False).head(10)
        
        fig = px.bar(
            retornos_producto,
            x='PRODUCTO',
            y='retornos',
            title="Top 10 Productos con más Retornos",
            color='retornos',
            color_continuous_scale='Reds',
            text='retornos'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key="chart_retornos_productos")
    else:
        st.info("No hay retornos registrados.")
    
    st.subheader("📋 Tabla de Productos")
    st.dataframe(productos_stats, use_container_width=True)
    descargar_csv(productos_stats, "analisis_productos.csv", key="download_productos")


def reporte_intentos_fallos(df: pd.DataFrame):
    """Reporte de intentos y razones de fallo"""
    
    st.subheader("🔄 Análisis de Intentos de Entrega")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        promedio_intentos = df['NUM_INTENTOS'].mean()
        st.metric("Promedio de Intentos", f"{promedio_intentos:.2f}")
    
    with col2:
        ordenes_reprogramadas = len(df[df['STATUS'] == 'REPROGRAMADO'])
        pct_reprogramadas = (ordenes_reprogramadas / len(df)) * 100
        st.metric("% Reprogramadas", f"{pct_reprogramadas:.1f}%")
    
    with col3:
        ordenes_mas_2_intentos = len(df[df['NUM_INTENTOS'] > 2])
        st.metric("Órdenes >2 Intentos", ordenes_mas_2_intentos)
    
    with col4:
        ordenes_0_intentos = len(df[df['NUM_INTENTOS'] == 0])
        st.metric("Sin Intentos (0)", ordenes_0_intentos)
    
    col1, col2 = st.columns(2)
    
    with col1:
        intentos_status = df.groupby('STATUS')['NUM_INTENTOS'].mean().reset_index()
        intentos_status = intentos_status.sort_values('NUM_INTENTOS', ascending=True)
        
        fig = px.bar(
            intentos_status,
            y='STATUS',
            x='NUM_INTENTOS',
            orientation='h',
            title="Promedio de Intentos por STATUS",
            color='NUM_INTENTOS',
            color_continuous_scale='Oranges',
            text=[f"{i:.2f}" for i in intentos_status['NUM_INTENTOS']]
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_intentos_status")
    
    with col2:
        fig = crear_grafico_intentos(df)
        st.plotly_chart(fig, use_container_width=True, key="chart_boxplot_intentos")
    
    # SUB STATUS con más intentos
    st.subheader("📊 SUB STATUS con más Intentos")
    
    df_sub = df[df['SUB STATUS'].notna() & (df['SUB STATUS'] != '')]
    if len(df_sub) > 0:
        intentos_sub = df_sub.groupby('SUB STATUS').agg(
            promedio_intentos=('NUM_INTENTOS', 'mean'),
            total_ordenes=('ORDEN', 'count')
        ).reset_index()
        
        intentos_sub = intentos_sub[intentos_sub['promedio_intentos'] > 1].sort_values('promedio_intentos', ascending=False)
        
        if len(intentos_sub) > 0:
            fig = px.bar(
                intentos_sub.head(10),
                x='SUB STATUS',
                y='promedio_intentos',
                title="SUB STATUS con Promedio >1 Intento",
                color='total_ordenes',
                color_continuous_scale='Blues',
                text=[f"{i:.2f}" for i in intentos_sub.head(10)['promedio_intentos']]
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key="chart_sub_intentos")
    
    # Órdenes con >2 intentos
    st.subheader("⚠️ Órdenes con más de 2 Intentos")
    
    df_muchos_intentos = df[df['NUM_INTENTOS'] > 2]
    
    if len(df_muchos_intentos) > 0:
        st.write(f"Total: **{len(df_muchos_intentos)}** órdenes")
        st.dataframe(
            df_muchos_intentos[['ORDEN', 'CLIENTE', 'STATUS', 'SUB STATUS', 'NUM_INTENTOS', 'DIRECCION']],
            use_container_width=True,
            height=300
        )
        descargar_csv(df_muchos_intentos, "ordenes_mas_2_intentos.csv", key="download_muchos_intentos")
    else:
        st.info("No hay órdenes con más de 2 intentos.")

    # Órdenes con 0 intentos
    st.subheader("⚠️ Órdenes con 0 Intentos y más de 2 días en bodega")
    #HOLIWIS REGRESAR
    df_no_intentos_muchos_dias = df[(df['NUM_INTENTOS'] == 0) & (df['EDAD_DIAS'] >= 2) & (df['STATUS'] != "ENTREGADO")& (df['STATUS'] != "ENTREGADO LIQUIDADO")]

    if len(df_no_intentos_muchos_dias) > 0:
        st.write(f"Total: **{len(df_no_intentos_muchos_dias)}** órdenes")
        st.dataframe(
            df_no_intentos_muchos_dias[['ORDEN', 'CLIENTE', 'STATUS', 'SUB STATUS', 'NUM_INTENTOS','EDAD_DIAS', 'DIRECCION']],
            use_container_width=True,
            height=300
        )
        descargar_csv(df_no_intentos_muchos_dias, "ordenes_0_intentos_+2_dias.csv", key="df_no_intentos_muchos_dias")
    else:
        st.info("No hay órdenes con más de 2 intentos.")

def reporte_valor_economico(df: pd.DataFrame):
    """Reporte de valor económico - EXTENDIDO"""
    
    st.subheader("💰 Análisis de Valor Económico")
    
    kpis = obtener_kpis_generales(df)
    
    # KPIs principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💵 Valor Total", f"Q{kpis['valor_total']:,.0f}")
    
    with col2:
        st.metric("✅ Valor Liquidado", f"Q{kpis['valor_liquidado']:,.0f}")
    
    with col3:
        st.metric("📦 Entregado s/Liquidar", f"Q{kpis['valor_entregado_sin_liquidar']:,.0f}")
    
    with col4:
        st.metric("🚚 Valor en Ruta", f"Q{kpis['valor_en_ruta']:,.0f}")
    
    with col5:
        st.metric("⏳ Valor No Entregado/Sin Liquidar", f"Q{kpis['valor_pendiente']:,.0f}")
    
    # Alerta de valor pendiente alto
    if kpis['valor_pendiente'] > 10000:
        st.warning(f"⚠️ **ALERTA**: Valor de 'No Entregados' superior a Q10,000 (Q{kpis['valor_pendiente']:,.0f})")
    
    st.markdown("---")
    
    # Análisis detallado
    col1, col2 = st.columns(2)
    
    with col1:
        fig = crear_grafico_valor_status(df)
        st.plotly_chart(fig, use_container_width=True, key="chart_valor_status")
    
    with col2:
        # Valor por zona
        valor_zona = df.groupby('GEO_KEY').agg(
            valor_total=('VALOR_NUM', 'sum'),
            valor_liquidado=('VALOR_NUM', lambda x: x[df.loc[x.index, 'ES_LIQUIDADO']].sum() if len(x) > 0 else 0)
        ).reset_index()
        valor_zona['valor_pendiente'] = valor_zona['valor_total'] - valor_zona['valor_liquidado']
        valor_zona = valor_zona.sort_values('valor_total', ascending=False).head(10)
        
        fig = px.bar(
            valor_zona,
            x='GEO_KEY',
            y=['valor_liquidado', 'valor_pendiente'],
            title="Valor por Zona (Top 10)",
            barmode='stack',
            color_discrete_map={'valor_liquidado': '#28a745', 'valor_entregado_sin_liquidar': '#dc3545'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True, key="chart_valor_zona")
    
    st.markdown("---")
    
    # Sección: Entregados sin Liquidar (NUEVO)
    st.subheader("📦 Detalle: Entregados sin Liquidar")
    
    df_entregados = df[df['ES_ENTREGADO']].copy()
    
    if len(df_entregados) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Órdenes", len(df_entregados))
        with col2:
            st.metric("Valor Total", f"Q{df_entregados['VALOR_NUM'].sum():,.0f}")
        with col3:
            promedio = df_entregados['VALOR_NUM'].mean()
            st.metric("Valor Promedio", f"Q{promedio:,.0f}")
        
        # Por cliente
        entregados_cliente = df_entregados.groupby('CLIENTE').agg(
            cantidad=('ORDEN', 'count'),
            valor=('VALOR_NUM', 'sum')
        ).reset_index().sort_values('valor', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                entregados_cliente,
                values='valor',
                names='CLIENTE',
                title="Valor Entregado sin Liquidar por Cliente",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_entregados_cliente")
        
        with col2:
            st.dataframe(entregados_cliente, use_container_width=True)
        
        st.subheader("📋 Listado de Órdenes Entregadas sin Liquidar")
        st.dataframe(
            df_entregados[['ORDEN', 'CLIENTE', 'FECHA', 'VALOR_NUM', 'DESTINATARIO', 'DIRECCION', 'ASESOR']],
            use_container_width=True,
            height=300
        )
        descargar_csv(df_entregados, "entregados_sin_liquidar_detalle.csv", key="download_entregados_detalle")
    else:
        st.success("✅ No hay órdenes entregadas pendientes de liquidar.")
    
    st.markdown("---")
    
    # Valor en retornos
    st.subheader("💸 Valor en Retornos y Rechazos")
    
    df_retornos = df[df['STATUS'].isin(['RETORNADO A WEBCORP', 'EN RUTA PARA DEVOLUCION', 'RECHAZADO'])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(df_retornos) > 0:
            valor_retornos = df_retornos['VALOR_NUM'].sum()
            st.metric("Valor Total en Retornos/Rechazos", f"Q{valor_retornos:,.0f}")
            st.metric("Órdenes en Retorno/Rechazo", len(df_retornos))
            
            retornos_status = df_retornos.groupby('STATUS')['VALOR_NUM'].sum().reset_index()
            fig = px.pie(
                retornos_status,
                values='VALOR_NUM',
                names='STATUS',
                title="Distribución del Valor en Retornos",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_valor_retornos")
        else:
            st.success("✅ No hay órdenes en retorno.")
    
    with col2:
        if len(df_retornos) > 0:
            st.dataframe(
                df_retornos[['ORDEN', 'CLIENTE', 'STATUS', 'VALOR_NUM', 'GEO_KEY']],
                use_container_width=True,
                height=300
            )
            descargar_csv(df_retornos, "ordenes_retorno_valor.csv", key="download_retornos_valor")
    
    st.markdown("---")
    
    # Análisis de recuperación potencial
    st.subheader("📈 Análisis de Recuperación Potencial")
    
    col1, col2, col3 = st.columns(3)
    
    # En Gestión
    df_en_gestion = df[df['STATUS'] == 'EN GESTION']
    with col1:
        st.markdown("**En Gestión**")
        st.metric("Órdenes", len(df_en_gestion))
        st.metric("Valor Potencial", f"Q{df_en_gestion['VALOR_NUM'].sum():,.0f}")
    
    # Reprogramados
    df_reprogramados = df[df['STATUS'] == 'REPROGRAMADO']
    with col2:
        st.markdown("**Reprogramados**")
        st.metric("Órdenes", len(df_reprogramados))
        st.metric("Valor Potencial", f"Q{df_reprogramados['VALOR_NUM'].sum():,.0f}")
    
    # Ilocalizables
    df_ilocalizables = df[df['STATUS'] == 'ILOCALIZABLE']
    with col3:
        st.markdown("**Ilocalizables**")
        st.metric("Órdenes", len(df_ilocalizables))
        st.metric("Valor en Riesgo", f"Q{df_ilocalizables['VALOR_NUM'].sum():,.0f}")
    
    # Resumen ejecutivo
    st.markdown("---")
    st.subheader("📊 Resumen Ejecutivo de Valor")
    
    resumen = pd.DataFrame({
        'Concepto': [
            'Valor Total en Sistema',
            'Valor Liquidado (Cobrado)',
            'Valor Entregado sin Liquidar',
            'Valor en Ruta',
            'Valor en Gestión',
            'Valor Reprogramado',
            'Valor Ilocalizable',
            'Valor en Retorno/Rechazo'
        ],
        'Monto (Q)': [
            kpis['valor_total'],
            kpis['valor_liquidado'],
            kpis['valor_entregado_sin_liquidar'],
            kpis['valor_en_ruta'],
            df_en_gestion['VALOR_NUM'].sum(),
            df_reprogramados['VALOR_NUM'].sum(),
            df_ilocalizables['VALOR_NUM'].sum(),
            df_retornos['VALOR_NUM'].sum() if len(df_retornos) > 0 else 0
        ]
    })
    
    resumen['% del Total'] = (resumen['Monto (Q)'] / kpis['valor_total'] * 100).round(1)
    resumen['Monto (Q)'] = resumen['Monto (Q)'].apply(lambda x: f"Q{x:,.0f}")
    
    st.dataframe(resumen, use_container_width=True)
    descargar_csv(resumen, "resumen_valor_economico.csv", key="download_resumen_valor")


def reporte_tendencias(df: pd.DataFrame):
    """Reporte de tendencias temporales - CORREGIDO"""
    
    st.subheader("📅 Tendencias Temporales")
    
    df_fechas = df[df['FECHA_DT'].notna()].copy()
    
    if len(df_fechas) == 0:
        st.warning("No hay datos de fecha válidos para mostrar tendencias.")
        return
    
    df_fechas['FECHA_SOLO'] = df_fechas['FECHA_DT'].dt.date
    df_fechas['SEMANA'] = df_fechas['FECHA_DT'].dt.isocalendar().week
    df_fechas['DIA_SEMANA'] = df_fechas['FECHA_DT'].dt.day_name()
    
    # Mostrar rango de fechas
    fecha_min = df_fechas['FECHA_SOLO'].min()
    fecha_max = df_fechas['FECHA_SOLO'].max()
    dias_unicos = df_fechas['FECHA_SOLO'].nunique()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fecha más antigua", fecha_min.strftime('%d/%m/%Y'))
    with col2:
        st.metric("Fecha más reciente", fecha_max.strftime('%d/%m/%Y'))
    with col3:
        st.metric("Días con órdenes", dias_unicos)
    
    # Tendencia diaria - CORREGIDO
    st.subheader("📈 Tendencia Diaria")
    
    tendencia = df_fechas.groupby('FECHA_SOLO').agg(
        total=('ORDEN', 'count'),
        liquidados=('ES_LIQUIDADO', 'sum')
    ).reset_index()
    tendencia['efectividad'] = (tendencia['liquidados'] / tendencia['total']) * 100
    tendencia = tendencia.sort_values('FECHA_SOLO')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            name='Total Órdenes', 
            x=tendencia['FECHA_SOLO'], 
            y=tendencia['total'],
            marker_color='#667eea', 
            opacity=0.7,
            text=tendencia['total'],
            textposition='outside'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            name='Efectividad %', 
            x=tendencia['FECHA_SOLO'], 
            y=tendencia['efectividad'],
            mode='lines+markers+text', 
            line=dict(color='#f5576c', width=3),
            text=[f"{e:.0f}%" for e in tendencia['efectividad']],
            textposition='top center'
        ),
        secondary_y=True
    )
    
    fig.add_hline(y=65, line_dash="dash", line_color="green", 
                  annotation_text="Meta 65%", secondary_y=True)
    
    fig.update_layout(
        title=f"Órdenes y Efectividad por Día ({len(tendencia)} días)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickformat='%d/%m', tickangle=-45)
    )
    
    fig.update_yaxes(title_text="Cantidad de Órdenes", secondary_y=False)
    fig.update_yaxes(title_text="Efectividad (%)", secondary_y=True, range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True, key="chart_tendencia_diaria_report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Efectividad semanal
        semanal = df_fechas.groupby('SEMANA').agg(
            total=('ORDEN', 'count'),
            liquidados=('ES_LIQUIDADO', 'sum')
        ).reset_index()
        semanal['efectividad'] = (semanal['liquidados'] / semanal['total']) * 100
        
        fig = px.line(
            semanal,
            x='SEMANA',
            y='efectividad',
            title="Efectividad por Semana",
            markers=True,
            text=[f"{e:.0f}%" for e in semanal['efectividad']]
        )
        fig.add_hline(y=65, line_dash="dash", line_color="red", annotation_text="Meta")
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True, key="chart_efectividad_semanal")
    
    with col2:
        # Órdenes por día de la semana
        dia_semana = df_fechas.groupby('DIA_SEMANA').agg(
            total=('ORDEN', 'count'),
            liquidados=('ES_LIQUIDADO', 'sum')
        ).reset_index()
        dia_semana['efectividad'] = (dia_semana['liquidados'] / dia_semana['total']) * 100
        
        # Ordenar días
        orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dia_semana['orden'] = dia_semana['DIA_SEMANA'].apply(lambda x: orden_dias.index(x) if x in orden_dias else 7)
        dia_semana = dia_semana.sort_values('orden')
        
        fig = px.bar(
            dia_semana,
            x='DIA_SEMANA',
            y='total',
            title="Órdenes por Día de la Semana",
            color='efectividad',
            color_continuous_scale=['red', 'yellow', 'green'],
            text='total'
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_dia_semana")
    
    # Antigüedad de no entregados
    st.subheader("⏰ Antigüedad de Órdenes No Liquidadas")
    
    df_no_liquidados = df[~df['ES_LIQUIDADO']]
    
    if len(df_no_liquidados) > 0:
        promedio_antiguedad = df_no_liquidados['EDAD_DIAS'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Antigüedad Promedio", f"{promedio_antiguedad:.1f} días")
            
            fig = px.histogram(
                df_no_liquidados,
                x='EDAD_DIAS',
                title="Distribución de Antigüedad (No Liquidados)",
                nbins=20,
                color_discrete_sequence=['#f5576c']
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_hist_antiguedad")
        
        with col2:
            fig = px.box(
                df_no_liquidados,
                x='STATUS',
                y='EDAD_DIAS',
                title="Antigüedad por STATUS",
                color='STATUS'
            )
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="chart_box_antiguedad")


def seccion_alertas_avanzadas(df: pd.DataFrame):
    """Dashboard de alertas avanzadas"""
    
    st.subheader("🚨 Alertas Avanzadas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ordenes_3_intentos = df[df['NUM_INTENTOS'] >= 3]
        
        st.markdown("### ⚠️ ≥3 Intentos")
        st.metric("Cantidad", len(ordenes_3_intentos))
        
        if len(ordenes_3_intentos) > 0:
            with st.expander("Ver detalle"):
                st.dataframe(ordenes_3_intentos[['ORDEN', 'CLIENTE', 'STATUS', 'NUM_INTENTOS', 'EDAD_DIAS']], height=200)
                descargar_csv(ordenes_3_intentos, "ordenes_3_mas_intentos.csv", key="download_3_intentos")
    
    with col2:
        df_rechazos = df[df['STATUS'] == 'RECHAZADO']
        rechazos_cliente = df_rechazos.groupby('CLIENTE').size().reset_index(name='rechazos')
        rechazos_cliente = rechazos_cliente[rechazos_cliente['rechazos'] > 1]
        
        st.markdown("### ❌ Rechazos Recurrentes")
        st.metric("Clientes Afectados", len(rechazos_cliente))
        
        if len(rechazos_cliente) > 0:
            with st.expander("Ver detalle"):
                st.dataframe(rechazos_cliente, height=200)
                descargar_csv(rechazos_cliente, "clientes_rechazos_recurrentes.csv", key="download_rechazos_rec")
    
    with col3:
        ordenes_antiguas_gestion = df[
            (df['STATUS'] == 'EN GESTION') & 
            (df['EDAD_DIAS'] >= 5)
        ]
        
        st.markdown("### ⏰ Antiguas en Gestión")
        st.metric("≥5 días", len(ordenes_antiguas_gestion))
        
        if len(ordenes_antiguas_gestion) > 0:
            with st.expander("Ver detalle"):
                st.dataframe(ordenes_antiguas_gestion[['ORDEN', 'CLIENTE', 'EDAD_DIAS', 'STATUS','SUB STATUS']], height=200)
                descargar_csv(ordenes_antiguas_gestion, "ordenes_antiguas_gestion.csv", key="download_antiguas")
    
    st.markdown("---")
    
    # Segunda fila de alertas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ordenes_muy_antiguas = df[(~df['ES_LIQUIDADO']) & (df['EDAD_DIAS'] >= 7)]
        
        st.markdown("### 🕐 ≥7 días sin Liquidar")
        st.metric("Cantidad", len(ordenes_muy_antiguas))
        st.metric("Valor en Riesgo", f"Q{ordenes_muy_antiguas['VALOR_NUM'].sum():,.0f}")
        
        if len(ordenes_muy_antiguas) > 0:
            with st.expander("Ver detalle"):
                st.dataframe(ordenes_muy_antiguas[['ORDEN', 'CLIENTE', 'STATUS', 'EDAD_DIAS', 'VALOR_NUM']], height=200)
                descargar_csv(ordenes_muy_antiguas, "ordenes_muy_antiguas.csv", key="download_muy_antiguas")
    
    with col2:
        df_ilocalizables = df[df['STATUS'] == 'ILOCALIZABLE']
        
        st.markdown("### 📍 Ilocalizables")
        st.metric("Cantidad", len(df_ilocalizables))
        st.metric("Valor", f"Q{df_ilocalizables['VALOR_NUM'].sum():,.0f}")
        
        if len(df_ilocalizables) > 0:
            with st.expander("Ver detalle"):
                st.dataframe(df_ilocalizables[['ORDEN', 'CLIENTE', 'SUB STATUS', 'EDAD_DIAS', 'VALOR_NUM']], height=200)
                descargar_csv(df_ilocalizables, "ordenes_ilocalizables.csv", key="download_ilocalizables")
    
    with col3:
        df_fuera_cob = df[df['STATUS'] == 'FUERA DE COBERTURA']
        
        st.markdown("### 🚫 Fuera de Cobertura")
        st.metric("Cantidad", len(df_fuera_cob))
        st.metric("Valor", f"Q{df_fuera_cob['VALOR_NUM'].sum():,.0f}")
        
        if len(df_fuera_cob) > 0:
            with st.expander("Ver detalle"):
                st.dataframe(df_fuera_cob[['ORDEN', 'CLIENTE', 'GEO_KEY', 'VALOR_NUM']], height=200)
                descargar_csv(df_fuera_cob, "ordenes_fuera_cobertura.csv", key="download_fuera_cob")
    
    st.markdown("---")
    
    # Resumen de alertas
    st.subheader("📋 Resumen de Alertas Críticas")
    
    alertas = []
    
    # Alertas de efectividad por zona
    efectividad_zona = df.groupby('GEO_KEY').agg(
        total=('ORDEN', 'count'),
        liquidados=('ES_LIQUIDADO', 'sum')
    ).reset_index()
    efectividad_zona['efectividad'] = (efectividad_zona['liquidados'] / efectividad_zona['total']) * 100
    
    zonas_criticas = efectividad_zona[(efectividad_zona['efectividad'] < 50) & (efectividad_zona['total'] >= 10)]
    for _, row in zonas_criticas.iterrows():
        alertas.append({
            'Tipo': '🔴 Efectividad Crítica',
            'Detalle': f"Zona: {row['GEO_KEY']}",
            'Valor': f"{row['efectividad']:.1f}% ({int(row['total'])} órdenes)",
            'Severidad': 'Alta'
        })
    
    # Alerta de valor pendiente
    kpis = obtener_kpis_generales(df)
    if kpis['valor_pendiente'] > 50000:
        alertas.append({
            'Tipo': '💰 Valor Pendiente Muy Alto',
            'Detalle': 'Total general',
            'Valor': f"Q{kpis['valor_pendiente']:,.0f}",
            'Severidad': 'Alta'
        })
    
    # Alerta de entregados sin liquidar
    if kpis['valor_entregado_sin_liquidar'] > 10000:
        alertas.append({
            'Tipo': '📦 Valor Entregado sin Liquidar',
            'Detalle': f"{kpis['total_entregadas_sin_liquidar']} órdenes",
            'Valor': f"Q{kpis['valor_entregado_sin_liquidar']:,.0f}",
            'Severidad': 'Media'
        })
    
    # Alerta de órdenes muy antiguas
    if len(ordenes_muy_antiguas) > 0:
        alertas.append({
            'Tipo': '🕐 Órdenes Muy Antiguas',
            'Detalle': '≥7 días sin liquidar',
            'Valor': f"{len(ordenes_muy_antiguas)} órdenes",
            'Severidad': 'Alta'
        })
    
    if alertas:
        df_alertas = pd.DataFrame(alertas)
        st.dataframe(df_alertas, use_container_width=True)
        descargar_csv(df_alertas, "resumen_alertas.csv", key="download_resumen_alertas")
    else:
        st.success("✅ No hay alertas críticas activas.")


def seccion_filtros_personalizados(df: pd.DataFrame):
    """Dashboard personalizado con filtros"""
    
    st.subheader("🔍 Dashboard Personalizado")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        clientes = sorted(df['CLIENTE'].dropna().unique().tolist())
        cliente_sel = st.multiselect(
            "Cliente:",
            options=clientes,
            default=clientes,
            key="filtro_cliente_pers"
        )
    
    with col2:
        asesores = ['Todos'] + sorted(df['ASESOR'].astype(str).unique().tolist())
        asesor_sel = st.selectbox("Asesor:", asesores, key="filtro_asesor_pers")
    
    with col3:
        status_list = sorted(df['STATUS'].dropna().unique().tolist())
        status_sel = st.multiselect(
            "Status:",
            options=status_list,
            default=status_list,
            key="filtro_status_pers"
        )

    with col4:
        deptos = sorted(df['DEPARTAMENTO'].dropna().unique().tolist())
        depto_sel = st.multiselect(
            "Departamento:",
            options=deptos,
            default=deptos,
            key="filtro_depto_pers"
        )

    # Aplicar filtros
    df_filtrado = df.copy()
    
    if cliente_sel:
        df_filtrado = df_filtrado[df_filtrado['CLIENTE'].isin(cliente_sel)]
    
    if asesor_sel != 'Todos':
        df_filtrado = df_filtrado[df_filtrado['ASESOR'].astype(str) == asesor_sel]
    
    if status_sel:
        df_filtrado = df_filtrado[df_filtrado['STATUS'].isin(status_sel)]

    if depto_sel:
        df_filtrado = df_filtrado[df_filtrado['DEPARTAMENTO'].isin(depto_sel)]

    if len(df_filtrado) == 0:
        st.warning("No hay datos con los filtros seleccionados.")
        return
    
    # KPIs filtrados
    kpis = obtener_kpis_generales(df_filtrado)
    
    st.write(f"Mostrando **{len(df_filtrado):,}** órdenes de **{len(df):,}** totales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Órdenes", f"{kpis['total_ordenes']:,}")
    
    with col2:
        st.metric("Liquidadas", f"{kpis['total_liquidadas']:,}")
    
    with col3:
        st.metric("Efectividad", f"{kpis['efectividad']:.1f}%")
    
    with col4:
        st.metric("Valor", f"Q{kpis['valor_total']:,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = crear_grafico_status(df_filtrado)
        st.plotly_chart(fig, use_container_width=True, key="chart_status_pers")
    
    with col2:
        fig = crear_grafico_sub_status(df_filtrado)
        st.plotly_chart(fig, use_container_width=True, key="chart_substatus_pers")
    
    # Tabla de datos filtrados
    st.subheader("📋 Datos Filtrados")
    st.dataframe(
        df_filtrado[['ORDEN', 'CLIENTE', 'ASESOR', 'STATUS', 'SUB STATUS', 'EDAD_DIAS', 'VALOR_NUM', 'DEPARTAMENTO', 'GEO_KEY']],
        use_container_width=True,
        height=400
    )
    descargar_csv(df_filtrado, "datos_filtrados.csv", key="download_datos_filtrados")


# ============================================================================
# APLICACIÓN PRINCIPAL
# ============================================================================

def main():
    st.title("Sistema de Control Logístico y Reportería")
    st.markdown("*Dashboard de análisis operativo y estratégico*")
    
    # Inicializar estado
    if 'datos_cargados' not in st.session_state:
        st.session_state.datos_cargados = False
        st.session_state.df = None
        st.session_state.errores = []
        st.session_state.archivos_procesados = 0
    
    # Sidebar
    with st.sidebar:
        st.header("Carga de Archivos")
        
        archivos = st.file_uploader(
            "Subir archivos CSV",
            type=['csv'],
            accept_multiple_files=True,
            help="Puedes subir múltiples archivos CSV que serán consolidados"
        )
        
        if archivos:
            st.info(f"📎 {len(archivos)} archivo(s) seleccionado(s)")
        
        procesar = st.button("Procesar Archivos", type="primary", disabled=not archivos)
        
        if procesar and archivos:
            errores_estructura = []
            errores_contenido = []
            dfs_validos = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, archivo in enumerate(archivos):
                status_text.text(f"Procesando: {archivo.name}")
                progress_bar.progress((i + 1) / len(archivos))
                
                try:
                    df_temp = pd.read_csv(archivo, encoding='utf-8-sig')
                    df_temp = normalizar_columnas(df_temp)
                    
                    es_valido, mensaje = validar_estructura_csv(df_temp, archivo.name)
                    
                    if not es_valido:
                        errores_estructura.append(mensaje)
                        continue
                    
                    for idx, row in df_temp.iterrows():
                        errores_fila = validar_contenido_fila(row, idx)
                        for error in errores_fila:
                            error['archivo'] = archivo.name
                        errores_contenido.extend(errores_fila)
                    
                    dfs_validos.append(df_temp)
                    
                except Exception as e:
                    errores_estructura.append(f"❌ **{archivo.name}**: Error al leer el archivo - {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            if errores_estructura:
                st.session_state.datos_cargados = False
                st.session_state.errores = errores_estructura
                st.session_state.tipo_error = 'estructura'
            elif errores_contenido:
                st.session_state.datos_cargados = False
                st.session_state.errores = errores_contenido
                st.session_state.tipo_error = 'contenido'
            elif dfs_validos:
                df_consolidado = pd.concat(dfs_validos, ignore_index=True)
                df_consolidado = procesar_dataframe(df_consolidado)
                
                st.session_state.df = df_consolidado
                st.session_state.datos_cargados = True
                st.session_state.errores = []
                st.session_state.archivos_procesados = len(dfs_validos)
                
                st.success(f" {len(dfs_validos)} archivo(s) procesado(s) correctamente")
                st.info(f" Total de registros: {len(df_consolidado):,}")
            else:
                st.error("No se pudieron procesar los archivos.")
        
        # Filtros globales
        if st.session_state.datos_cargados:
            st.markdown("---")
            st.header("🔧 Filtros Globales")
            
            df = st.session_state.df
            
            if df['FECHA_DT'].notna().any():
                fecha_min = df['FECHA_DT'].min().date()
                fecha_max = df['FECHA_DT'].max().date()
                
                rango_fechas = st.date_input(
                    "Rango de fechas:",
                    value=(fecha_min, fecha_max),
                    min_value=fecha_min,
                    max_value=fecha_max,
                    key="filtro_fecha_global"
                )
                
                if len(rango_fechas) == 2:
                    st.session_state.filtro_fecha = rango_fechas

        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <p style="color: #6b6b6b; font-size: 0.85rem; margin-top: 0.5rem;">
                Powered by Steff 
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Contenido principal
    if st.session_state.errores:
        if st.session_state.get('tipo_error') == 'estructura':
            st.error("### ❌ Errores Estructurales Detectados")
            st.markdown("Los siguientes archivos tienen problemas de estructura y no pueden ser procesados:")
            for error in st.session_state.errores:
                st.markdown(error)
            st.warning("⚠️ Corrija los archivos y vuelva a intentar.")
        else:
            seccion_errores(st.session_state.errores)
    
    elif st.session_state.datos_cargados:
        df = st.session_state.df.copy()
        
        # Aplicar filtro de fecha
        if hasattr(st.session_state, 'filtro_fecha') and len(st.session_state.filtro_fecha) == 2:
            fecha_inicio, fecha_fin = st.session_state.filtro_fecha
            df = df[
                (df['FECHA_DT'].dt.date >= fecha_inicio) & 
                (df['FECHA_DT'].dt.date <= fecha_fin)
            ]
        
        if len(df) == 0:
            st.warning("No hay datos en el rango de fechas seleccionado.")
            return
        
        # Tabs principales
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard Principal",
            "📈 Reportes",
            "🚨 Alertas Avanzadas",
            "🔍 Personalizado"
        ])
        
        with tab1:
            seccion_dashboard_principal(df)
        
        with tab2:
            seccion_reportes(df)
        
        with tab3:
            seccion_alertas_avanzadas(df)
        
        with tab4:
            seccion_filtros_personalizados(df)
    
    else:
        st.markdown("""
        ### Bienvenido al Sistema de Control Logístico
        
        Este sistema te permite:
        
        - **Monitorear KPIs** de efectividad y rendimiento
        - **Detectar alertas** operativas tempranas
        - **Generar reportes** por cliente, asesor, zona y producto
        - **Analizar valor económico** de las operaciones
        - **Visualizar tendencias** temporales
        
        #### Para comenzar:
        1. Sube uno o más archivos CSV en el panel lateral
        2. Haz clic en "Procesar Archivos"
        3. Explora los diferentes dashboards y reportes
        
        #### Columnas requeridas en el CSV:
        """)
        
        cols = st.columns(3)
        for i, col in enumerate(COLUMNAS_OBLIGATORIAS):
            cols[i % 3].markdown(f"- `{col}`")


if __name__ == "__main__":
    main()