# Sistema de Control Logístico y Reportería

Aplicación web desarrollada en **Streamlit** para reportería logística y control operativo, basada en archivos CSV.

## Características

### Dashboard Principal
- **KPIs en tiempo real**: Total de órdenes, entregas, efectividad y valor económico
- **Alertas operativas**: 
  - Órdenes no entregadas ≥2 días en Región Metropolitana Guatemala
  - Órdenes no entregadas ≥3 días a nivel general
- **Visualizaciones interactivas**: Distribución por STATUS, tendencias temporales

### Reportes Incluidos
1. **Por Cliente/Asesor**: Efectividad, órdenes y rendimiento
2. **Geográfico**: Análisis por zona (geo_key), mapa de calor
3. **Productos**: Análisis desde campo REFERENCIA, top retornos
4. **Intentos y Fallos**: Promedio de intentos, razones de fallo
5. **Valor Económico**: Suma por STATUS, alertas de valor pendiente
6. **Tendencias**: Análisis temporal, antigüedad de pedidos

### Alertas Avanzadas
- Órdenes con ≥3 intentos de entrega
- Clientes con rechazos recurrentes
- Órdenes antiguas en estado EN GESTION

## Requisitos del CSV

El archivo CSV debe contener **exactamente** las siguientes columnas:

| Columna | Descripción |
|---------|-------------|
| CLIENTE | Nombre del cliente |
| ASESOR | Identificador del asesor |
| GUIA | Número de guía |
| FECHA | Fecha de la orden (DD/MM/YY o DD/MM/YYYY) |
| REMITENTE | Nombre del remitente |
| DESTINATARIO | Nombre del destinatario |
| DIRECCION | Dirección completa (separada por comas) |
| TELEFONO | Número de teléfono |
| COD | Código |
| VALOR | Valor monetario de la orden |
| ORDEN | Número de orden (obligatorio) |
| REFERENCIA | Referencia del producto |
| FECHA DEPOSITO | Fecha de depósito |
| STATUS | Estado actual (ver catálogo) |
| CONTROL INTERNO | Control interno |
| SUB STATUS | Sub-estado (ver catálogo) |
| INTENTOS DE ENTREGA | Número de intentos |
| REPROGRAMADO | Información de reprogramación |

### Catálogo de STATUS válidos
- ENTREGADO LIQUIDADO
- EN RUTA
- EN GESTION
- REPROGRAMADO
- ILOCALIZABLE
- RECHAZADO
- RECLAMO
- FUERA DE COBERTURA
- EN RUTA PARA DEVOLUCION
- RETORNADO A WEBCORP

### Catálogo de SUB STATUS válidos
- PUNTO DE ENCUENTRO
- ENTREGADO
- CONFIRMADO POR CLIENTE
- EN GESTION
- CONFIRMADO NUEVA FECHA
- ALMACENADO
- REPROGRAMADO CC
- DIRECCIÓN Y TELEFONO ERRONEO
- FUERA DE COBERTURA
- NO TIENE DINERO
- DUPLICADO
- NO HIZO PEDIDO
- PRECIO INCORRECTO
- CAMBIO DE DIRECCIÓN
- FUERA DE TIEMPO
- NUMERO INCORRECTO
- ERROR EN PRODUCTO
- RECHAZADO CC
- TIEMPO DE ESPERA
- COMPRA OTRO PRODUCTO
- NADIE EN CASA
- ESPERA DE PAGO CARGO
- AGENCIA FUERA DE COBERTURA
- AGENCIA A PETICION DEL CLIENTE
- DIRECTO A AGENCIA
- CUMPLIO INTENTOS DE ENTREGA
- RETORNO A SOLICITUD DE CC
- RETORNADO A WEBCORP

## Instalación

### Requisitos previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar o descargar los archivos**

2. **Crear entorno virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Linux/Mac
venv\Scripts\activate     # En Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

5. **Abrir en el navegador**
La aplicación se abrirá automáticamente en `http://localhost:8501`

## Uso

1. **Cargar archivos**: Usa el panel lateral para subir uno o más archivos CSV
2. **Procesar**: Haz clic en "Procesar Archivos"
3. **Validación**: 
   - Si hay errores estructurales, se mostrarán inmediatamente
   - Si hay errores de contenido, se mostrará un detalle descargable
4. **Explorar**: Navega por los diferentes tabs del dashboard
5. **Filtrar**: Usa los filtros globales en el panel lateral
6. **Descargar**: Todos los reportes tienen botón de descarga CSV

## 🔍 Segmentación Geográfica

Las direcciones se procesan automáticamente para extraer el **geo_key**:
- Se toma el **3er componente** del split por comas
- Ejemplo: `Region Metropolitana,Guatemala,Villa Nueva,...` → geo_key = `Villa Nueva`
- Si hay menos de 3 componentes → geo_key = `DESCONOCIDO`
- 
## Métricas Clave

### Efectividad
```
Efectividad (%) = (Órdenes ENTREGADO LIQUIDADO / Total órdenes) × 100
```

### Antigüedad
```
Edad (días) = Fecha actual - FECHA de la orden
```

### Reglas visuales
- Efectividad < 65% → Rojo (requiere atención)
- Efectividad ≥ 65% → Verde (objetivo cumplido)

## Estructura de archivos

```
logistics_app/
├── app.py              # Aplicación principal
├── requirements.txt    # Dependencias
└── README.md          # Este archivo
```

## Tecnologías utilizadas

- **Streamlit**: Framework de aplicaciones web
- **Pandas**: Manipulación de datos
- **Plotly**: Visualizaciones interactivas
- **NumPy**: Cálculos numéricos

## Notas importantes

- Los archivos CSV deben usar codificación UTF-8
- Los campos vacíos en SUB STATUS son permitidos
- Las fechas soportan formatos: DD/MM/YY, DD/MM/YYYY, YYYY-MM-DD
- El valor #N/A en FECHA DEPOSITO se considera como vacío válido

---

Desarrollado para análisis logístico y control operativo 📦
