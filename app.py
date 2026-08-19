import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import warnings
import requests
import io
import re
from itertools import combinations, product
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="📊 Diseños Experimentales y ANOVA",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Calculadora de Diseños Experimentales y ANOVA")
st.markdown("---")

# ============================================================
# INICIALIZAR SESSION STATE
# ============================================================

if 'datos_cargados' not in st.session_state:
    st.session_state.datos_cargados = {}
if 'resultados_anova' not in st.session_state:
    st.session_state.resultados_anova = {}
if 'diseno_actual' not in st.session_state:
    st.session_state.diseno_actual = None
if 'factores_definidos' not in st.session_state:
    st.session_state.factores_definidos = {}
if 'respuestas_definidas' not in st.session_state:
    st.session_state.respuestas_definidas = {}

# ============================================================
# FUNCIONES PARA PRUEBAS MÚLTIPLES
# ============================================================

def obtener_mse_anova(anova_table):
    """
    Obtiene el MSE (Mean Square Error) del DataFrame de ANOVA
    """
    try:
        if 'mean_sq' in anova_table.columns:
            return anova_table['mean_sq'].iloc[-1], anova_table['df'].iloc[-1]
        if 'MSE' in anova_table.columns:
            return anova_table['MSE'].iloc[-1], anova_table['df'].iloc[-1]
        if 'MS' in anova_table.columns:
            return anova_table['MS'].iloc[-1], anova_table['df'].iloc[-1]
        if 'Mean Sq' in anova_table.columns:
            return anova_table['Mean Sq'].iloc[-1], anova_table['df'].iloc[-1]
        
        for col in anova_table.columns:
            col_lower = col.lower()
            if 'mean' in col_lower or 'square' in col_lower or 'ms' in col_lower:
                if col != 'df' and col != 'F' and col != 'PR(>F)':
                    try:
                        return anova_table[col].iloc[-1], anova_table['df'].iloc[-1]
                    except:
                        continue
        
        for col in anova_table.columns:
            if col not in ['df', 'sum_sq', 'F', 'PR(>F)']:
                try:
                    if pd.api.types.is_numeric_dtype(anova_table[col]):
                        return anova_table[col].iloc[-1], anova_table['df'].iloc[-1]
                except:
                    continue
        
        return None, None
    except Exception as e:
        return None, None

def realizar_tukey(df, factor, respuesta, alpha=0.05):
    """Realiza prueba de Tukey HSD"""
    try:
        df[factor] = df[factor].astype('category')
        tukey = pairwise_tukeyhsd(df[respuesta], df[factor], alpha=alpha)
        return tukey
    except Exception as e:
        return None

def interpretar_tukey(tukey):
    """Interpreta los resultados de Tukey HSD"""
    interpretacion = []

    if tukey is None:
        return ["❌ No se pudo realizar la prueba de Tukey"]

    try:
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

        significativas = tukey_df[tukey_df['reject'] == True]

        if len(significativas) == 0:
            interpretacion.append("📊 **Tukey HSD:** No hay diferencias significativas entre ningún par de tratamientos")
            return interpretacion

        interpretacion.append(f"📊 **Tukey HSD:** Se encontraron {len(significativas)} diferencias significativas")

        interpretacion.append("\n**Diferencias significativas encontradas:**")
        for _, row in significativas.iterrows():
            interpretacion.append(f"  • {row['group1']} vs {row['group2']}: Diferencia = {row['meandiff']:.4f} (p = {row['p-adj']:.6f})")
    except Exception as e:
        interpretacion.append(f"❌ Error al interpretar Tukey: {str(e)}")

    return interpretacion

# ============================================================
# FUNCIONES PARA GRÁFICOS DE RESIDUOS - CORREGIDAS
# ============================================================

def graficos_residuos_universal(modelo, df, factor=None, respuesta=None, titulo="Análisis de Residuos"):
    """
    Genera gráficos de residuos para cualquier modelo ANOVA
    """
    if modelo is None:
        st.warning("⚠️ No hay modelo para generar gráficos de residuos")
        return None
    
    try:
        # Obtener residuos y valores predichos
        residuos = modelo.resid
        predichos = modelo.fittedvalues
        run_order = np.arange(1, len(residuos) + 1)
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Residuos vs Factor Level",
                "Residuos vs Predichos",
                "Residuos vs Run Order",
                "Residual Probability Plot"
            ]
        )
        
        # 1. Residuos vs Factor Level
        if factor is not None and factor in df.columns:
            try:
                factor_values = df[factor].values
            except:
                factor_values = run_order
        else:
            factor_values = run_order
        
        fig.add_trace(
            go.Scatter(
                x=factor_values,
                y=residuos,
                mode='markers',
                marker=dict(color='#3498db', size=8),
                name='Residuos vs Factor'
            ),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        fig.update_xaxes(title_text="Factor Level" if factor is not None else "Index", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        
        # 2. Residuos vs Predichos
        fig.add_trace(
            go.Scatter(
                x=predichos,
                y=residuos,
                mode='markers',
                marker=dict(color='#2ecc71', size=8),
                name='Residuos vs Predichos'
            ),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        
        # 3. Residuos vs Run Order
        fig.add_trace(
            go.Scatter(
                x=run_order,
                y=residuos,
                mode='lines+markers',
                marker=dict(color='#e74c3c', size=8),
                line=dict(color='#e74c3c', width=1),
                name='Residuos vs Run Order'
            ),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        fig.update_xaxes(title_text="Run Order", row=2, col=1)
        fig.update_yaxes(title_text="Residuals", row=2, col=1)
        
        # 4. Gráfico de Probabilidad Normal
        from scipy.stats import probplot
        try:
            (osm, osr), (slope, intercept, r) = probplot(residuos, dist="norm", fit=True)
            
            fig.add_trace(
                go.Scatter(
                    x=osm,
                    y=osr,
                    mode='markers',
                    marker=dict(color='#9b59b6', size=8),
                    name='Residuals'
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=[min(osm), max(osm)],
                    y=[min(osm)*slope + intercept, max(osm)*slope + intercept],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Expected Normal'
                ),
                row=2, col=2
            )
        except:
            fig.add_annotation(
                text="No se pudo generar el gráfico de probabilidad",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
        fig.update_yaxes(title_text="Ordered Residuals", row=2, col=2)
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text=titulo
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Error al generar gráficos de residuos: {str(e)}")
        return None

def mostrar_graficos_residuos_directo(modelo, df, factor=None, respuesta=None, key_suffix="", titulo="Gráficos de Residuos"):
    """
    Muestra los gráficos de residuos directamente sin botón
    """
    
    if modelo is None:
        st.warning("⚠️ No hay modelo para generar gráficos de residuos")
        return
    
    st.subheader(f"📈 {titulo}")
    
    with st.spinner("Generando gráficos de residuos..."):
        fig_residuos = graficos_residuos_universal(modelo, df, factor, respuesta, f"Análisis de Residuos - {titulo}")
        if fig_residuos:
            st.plotly_chart(fig_residuos, use_container_width=True)
            
            # Pruebas de normalidad
            st.subheader("📊 Pruebas de Normalidad de Residuos")
            from scipy.stats import shapiro, anderson
            
            residuos = modelo.resid
            col1, col2 = st.columns(2)
            with col1:
                try:
                    shapiro_stat, shapiro_p = shapiro(residuos)
                    st.metric("Shapiro-Wilk", f"p = {shapiro_p:.6f}")
                    if shapiro_p > 0.05:
                        st.success("✅ Los residuos siguen distribución normal (p > 0.05)")
                    else:
                        st.warning("⚠️ Los residuos NO siguen distribución normal (p ≤ 0.05)")
                except:
                    st.warning("⚠️ No se pudo realizar la prueba de Shapiro-Wilk")
            
            with col2:
                try:
                    anderson_result = anderson(residuos)
                    st.metric("Anderson-Darling", f"Stat = {anderson_result.statistic:.4f}")
                    if anderson_result.statistic < anderson_result.critical_values[2]:
                        st.success("✅ Los residuos siguen distribución normal (al 5%)")
                    else:
                        st.warning("⚠️ Los residuos NO siguen distribución normal (al 5%)")
                except:
                    st.warning("⚠️ No se pudo realizar la prueba de Anderson-Darling")

# ============================================================
# FUNCIONES PARA DISEÑO INTERACTIVO
# ============================================================

def crear_diseno_completo(factores_dict, respuestas_dict, n_replicas=1, randomize=True):
    try:
        niveles_factores = [factores_dict[f]['niveles'] for f in factores_dict]
        nombres_factores = list(factores_dict.keys())
        
        combinaciones = list(product(*niveles_factores))
        
        df = pd.DataFrame(combinaciones, columns=nombres_factores)
        
        if n_replicas > 1:
            df = pd.concat([df] * n_replicas, ignore_index=True)
        
        if randomize:
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        for resp_name, resp_data in respuestas_dict.items():
            if 'valores' in resp_data and resp_data['valores'] is not None and len(resp_data['valores']) > 0:
                valores = resp_data['valores']
                if len(valores) >= len(df):
                    df[resp_name] = valores[:len(df)]
                else:
                    df[resp_name] = np.tile(valores, int(np.ceil(len(df)/len(valores))))[:len(df)]
            else:
                df[resp_name] = np.random.normal(10, 2, len(df))
        
        return df
    except Exception as e:
        st.error(f"❌ Error al crear diseño: {str(e)}")
        return None

def mostrar_definicion_factores():
    st.subheader("🔧 Definición de Factores")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        n_factores = st.number_input(
            "Número de factores:",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="n_factores_def"
        )
    
    with col2:
        st.caption("")
        st.caption("")
        if st.button("🔄 Actualizar Factores", key="actualizar_factores"):
            st.rerun()
    
    st.markdown("---")
    
    factores_def = {}
    for i in range(n_factores):
        st.subheader(f"📌 Factor {chr(65+i)}")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            nombre = st.text_input(
                f"Nombre del Factor {chr(65+i)}:",
                value=f"Factor_{chr(65+i)}",
                key=f"nombre_factor_{i}"
            )
        
        with col2:
            n_niveles = st.number_input(
                f"Niveles:",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                key=f"niveles_factor_{i}"
            )
        
        with col3:
            tipo = st.selectbox(
                "Tipo:",
                ["Cualitativo", "Cuantitativo"],
                key=f"tipo_factor_{i}"
            )
        
        st.caption(f"**Niveles para {nombre}:**")
        niveles = []
        cols = st.columns(min(n_niveles, 5))
        for j in range(n_niveles):
            with cols[j % len(cols)]:
                nivel = st.text_input(
                    f"Nivel {j+1}:",
                    value=str(j+1),
                    key=f"nivel_{i}_{j}"
                )
                niveles.append(nivel)
        
        factores_def[nombre] = {
            'niveles': niveles,
            'tipo': tipo,
            'n_niveles': n_niveles
        }
        
        st.markdown("---")
    
    return factores_def

def mostrar_definicion_respuestas():
    st.subheader("📊 Definición de Respuestas")
    
    n_respuestas = st.number_input(
        "Número de respuestas:",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        key="n_respuestas_def"
    )
    
    st.markdown("---")
    
    respuestas_def = {}
    for i in range(n_respuestas):
        st.subheader(f"📈 Respuesta {i+1}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            nombre = st.text_input(
                f"Nombre de la Respuesta {i+1}:",
                value=f"Respuesta_{i+1}",
                key=f"nombre_respuesta_{i}"
            )
        
        with col2:
            n_valores = st.number_input(
                f"Valores predefinidos:",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
                key=f"n_valores_respuesta_{i}"
            )
        
        if n_valores > 0:
            st.caption(f"**Valores para {nombre}:**")
            valores = []
            cols = st.columns(min(n_valores, 5))
            for j in range(n_valores):
                with cols[j % len(cols)]:
                    valor = st.text_input(
                        f"Valor {j+1}:",
                        value=str(np.random.uniform(5, 15)),
                        key=f"valor_respuesta_{i}_{j}"
                    )
                    try:
                        valores.append(float(valor))
                    except:
                        valores.append(float(valor) if valor.replace('.', '').isdigit() else 0)
            respuestas_def[nombre] = {'valores': valores}
        else:
            respuestas_def[nombre] = {'valores': None}
        
        st.markdown("---")
    
    return respuestas_def

def mostrar_configuracion_diseno():
    st.subheader("⚙️ Configuración del Diseño")
    
    col1, col2 = st.columns(2)
    
    with col1:
        replicas = st.number_input(
            "Número de réplicas:",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            key="replicas_diseno"
        )
    
    with col2:
        randomizar = st.checkbox(
            "Randomizar diseño",
            value=True,
            key="randomizar_diseno"
        )
    
    return replicas, randomizar

def mostrar_resumen_diseno(df, factores_def, respuestas_def):
    st.subheader("📋 Resumen del Diseño")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de corridas", len(df))
    with col2:
        st.metric("Factores", len(factores_def))
    with col3:
        st.metric("Respuestas", len(respuestas_def))
    
    st.write("**Estructura del diseño:**")
    info_df = pd.DataFrame({
        'Tipo': ['Factor' if f in factores_def else 'Respuesta' for f in df.columns],
        'Niveles/Valores': [len(factores_def[f]['niveles']) if f in factores_def else 'Continuo' for f in df.columns]
    }, index=df.columns)
    st.dataframe(info_df)
    
    st.write("**Datos del diseño:**")
    st.dataframe(df.head(10))
    
    st.write("**Estadísticas descriptivas de las respuestas:**")
    respuestas = [col for col in df.columns if col in respuestas_def]
    if respuestas:
        st.dataframe(df[respuestas].describe())

# ============================================================
# FUNCIONES DE ANÁLISIS Y CONCLUSIONES
# ============================================================

def generar_conclusiones_anova(anova_table, modelo, df, factor, respuesta, alpha=0.05):
    conclusiones = []

    if 'PR(>F)' in anova_table.columns:
        p_valor = anova_table['PR(>F)'].iloc[0]
    else:
        p_valor = anova_table['F'].iloc[0]

    if p_valor < alpha:
        conclusiones.append(f"✅ **Conclusión Principal:** Existe diferencia significativa entre los tratamientos (p = {p_valor:.6f} < {alpha})")
    else:
        conclusiones.append(f"❌ **Conclusión Principal:** No existe diferencia significativa entre los tratamientos (p = {p_valor:.6f} ≥ {alpha})")

    medias = df.groupby(factor)[respuesta].agg(['mean', 'std', 'count']).reset_index()
    medias.columns = [factor, 'Media', 'Desv_Estandar', 'n']
    medias_ordenadas = medias.sort_values('Media', ascending=False)

    conclusiones.append("\n📊 **Estadísticas Descriptivas:**")
    for _, row in medias_ordenadas.iterrows():
        conclusiones.append(f"  • {row[factor]}: Media = {row['Media']:.4f} ± {row['Desv_Estandar']:.4f} (n={int(row['n'])})")

    mejor = medias_ordenadas.iloc[0]
    peor = medias_ordenadas.iloc[-1]

    conclusiones.append(f"\n🏆 **Mejor tratamiento:** {mejor[factor]} (Media = {mejor['Media']:.4f})")
    conclusiones.append(f"📉 **Peor tratamiento:** {peor[factor]} (Media = {peor['Media']:.4f})")

    diff = mejor['Media'] - peor['Media']
    conclusiones.append(f"📊 **Diferencia máxima:** {diff:.4f} unidades")

    media_total = df[respuesta].mean()
    desv_total = df[respuesta].std()
    cv = (desv_total / media_total) * 100
    conclusiones.append(f"📈 **Coeficiente de variación:** {cv:.2f}%")

    if cv < 10:
        conclusiones.append("  • Alta precisión experimental (CV < 10%)")
    elif cv < 20:
        conclusiones.append("  • Precisión experimental aceptable (CV < 20%)")
    else:
        conclusiones.append("  • Baja precisión experimental (CV ≥ 20%)")

    conclusiones.append(f"\n📐 **Calidad del modelo:**")
    conclusiones.append(f"  • R² = {modelo.rsquared:.6f}")
    conclusiones.append(f"  • R² ajustado = {modelo.rsquared_adj:.6f}")

    if modelo.rsquared > 0.8:
        conclusiones.append("  • Excelente ajuste del modelo")
    elif modelo.rsquared > 0.6:
        conclusiones.append("  • Buen ajuste del modelo")
    else:
        conclusiones.append("  • Ajuste moderado del modelo")

    return conclusiones, medias_ordenadas

def generar_conclusiones_factorial(anova_table, modelo, df, factores, respuesta, alpha=0.05):
    conclusiones = []

    if 'PR(>F)' in anova_table.columns:
        for i, row in anova_table.iterrows():
            factor = row.name
            p_valor = row['PR(>F)']

            if p_valor < alpha:
                conclusiones.append(f"✅ **{factor}:** Efecto significativo (p = {p_valor:.6f} < {alpha})")
            else:
                conclusiones.append(f"❌ **{factor}:** Efecto no significativo (p = {p_valor:.6f} ≥ {alpha})")

    conclusiones.append(f"\n📐 **Calidad del modelo:**")
    conclusiones.append(f"  • R² = {modelo.rsquared:.6f}")
    conclusiones.append(f"  • R² ajustado = {modelo.rsquared_adj:.6f}")

    return conclusiones

def analizar_anova(df, formula, titulo="ANOVA", key_suffix="", tipo="dca"):
    try:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')

        modelo = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(modelo, typ=2)

        st.success(f"✅ {titulo} calculado")
        st.dataframe(anova_table)

        st.write(f"**R² = {modelo.rsquared:.6f}**")
        st.write(f"**R² ajustado = {modelo.rsquared_adj:.6f}**")

        st.session_state.resultados_anova[key_suffix] = {
            'modelo': modelo,
            'anova': anova_table,
            'df': df,
            'tipo': tipo,
            'formula': formula
        }

        return modelo, anova_table
    except Exception as e:
        st.error(f"❌ Error en ANOVA: {str(e)}")
        st.error("🔍 Verifica que los datos tengan el formato correcto")
        return None, None

# ============================================================
# FUNCIONES PARA CARGA DE DATOS
# ============================================================

def detectar_separador(texto):
    lineas = [line.strip() for line in texto.split('\n') if line.strip()]
    if not lineas:
        return ','

    separadores = [',', ';', '\t', '|', ' ']
    conteos = {}

    for sep in separadores:
        conteos[sep] = sum(linea.count(sep) for linea in lineas)

    mejor_sep = max(conteos, key=conteos.get)
    return mejor_sep if conteos[mejor_sep] > 0 else ','

def limpiar_nombres_columnas(df):
    df.columns = [col.strip().replace(' ', '_').replace(';', '').replace(',', '') for col in df.columns]
    return df

def convertir_tipos(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            df[col] = df[col].astype('category')
    return df

def cargar_datos_desde_csv(archivo, separador=None):
    try:
        contenido = archivo.getvalue().decode('utf-8')

        if separador is None:
            separador = detectar_separador(contenido)

        try:
            df = pd.read_csv(io.StringIO(contenido), sep=separador)
        except:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(contenido), sep=sep)
                    if len(df.columns) > 1:
                        break
                except:
                    continue
            else:
                df = pd.read_csv(io.StringIO(contenido), sep=separador, engine='python')

        df = limpiar_nombres_columnas(df)
        df = convertir_tipos(df)

        return df
    except Exception as e:
        st.error(f"❌ Error al cargar CSV: {str(e)}")
        return None

def cargar_datos_desde_url(url, separador=None):
    try:
        try:
            response = requests.get(url, timeout=30)
        except:
            response = requests.get(url, timeout=30, verify=False)

        response.raise_for_status()

        if response.encoding is None:
            content = response.text
        else:
            content = response.content.decode(response.encoding, errors='ignore')

        if separador is None:
            separador = detectar_separador(content)

        try:
            df = pd.read_csv(io.StringIO(content), sep=separador)
        except:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep)
                    if len(df.columns) > 1:
                        break
                except:
                    continue
            else:
                df = pd.read_csv(io.StringIO(content), sep=separador, engine='python')

        df = limpiar_nombres_columnas(df)
        df = convertir_tipos(df)

        return df
    except Exception as e:
        st.error(f"❌ Error al cargar desde URL: {str(e)}")
        st.error("🔍 Verifica que la URL sea correcta y accesible")
        return None

def procesar_datos_texto(texto, n_columnas=None):
    try:
        separador = detectar_separador(texto)

        lineas = [line.strip() for line in texto.split('\n') if line.strip()]

        primera_linea = lineas[0]
        valores = primera_linea.split(separador)
        tiene_encabezado = False

        try:
            for v in valores:
                float(v.strip())
            tiene_encabezado = False
        except:
            tiene_encabezado = True

        datos = []
        columnas = None

        if tiene_encabezado:
            columnas = [v.strip() for v in valores]
            lineas = lineas[1:]

        for linea in lineas:
            valores = [v.strip() for v in linea.split(separador) if v.strip()]
            if n_columnas is None or len(valores) == n_columnas:
                try:
                    valores_num = [float(v) for v in valores]
                    datos.append(valores_num)
                except:
                    datos.append(valores)

        if not datos:
            return None

        df = pd.DataFrame(datos)

        if columnas and len(columnas) == len(df.columns):
            df.columns = columnas
        else:
            df.columns = [f'V{i+1}' for i in range(len(df.columns))]

        df = limpiar_nombres_columnas(df)
        df = convertir_tipos(df)

        return df
    except Exception as e:
        st.error(f"❌ Error al procesar texto: {str(e)}")
        return None

def obtener_datos_generales(key_suffix, n_columnas_esperadas=None):
    st.subheader("📥 Carga de Datos")

    tipo_carga = st.radio(
        "Selecciona el método de carga:",
        ["✏️ Ingresar manualmente", "📁 Subir archivo CSV", "🔗 Desde URL"],
        horizontal=True,
        key=f"tipo_carga_{key_suffix}"
    )

    df_resultado = None

    if tipo_carga == "✏️ Ingresar manualmente":
        if n_columnas_esperadas:
            st.caption(f"Formato: {', '.join([f'Col{i+1}' for i in range(n_columnas_esperadas)])}")
            ejemplo = ",".join(["valor" for _ in range(n_columnas_esperadas)])
            st.caption(f"Ejemplo: {ejemplo}")

        datos_texto = st.text_area(
            "Datos:",
            value="",
            height=200,
            key=f"text_area_{key_suffix}",
            placeholder="Ingresa tus datos aquí, uno por línea\nEjemplo: A,23\nA,25\nB,30"
        )

        if datos_texto and st.button("📊 Procesar datos", key=f"procesar_text_{key_suffix}"):
            df_resultado = procesar_datos_texto(datos_texto, n_columnas_esperadas)
            if df_resultado is not None:
                st.success(f"✅ Datos cargados correctamente ({len(df_resultado)} filas, {len(df_resultado.columns)} columnas)")
                st.dataframe(df_resultado.head())
                st.session_state.datos_cargados[key_suffix] = df_resultado
            else:
                st.error("❌ Error al procesar los datos")

    elif tipo_carga == "📁 Subir archivo CSV":
        archivo = st.file_uploader(
            "Selecciona un archivo CSV",
            type=['csv', 'txt'],
            key=f"csv_upload_{key_suffix}"
        )

        if archivo is not None:
            if st.button("📊 Cargar archivo", key=f"cargar_csv_{key_suffix}"):
                df_resultado = cargar_datos_desde_csv(archivo)
                if df_resultado is not None:
                    if len(df_resultado.columns) >= 1:
                        st.success(f"✅ Archivo cargado correctamente ({len(df_resultado)} filas, {len(df_resultado.columns)} columnas)")
                        st.dataframe(df_resultado.head())
                        st.session_state.datos_cargados[key_suffix] = df_resultado
                    else:
                        st.error("❌ El archivo no tiene columnas válidas")

    elif tipo_carga == "🔗 Desde URL":
        url = st.text_input(
            "URL del archivo CSV:",
            placeholder="https://raw.githubusercontent.com/usuario/repo/datos.csv",
            key=f"url_input_{key_suffix}"
        )

        if url:
            separador = st.selectbox(
                "Separador (opcional - auto-detecta si se deja vacío):",
                ['Auto-detectar', ',', ';', '\t', '|'],
                key=f"separador_{key_suffix}"
            )

            if st.button("📊 Cargar desde URL", key=f"cargar_url_{key_suffix}"):
                with st.spinner("Cargando datos desde URL..."):
                    sep = None if separador == 'Auto-detectar' else separador
                    df_resultado = cargar_datos_desde_url(url, sep)
                    if df_resultado is not None:
                        if len(df_resultado.columns) >= 1:
                            st.success(f"✅ Datos cargados desde URL ({len(df_resultado)} filas, {len(df_resultado.columns)} columnas)")
                            st.dataframe(df_resultado.head())
                            st.session_state.datos_cargados[key_suffix] = df_resultado
                        else:
                            st.error("❌ La URL no contiene datos válidos")

    if key_suffix in st.session_state.datos_cargados:
        df_guardado = st.session_state.datos_cargados[key_suffix]
        st.info(f"📊 Datos cargados previamente: {len(df_guardado)} filas, {len(df_guardado.columns)} columnas")
        if st.checkbox("Mostrar datos cargados", key=f"mostrar_datos_{key_suffix}"):
            st.dataframe(df_guardado.head())
        return df_guardado

    return df_resultado

def mostrar_informacion_datos(df, key_suffix):
    if df is not None and len(df.columns) > 0:
        st.subheader("📊 Información de los Datos")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Filas", len(df))
        with col_info2:
            st.metric("Columnas", len(df.columns))
        with col_info3:
            tipos = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    tipos.append('numérico')
                else:
                    tipos.append('categórico')
            st.metric("Tipos de datos", ", ".join(set(tipos)))

        st.caption("**Estructura de los datos:**")
        info_df = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': [str(df[col].dtype) for col in df.columns],
            'Valores únicos': [len(df[col].unique()) for col in df.columns]
        })
        st.dataframe(info_df)

# ============================================================
# FUNCIONES PARA GENERAR FÓRMULAS DE INTERACCIONES
# ============================================================

def generar_formula_interacciones(factores, nivel_interaccion):
    k = len(factores)

    if nivel_interaccion == "main":
        return 'Y ~ ' + ' + '.join(factores)

    elif nivel_interaccion == "2fi":
        formula = 'Y ~ ' + ' + '.join(factores)
        interacciones_2 = []
        for combo in combinations(range(k), 2):
            interacciones_2.append(':'.join([factores[i] for i in combo]))
        if interacciones_2:
            formula += ' + ' + ' + '.join(interacciones_2)
        return formula

    elif nivel_interaccion == "3fi":
        formula = 'Y ~ ' + ' + '.join(factores)
        interacciones_2 = []
        for combo in combinations(range(k), 2):
            interacciones_2.append(':'.join([factores[i] for i in combo]))
        if interacciones_2:
            formula += ' + ' + ' + '.join(interacciones_2)
        interacciones_3 = []
        for combo in combinations(range(k), 3):
            interacciones_3.append(':'.join([factores[i] for i in combo]))
        if interacciones_3:
            formula += ' + ' + ' + '.join(interacciones_3)
        return formula

    else:
        return 'Y ~ ' + ' * '.join(factores)

def obtener_info_interacciones(k, nivel_interaccion):
    if nivel_interaccion == "main":
        return f"Efectos principales ({k} términos)"
    elif nivel_interaccion == "2fi":
        n_interacciones = len(list(combinations(range(k), 2)))
        return f"Efectos principales ({k}) + Interacciones de 2 factores ({n_interacciones})"
    elif nivel_interaccion == "3fi":
        n_interacciones_2 = len(list(combinations(range(k), 2)))
        n_interacciones_3 = len(list(combinations(range(k), 3)))
        return f"Efectos principales ({k}) + 2F ({n_interacciones_2}) + 3F ({n_interacciones_3})"
    else:
        total = 2**k - 1
        return f"Modelo completo ({total} términos)"

# ============================================================
# FUNCIÓN PARA ANÁLISIS INTERACTIVO
# ============================================================

def analizar_diseno_interactivo(df, factores_def, respuestas_def):
    st.subheader("📊 Análisis del Diseño")
    
    respuestas = [col for col in df.columns if col in respuestas_def]
    if not respuestas:
        st.warning("No hay respuestas definidas para analizar")
        return
    
    respuesta_seleccionada = st.selectbox(
        "Selecciona la respuesta a analizar:",
        respuestas,
        key="respuesta_analisis_interactivo"
    )
    
    factores = [col for col in df.columns if col in factores_def]
    
    if len(factores) == 1:
        st.subheader("📊 ANOVA de un factor")
        factor = factores[0]
        df[factor] = df[factor].astype('category')
        
        formula = f'{respuesta_seleccionada} ~ C({factor})'
        
        try:
            modelo = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(modelo, typ=2)
            st.dataframe(anova_table)
            st.write(f"**R² = {modelo.rsquared:.6f}**")
            
            # Gráficos de residuos - DIRECTO
            mostrar_graficos_residuos_directo(modelo, df, factor, respuesta_seleccionada, "interactivo", "Gráficos de Residuos")
            
        except Exception as e:
            st.error(f"❌ Error en ANOVA: {str(e)}")
            return
        
    elif len(factores) == 2:
        st.subheader("📊 Diseño Factorial 2²")
        f1, f2 = factores[0], factores[1]
        df[f1] = df[f1].astype('category')
        df[f2] = df[f2].astype('category')
        
        formula = f'{respuesta_seleccionada} ~ {f1} * {f2}'
        
        try:
            modelo = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(modelo, typ=2)
            st.dataframe(anova_table)
            st.write(f"**R² = {modelo.rsquared:.6f}**")
            
            # Gráficos de residuos - DIRECTO
            mostrar_graficos_residuos_directo(modelo, df, f1, respuesta_seleccionada, "interactivo_22", "Gráficos de Residuos - Factorial 2²")
            
        except Exception as e:
            st.error(f"❌ Error en ANOVA: {str(e)}")
            return
        
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            try:
                fig = px.box(df, x=f1, y=respuesta_seleccionada, color=f2,
                            title=f"Boxplot por combinación")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Error al generar boxplot: {str(e)}")
        
        with col_graf2:
            try:
                medias = df.groupby([f1, f2])[respuesta_seleccionada].mean().reset_index()
                fig = px.line(medias, x=f1, y=respuesta_seleccionada, color=f2,
                             markers=True, title="Gráfico de Interacción")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Error al generar gráfico de interacción: {str(e)}")
            
    elif len(factores) >= 3:
        st.subheader("📊 ANOVA Multifactorial")
        
        nivel_interaccion = st.radio(
            "Selecciona el nivel de interacciones:",
            ["main", "2fi", "3fi", "all"],
            horizontal=True,
            key="interacciones_interactivo"
        )
        
        if nivel_interaccion == "main":
            formula = f'{respuesta_seleccionada} ~ ' + ' + '.join(factores)
        elif nivel_interaccion == "2fi":
            formula = f'{respuesta_seleccionada} ~ ' + ' + '.join(factores)
            for combo in combinations(factores, 2):
                formula += f' + {combo[0]}:{combo[1]}'
        elif nivel_interaccion == "3fi":
            formula = f'{respuesta_seleccionada} ~ ' + ' + '.join(factores)
            for combo in combinations(factores, 2):
                formula += f' + {combo[0]}:{combo[1]}'
            for combo in combinations(factores, 3):
                formula += f' + {combo[0]}:{combo[1]}:{combo[2]}'
        else:
            formula = f'{respuesta_seleccionada} ~ ' + ' * '.join(factores)
        
        for f in factores:
            df[f] = df[f].astype('category')
        
        try:
            modelo = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(modelo, typ=2)
            st.dataframe(anova_table)
            st.write(f"**R² = {modelo.rsquared:.6f}**")
            st.write(f"**R² ajustado = {modelo.rsquared_adj:.6f}**")
            
            # Gráficos de residuos - DIRECTO
            mostrar_graficos_residuos_directo(modelo, df, factores[0], respuesta_seleccionada, "interactivo_multi", "Gráficos de Residuos - Multifactorial")
            
        except Exception as e:
            st.error(f"❌ Error en ANOVA: {str(e)}")
            return
        
        st.subheader("📊 Visualización")
        n_filas = (len(factores) + 1) // 2
        fig = make_subplots(rows=n_filas, cols=2, subplot_titles=factores)
        for i, factor in enumerate(factores):
            row = i // 2 + 1
            col = i % 2 + 1
            medias = df.groupby(factor)[respuesta_seleccionada].mean().reset_index()
            fig.add_trace(go.Bar(x=medias[factor], y=medias[respuesta_seleccionada],
                                marker_color='#3498db'), row=row, col=col)
        fig.update_layout(height=min(400, 200 * n_filas), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# INTERFAZ PRINCIPAL - CREAR PESTAÑAS
# ============================================================

tab_interactivo, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "🎯 Diseño Interactivo",
    "📊 ANOVA Un Factor",
    "📊 Diseño en Bloques",
    "📐 Cuadrado Latino",
    "📐 Greco-Latino",
    "🎯 Factorial Dos Factores",
    "🎯 Factorial Tres Factores",
    "📊 Diseño 2^k",
    "📊 Fraccionado",
    "📈 MSR",
    "⚙️ Taguchi",
    "🎯 Optimización",
    "➕ Aditivos",
    "📦 Split-Plot"
])

# ============================================================
# PESTAÑA INTERACTIVA: DISEÑO EXPERIMENTAL
# ============================================================
with tab_interactivo:
    st.header("🎯 Diseño Experimental Interactivo")
    st.markdown("Define tus factores, niveles y respuestas para generar un diseño experimental personalizado.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📌 Paso 1: Definir Factores")
        factores_def = mostrar_definicion_factores()
        st.session_state.factores_definidos = factores_def
        
        st.subheader("📊 Paso 2: Definir Respuestas")
        respuestas_def = mostrar_definicion_respuestas()
        st.session_state.respuestas_definidas = respuestas_def
        
        st.subheader("⚙️ Paso 3: Configuración del Diseño")
        replicas, randomizar = mostrar_configuracion_diseno()
        
        if st.button("🚀 Generar Diseño Experimental", key="generar_diseno_interactivo"):
            if factores_def:
                df_diseno = crear_diseno_completo(
                    factores_def,
                    respuestas_def,
                    n_replicas=replicas,
                    randomize=randomizar
                )
                if df_diseno is not None:
                    st.session_state.diseno_actual = {
                        'df': df_diseno,
                        'factores': factores_def,
                        'respuestas': respuestas_def
                    }
                    st.success(f"✅ Diseño generado exitosamente! {len(df_diseno)} corridas")
                    st.balloons()
            else:
                st.error("❌ Define al menos un factor para generar el diseño")
    
    with col2:
        st.subheader("📋 Información del Diseño")
        if 'diseno_actual' in st.session_state and st.session_state.diseno_actual is not None:
            diseno = st.session_state.diseno_actual
            df = diseno['df']
            factores = diseno['factores']
            respuestas = diseno['respuestas']
            
            st.metric("Total de corridas", len(df))
            st.metric("Factores", len(factores))
            st.metric("Respuestas", len(respuestas))
            
            st.caption("**Vista previa de los datos:**")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🗑️ Limpiar diseño", key="limpiar_diseno"):
                st.session_state.diseno_actual = None
                st.rerun()
        else:
            st.info("Genera un diseño para comenzar el análisis")
    
    if 'diseno_actual' in st.session_state and st.session_state.diseno_actual is not None:
        st.markdown("---")
        diseno = st.session_state.diseno_actual
        df = diseno['df']
        factores_def = diseno['factores']
        respuestas_def = diseno['respuestas']
        
        mostrar_resumen_diseno(df, factores_def, respuestas_def)
        
        st.markdown("---")
        analizar_diseno_interactivo(df, factores_def, respuestas_def)
        
        st.markdown("---")
        st.subheader("💾 Descargar Diseño")
        
        try:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name="diseno_experimental.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"❌ Error al descargar CSV: {str(e)}")

# ============================================================
# PESTAÑA 1: ANOVA UN FACTOR
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Completamente Aleatorizado")
        df_temp = obtener_datos_generales("dca", 2)
        if df_temp is not None and len(df_temp.columns) >= 2:
            st.subheader("🔧 Configuración del ANOVA")
            col_factor = st.selectbox("Selecciona la columna del factor (Tratamiento):", df_temp.columns.tolist(), key="factor_dca")
            col_respuesta = st.selectbox("Selecciona la columna de respuesta (Valor):", [col for col in df_temp.columns if col != col_factor], key="respuesta_dca")
            mostrar_informacion_datos(df_temp, "dca")
            if st.button("🔍 Calcular ANOVA", key="dca_btn"):
                df = df_temp.copy()
                df.columns = ['Tratamiento' if col == col_factor else 'Valor' if col == col_respuesta else col for col in df.columns]
                df = df[['Tratamiento', 'Valor']]
                df['Tratamiento'] = df['Tratamiento'].astype('category')
                formula = 'Valor ~ C(Tratamiento)'
                modelo, anova = analizar_anova(df, formula, "ANOVA Un Factor", "dca", "dca")
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    
                    # Prueba de Tukey HSD
                    if len(df['Tratamiento'].unique()) > 2:
                        st.subheader("🔬 Prueba de Tukey HSD")
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None:
                            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                            st.dataframe(tukey_df)
                            for linea in interpretar_tukey(tukey):
                                st.write(linea)
                    
                    # Gráficos de residuos - DIRECTO
                    mostrar_graficos_residuos_directo(modelo, df, 'Tratamiento', 'Valor', "dca", "Gráficos de Residuos - DCA")
        elif df_temp is not None:
            st.warning("⚠️ Se necesitan al menos 2 columnas para el análisis")
    
    with col2:
        if 'dca' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['dca']
            df = resultados['df']
            modelo = resultados['modelo']
            anova = resultados['anova']
            st.subheader("📊 Visualización de Resultados")
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento')
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor', title="Medias por Tratamiento", color='Tratamiento', text_auto=True)
            fig2.update_layout(showlegend=False, height=300)
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
            st.subheader("📝 Conclusiones del Análisis")
            conclusiones, medias_ordenadas = generar_conclusiones_anova(anova, modelo, df, 'Tratamiento', 'Valor')
            for conclusion in conclusiones:
                if conclusion.startswith("✅") or conclusion.startswith("🏆") or conclusion.startswith("📈"):
                    st.success(conclusion)
                elif conclusion.startswith("❌"):
                    st.error(conclusion)
                else:
                    st.info(conclusion)
            if st.button("🗑️ Limpiar resultados", key="limpiar_dca"):
                del st.session_state.resultados_anova['dca']
                st.rerun()

# ============================================================
# PESTAÑA 2: DISEÑO EN BLOQUES
# ============================================================
with tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño en Bloques")
        df_temp = obtener_datos_generales("dbca", 3)
        if df_temp is not None and len(df_temp.columns) >= 3:
            st.subheader("🔧 Configuración del ANOVA")
            col_factor = st.selectbox("Selecciona la columna del factor (Tratamiento):", df_temp.columns.tolist(), key="factor_dbca")
            col_bloque = st.selectbox("Selecciona la columna del bloque:", [col for col in df_temp.columns if col != col_factor], key="bloque_dbca")
            col_respuesta = st.selectbox("Selecciona la columna de respuesta (Valor):", [col for col in df_temp.columns if col not in [col_factor, col_bloque]], key="respuesta_dbca")
            mostrar_informacion_datos(df_temp, "dbca")
            if st.button("🔍 Calcular", key="dbca_btn"):
                df = df_temp.copy()
                df.columns = ['Tratamiento' if col == col_factor else 'Bloque' if col == col_bloque else 'Valor' if col == col_respuesta else col for col in df.columns]
                df = df[['Tratamiento', 'Bloque', 'Valor']]
                df['Tratamiento'] = df['Tratamiento'].astype('category')
                df['Bloque'] = df['Bloque'].astype('category')
                formula = 'Valor ~ C(Tratamiento) + C(Bloque)'
                modelo, anova = analizar_anova(df, formula, "ANOVA en Bloques", "dbca", "dbca")
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    
                    if len(df['Tratamiento'].unique()) > 2:
                        st.subheader("🔬 Prueba de Tukey HSD para Tratamientos")
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None:
                            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                            st.dataframe(tukey_df)
                            for linea in interpretar_tukey(tukey):
                                st.write(linea)
                    
                    # Gráficos de residuos - DIRECTO
                    mostrar_graficos_residuos_directo(modelo, df, 'Tratamiento', 'Valor', "dbca", "Gráficos de Residuos - DBCA")
        elif df_temp is not None:
            st.warning("⚠️ Se necesitan al menos 3 columnas para el análisis")
    
    with col2:
        if 'dbca' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['dbca']
            df = resultados['df']
            modelo = resultados['modelo']
            anova = resultados['anova']
            st.subheader("📊 Visualización de Resultados")
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento', facet_col='Bloque')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor', title="Medias por Tratamiento", color='Tratamiento', text_auto=True)
            fig2.update_layout(showlegend=False, height=300)
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
            st.subheader("📝 Conclusiones del Análisis")
            conclusiones, medias_ordenadas = generar_conclusiones_anova(anova, modelo, df, 'Tratamiento', 'Valor')
            for conclusion in conclusiones:
                if conclusion.startswith("✅") or conclusion.startswith("🏆") or conclusion.startswith("📈"):
                    st.success(conclusion)
                elif conclusion.startswith("❌"):
                    st.error(conclusion)
                else:
                    st.info(conclusion)
            if st.button("🗑️ Limpiar resultados", key="limpiar_dbca"):
                del st.session_state.resultados_anova['dbca']
                st.rerun()

# ============================================================
# PESTAÑA 3: CUADRADO LATINO
# ============================================================
with tab3:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Cuadrado Latino")
        df_temp = obtener_datos_generales("cl", 4)
        if df_temp is not None and len(df_temp.columns) >= 4:
            st.subheader("🔧 Configuración del ANOVA")
            col_fila = st.selectbox("Columna de fila:", df_temp.columns.tolist(), key="fila_cl")
            cols_restantes = [c for c in df_temp.columns if c != col_fila]
            col_columna = st.selectbox("Columna de columna:", cols_restantes, key="columna_cl")
            cols_restantes = [c for c in cols_restantes if c != col_columna]
            col_tratamiento = st.selectbox("Columna de tratamiento:", cols_restantes, key="tratamiento_cl")
            cols_restantes = [c for c in cols_restantes if c != col_tratamiento]
            col_respuesta = st.selectbox("Columna de respuesta (Valor):", cols_restantes, key="respuesta_cl")
            mostrar_informacion_datos(df_temp, "cl")
            if st.button("🔍 Calcular", key="cl_btn"):
                df = df_temp.copy()
                df.columns = ['Fila' if col == col_fila else 'Columna' if col == col_columna else 'Tratamiento' if col == col_tratamiento else 'Valor' if col == col_respuesta else col for col in df.columns]
                df = df[['Fila', 'Columna', 'Tratamiento', 'Valor']]
                for col in ['Fila', 'Columna', 'Tratamiento']:
                    df[col] = df[col].astype('category')
                formula = 'Valor ~ C(Fila) + C(Columna) + C(Tratamiento)'
                modelo, anova = analizar_anova(df, formula, "Cuadrado Latino", "cl", "cl")
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    
                    if len(df['Tratamiento'].unique()) > 2:
                        st.subheader("🔬 Prueba de Tukey HSD para Tratamientos")
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None:
                            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                            st.dataframe(tukey_df)
                            for linea in interpretar_tukey(tukey):
                                st.write(linea)
                    
                    # Gráficos de residuos - DIRECTO
                    mostrar_graficos_residuos_directo(modelo, df, 'Tratamiento', 'Valor', "cl", "Gráficos de Residuos - Cuadrado Latino")
        elif df_temp is not None:
            st.warning("⚠️ Se necesitan al menos 4 columnas para el análisis")
    
    with col2:
        if 'cl' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['cl']
            df = resultados['df']
            modelo = resultados['modelo']
            anova = resultados['anova']
            st.subheader("📊 Visualización de Resultados")
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento')
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor', title="Medias por Tratamiento", color='Tratamiento', text_auto=True)
            fig2.update_layout(showlegend=False, height=300)
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
            pivot = df.pivot(index='Fila', columns='Columna', values='Valor')
            fig3 = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='Blues')
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)
            st.subheader("📝 Conclusiones del Análisis")
            conclusiones, medias_ordenadas = generar_conclusiones_anova(anova, modelo, df, 'Tratamiento', 'Valor')
            for conclusion in conclusiones:
                if conclusion.startswith("✅") or conclusion.startswith("🏆") or conclusion.startswith("📈"):
                    st.success(conclusion)
                elif conclusion.startswith("❌"):
                    st.error(conclusion)
                else:
                    st.info(conclusion)
            if st.button("🗑️ Limpiar resultados", key="limpiar_cl"):
                del st.session_state.resultados_anova['cl']
                st.rerun()

# ============================================================
# PESTAÑA 4: CUADRADO GRECO-LATINO
# ============================================================
with tab4:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Cuadrado Greco-Latino")
        df_temp = obtener_datos_generales("cgl", 5)
        if df_temp is not None and len(df_temp.columns) >= 5:
            st.subheader("🔧 Configuración del ANOVA")
            cols_disponibles = df_temp.columns.tolist()
            col_fila = st.selectbox("Columna de fila:", cols_disponibles, key="fila_cgl")
            cols_restantes = [c for c in cols_disponibles if c != col_fila]
            col_columna = st.selectbox("Columna de columna:", cols_restantes, key="columna_cgl")
            cols_restantes = [c for c in cols_restantes if c != col_columna]
            col_tratamiento = st.selectbox("Columna de tratamiento:", cols_restantes, key="tratamiento_cgl")
            cols_restantes = [c for c in cols_restantes if c != col_tratamiento]
            col_greco = st.selectbox("Columna de greco:", cols_restantes, key="greco_cgl")
            cols_restantes = [c for c in cols_restantes if c != col_greco]
            col_respuesta = st.selectbox("Columna de respuesta (Valor):", cols_restantes, key="respuesta_cgl")
            mostrar_informacion_datos(df_temp, "cgl")
            if st.button("🔍 Calcular", key="cgl_btn"):
                df = df_temp.copy()
                df.columns = ['Fila' if col == col_fila else 'Columna' if col == col_columna else 'Tratamiento' if col == col_tratamiento else 'Greco' if col == col_greco else 'Valor' if col == col_respuesta else col for col in df.columns]
                df = df[['Fila', 'Columna', 'Tratamiento', 'Greco', 'Valor']]
                for col in ['Fila', 'Columna', 'Tratamiento', 'Greco']:
                    df[col] = df[col].astype('category')
                formula = 'Valor ~ C(Fila) + C(Columna) + C(Tratamiento) + C(Greco)'
                modelo, anova = analizar_anova(df, formula, "Greco-Latino", "cgl", "cgl")
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    
                    if len(df['Tratamiento'].unique()) > 2:
                        st.subheader("🔬 Prueba de Tukey HSD para Tratamientos")
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None:
                            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                            st.dataframe(tukey_df)
                            for linea in interpretar_tukey(tukey):
                                st.write(linea)
                    
                    # Gráficos de residuos - DIRECTO
                    mostrar_graficos_residuos_directo(modelo, df, 'Tratamiento', 'Valor', "cgl", "Gráficos de Residuos - Greco-Latino")
        elif df_temp is not None:
            st.warning("⚠️ Se necesitan al menos 5 columnas para el análisis")
    
    with col2:
        if 'cgl' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['cgl']
            df = resultados['df']
            modelo = resultados['modelo']
            anova = resultados['anova']
            st.subheader("📊 Visualización de Resultados")
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento')
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor', title="Medias por Tratamiento", color='Tratamiento', text_auto=True)
            fig2.update_layout(showlegend=False, height=300)
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
            pivot = df.pivot(index='Fila', columns='Columna', values='Valor')
            fig3 = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='Greens')
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)
            st.subheader("📝 Conclusiones del Análisis")
            conclusiones, medias_ordenadas = generar_conclusiones_anova(anova, modelo, df, 'Tratamiento', 'Valor')
            for conclusion in conclusiones:
                if conclusion.startswith("✅") or conclusion.startswith("🏆") or conclusion.startswith("📈"):
                    st.success(conclusion)
                elif conclusion.startswith("❌"):
                    st.error(conclusion)
                else:
                    st.info(conclusion)
            if st.button("🗑️ Limpiar resultados", key="limpiar_cgl"):
                del st.session_state.resultados_anova['cgl']
                st.rerun()

# ============================================================
# PESTAÑA 5: FACTORIAL 2²
# ============================================================
with tab5:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Factorial 2²")
        df_temp = obtener_datos_generales("f22", 3)
        if df_temp is not None and len(df_temp.columns) >= 3:
            st.subheader("🔧 Configuración del ANOVA")
            col_a = st.selectbox("Factor A:", df_temp.columns.tolist(), key="a_f22")
            cols_restantes = [c for c in df_temp.columns if c != col_a]
            col_b = st.selectbox("Factor B:", cols_restantes, key="b_f22")
            cols_restantes = [c for c in cols_restantes if c != col_b]
            col_respuesta = st.selectbox("Respuesta (Y):", cols_restantes, key="respuesta_f22")
            mostrar_informacion_datos(df_temp, "f22")
            if st.button("🔍 Calcular", key="f22_btn"):
                df = df_temp.copy()
                df.columns = ['A' if col == col_a else 'B' if col == col_b else 'Y' if col == col_respuesta else col for col in df.columns]
                df = df[['A', 'B', 'Y']]
                df['A'] = df['A'].astype('category')
                df['B'] = df['B'].astype('category')
                formula = 'Y ~ A * B'
                modelo, anova = analizar_anova(df, formula, "Factorial 2²", "f22", "f22")
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    
                    if len(df['A'].unique()) > 2:
                        st.subheader("🔬 Prueba de Tukey HSD para Factor A")
                        tukey = realizar_tukey(df, 'A', 'Y')
                        if tukey is not None:
                            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                            st.dataframe(tukey_df)
                            for linea in interpretar_tukey(tukey):
                                st.write(linea)
                    
                    if len(df['B'].unique()) > 2:
                        st.subheader("🔬 Prueba de Tukey HSD para Factor B")
                        tukey = realizar_tukey(df, 'B', 'Y')
                        if tukey is not None:
                            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                            st.dataframe(tukey_df)
                            for linea in interpretar_tukey(tukey):
                                st.write(linea)
                    
                    # Gráficos de residuos - DIRECTO
                    mostrar_graficos_residuos_directo(modelo, df, 'A', 'Y', "f22", "Gráficos de Residuos - Factorial 2²")
        elif df_temp is not None:
            st.warning("⚠️ Se necesitan al menos 3 columnas para el análisis")
    
    with col2:
        if 'f22' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['f22']
            df = resultados['df']
            modelo = resultados['modelo']
            anova = resultados['anova']
            st.subheader("📊 Visualización de Resultados")
            df['Combinacion'] = df['A'].astype(str) + ',' + df['B'].astype(str)
            fig1 = px.box(df, x='Combinacion', y='Y', color='Combinacion', title="Distribución por Combinación de Factores")
            fig1.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig1, use_container_width=True)
            medias = df.groupby(['A', 'B'])['Y'].mean().reset_index()
            fig2 = px.line(medias, x='A', y='Y', color='B', markers=True, title="Gráfico de Interacción")
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)
            fig3 = px.bar(medias, x='A', y='Y', color='B', title="Medias por Combinación de Factores", barmode='group', text_auto=True)
            fig3.update_layout(height=350)
            fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)
            st.subheader("📝 Conclusiones del Análisis")
            conclusiones = generar_conclusiones_factorial(anova, modelo, df, ['A', 'B'], 'Y')
            for conclusion in conclusiones:
                if conclusion.startswith("✅") or conclusion.startswith("📈"):
                    st.success(conclusion)
                elif conclusion.startswith("❌"):
                    st.error(conclusion)
                else:
                    st.info(conclusion)
            if st.button("🗑️ Limpiar resultados", key="limpiar_f22"):
                del st.session_state.resultados_anova['f22']
                st.rerun()

# ============================================================
# PESTAÑA 6: FACTORIAL 2³
# ============================================================
with tab6:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Factorial 2³")
        df_temp = obtener_datos_generales("f23", 4)
        if df_temp is not None and len(df_temp.columns) >= 4:
            st.subheader("🔧 Configuración del ANOVA")
            col_a = st.selectbox("Factor A:", df_temp.columns.tolist(), key="a_f23")
            cols_restantes = [c for c in df_temp.columns if c != col_a]
            col_b = st.selectbox("Factor B:", cols_restantes, key="b_f23")
            cols_restantes = [c for c in cols_restantes if c != col_b]
            col_c = st.selectbox("Factor C:", cols_restantes, key="c_f23")
            cols_restantes = [c for c in cols_restantes if c != col_c]
            col_respuesta = st.selectbox("Respuesta (Y):", cols_restantes, key="respuesta_f23")
            mostrar_informacion_datos(df_temp, "f23")
            if st.button("🔍 Calcular", key="f23_btn"):
                df = df_temp.copy()
                df.columns = ['A' if col == col_a else 'B' if col == col_b else 'C' if col == col_c else 'Y' if col == col_respuesta else col for col in df.columns]
                df = df[['A', 'B', 'C', 'Y']]
                for col in ['A', 'B', 'C']:
                    df[col] = df[col].astype('category')
                formula = 'Y ~ A * B * C'
                modelo, anova = analizar_anova(df, formula, "Factorial 2³", "f23", "f23")
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    
                    for factor in ['A', 'B', 'C']:
                        if len(df[factor].unique()) > 2:
                            st.subheader(f"🔬 Prueba de Tukey HSD para Factor {factor}")
                            tukey = realizar_tukey(df, factor, 'Y')
                            if tukey is not None:
                                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                                st.dataframe(tukey_df)
                                for linea in interpretar_tukey(tukey):
                                    st.write(linea)
                    
                    # Gráficos de residuos - DIRECTO
                    mostrar_graficos_residuos_directo(modelo, df, 'A', 'Y', "f23", "Gráficos de Residuos - Factorial 2³")
        elif df_temp is not None:
            st.warning("⚠️ Se necesitan al menos 4 columnas para el análisis")
    
    with col2:
        if 'f23' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['f23']
            df = resultados['df']
            modelo = resultados['modelo']
            anova = resultados['anova']
            st.subheader("📊 Visualización de Resultados")
            df['Combinacion'] = df['A'].astype(str) + ',' + df['B'].astype(str) + ',' + df['C'].astype(str)
            fig1 = px.box(df, x='Combinacion', y='Y', color='Combinacion', title="Distribución por Combinación de Factores")
            fig1.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig1, use_container_width=True)
            medias = df.groupby(['A', 'B', 'C'])['Y'].mean().reset_index()
            fig3 = px.bar(medias, x='A', y='Y', color='B', facet_col='C', title="Medias por Combinación de Factores", barmode='group', text_auto=True)
            fig3.update_layout(height=350)
            fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)
            st.subheader("📝 Conclusiones del Análisis")
            conclusiones = generar_conclusiones_factorial(anova, modelo, df, ['A', 'B', 'C'], 'Y')
            for conclusion in conclusiones:
                if conclusion.startswith("✅") or conclusion.startswith("📈"):
                    st.success(conclusion)
                elif conclusion.startswith("❌"):
                    st.error(conclusion)
                else:
                    st.info(conclusion)
            if st.button("🗑️ Limpiar resultados", key="limpiar_f23"):
                del st.session_state.resultados_anova['f23']
                st.rerun()

# ============================================================
# PESTAÑA 7: DISEÑO 2^k
# ============================================================
with tab7:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Factorial 2^k")
        k = st.number_input("Número de factores (k):", min_value=2, max_value=8, value=3, step=1, key="k_2k")
        st.caption(f"Se esperan {k+1} columnas: k factores + respuesta (mínimo {2**k} filas)")
        
        st.subheader("🔧 Configuración del Modelo")
        opciones_interacciones = ["main - Solo efectos principales", "2fi - Interacciones de 2 factores"]
        if k >= 3:
            opciones_interacciones.append("3fi - Interacciones de 3 factores")
        if k >= 4:
            opciones_interacciones.append("4fi - Interacciones de 4 factores")
        if k >= 5:
            opciones_interacciones.append("5fi - Interacciones de 5 factores")
        if k >= 6:
            opciones_interacciones.append("6fi - Interacciones de 6 factores")
        opciones_interacciones.append("all - Todas las interacciones")
        
        nivel_interaccion = st.radio("Selecciona el nivel de interacciones:", opciones_interacciones, key="nivel_interaccion_2k")
        nivel_map = {"main - Solo efectos principales": "main", "2fi - Interacciones de 2 factores": "2fi",
                    "3fi - Interacciones de 3 factores": "3fi", "4fi - Interacciones de 4 factores": "4fi",
                    "5fi - Interacciones de 5 factores": "5fi", "6fi - Interacciones de 6 factores": "6fi",
                    "all - Todas las interacciones": "all"}
        nivel_seleccionado = nivel_map.get(nivel_interaccion, "main")
        info_modelo = obtener_info_interacciones(k, nivel_seleccionado)
        st.info(f"📌 Modelo seleccionado: {info_modelo}")
        st.caption(f"📊 Número de combinaciones posibles: {2**k}")
        
        df_temp = obtener_datos_generales("d2k", k+1)
        if df_temp is not None and len(df_temp.columns) >= k+1:
            st.subheader("🔧 Configuración de Columnas")
            factores = []
            cols_disponibles = df_temp.columns.tolist()
            for i in range(k):
                factor = st.selectbox(f"Factor {i+1}:", cols_disponibles if i == 0 else [c for c in cols_disponibles if c not in factores], key=f"factor_{i}_d2k")
                factores.append(factor)
            col_respuesta = st.selectbox("Respuesta (Y):", [c for c in cols_disponibles if c not in factores], key="respuesta_d2k")
            mostrar_informacion_datos(df_temp, "d2k")
            if st.button("🔍 Calcular", key="2k_btn"):
                df = df_temp.copy()
                nombres = [f'F{i+1}' for i in range(k)] + ['Y']
                mapeo = {factores[i]: f'F{i+1}' for i in range(k)}
                mapeo[col_respuesta] = 'Y'
                for old, new in mapeo.items():
                    df.rename(columns={old: new}, inplace=True)
                df = df[nombres]
                for i in range(k):
                    df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
                factores_list = [f'F{i+1}' for i in range(k)]
                formula = generar_formula_interacciones(factores_list, nivel_seleccionado).replace('Y ~ ', 'Y ~ ')
                modelo, anova = analizar_anova(df, formula, f"Factorial 2^{k}", "d2k", "d2k")
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    
                    for factor in factores_list:
                        if len(df[factor].unique()) > 2:
                            st.subheader(f"🔬 Prueba de Tukey HSD para {factor}")
                            tukey = realizar_tukey(df, factor, 'Y')
                            if tukey is not None:
                                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                                st.dataframe(tukey_df)
                                for linea in interpretar_tukey(tukey):
                                    st.write(linea)
                    
                    # Gráficos de residuos - DIRECTO
                    mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "d2k", "Gráficos de Residuos - Diseño 2^k")
        elif df_temp is not None:
            st.warning(f"⚠️ Se necesitan {k+1} columnas para el análisis")
    
    with col2:
        if 'd2k' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['d2k']
            df = resultados['df']
            modelo = resultados['modelo']
            anova = resultados['anova']
            k = len([col for col in df.columns if col.startswith('F')])
            factores = [f'F{i+1}' for i in range(k)]
            st.subheader("📊 Visualización de Resultados")
            n_filas = (k + 1) // 2
            st.write("**Efectos principales por factor:**")
            fig = make_subplots(rows=n_filas, cols=2, subplot_titles=factores)
            for i in range(k):
                row = i // 2 + 1
                col = i % 2 + 1
                medias = df.groupby(factores[i])['Y'].mean().reset_index()
                fig.add_trace(go.Bar(x=medias[factores[i]], y=medias['Y'], marker_color='#3498db'), row=row, col=col)
            fig.update_layout(height=min(400, 200 * n_filas), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            if k >= 2:
                st.write("**Distribución por combinación de factores:**")
                df['Combinacion'] = ''
                for factor in factores:
                    df['Combinacion'] += df[factor].astype(str) + ','
                df['Combinacion'] = df['Combinacion'].str.rstrip(',')
                fig2 = px.box(df, x='Combinacion', y='Y', color='Combinacion', title="Distribución por Combinación de Factores")
                fig2.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig2, use_container_width=True)
            st.subheader("📝 Conclusiones del Análisis")
            conclusiones = generar_conclusiones_factorial(anova, modelo, df, factores, 'Y')
            for conclusion in conclusiones:
                if conclusion.startswith("✅") or conclusion.startswith("📈"):
                    st.success(conclusion)
                elif conclusion.startswith("❌"):
                    st.error(conclusion)
                else:
                    st.info(conclusion)
            if st.button("🗑️ Limpiar resultados", key="limpiar_d2k"):
                del st.session_state.resultados_anova['d2k']
                st.rerun()

# ============================================================
# PESTAÑA 8: FRACCIONADO
# ============================================================
with tab8:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Fraccionado 2^(k-p)")
        k = st.number_input("Número de factores (k):", min_value=3, max_value=8, value=4, step=1, key="k_fracc")
        p = st.number_input("Fracción (p):", min_value=1, max_value=min(3, k-1), value=1, step=1, key="p_fracc")
        n_corridas = 2**(k-p)
        st.caption(f"📊 Diseño: 2^({k}-{p}) = {n_corridas} corridas")
        st.caption(f"Se esperan {k+1} columnas: {k} factores + respuesta")
        
        st.subheader("🔧 Configuración del Modelo")
        opciones_interacciones_fracc = ["main - Solo efectos principales", "2fi - Interacciones de 2 factores"]
        if k >= 3:
            opciones_interacciones_fracc.append("3fi - Interacciones de 3 factores")
        if k >= 4:
            opciones_interacciones_fracc.append("4fi - Interacciones de 4 factores")
        if k >= 5:
            opciones_interacciones_fracc.append("5fi - Interacciones de 5 factores")
        opciones_interacciones_fracc.append("all - Todas las interacciones")
        
        nivel_interaccion_fracc = st.radio("Selecciona el nivel de interacciones:", opciones_interacciones_fracc, key="nivel_interaccion_fracc")
        nivel_map_fracc = {"main - Solo efectos principales": "main", "2fi - Interacciones de 2 factores": "2fi",
                          "3fi - Interacciones de 3 factores": "3fi", "4fi - Interacciones de 4 factores": "4fi",
                          "5fi - Interacciones de 5 factores": "5fi", "all - Todas las interacciones": "all"}
        nivel_seleccionado_fracc = nivel_map_fracc.get(nivel_interaccion_fracc, "main")
        info_modelo_fracc = obtener_info_interacciones(k, nivel_seleccionado_fracc)
        st.info(f"📌 Modelo seleccionado: {info_modelo_fracc}")
        st.info(f"📌 Resolución del diseño: {k-p} (2^({k-p}) = {n_corridas} corridas)")
        
        df_temp = obtener_datos_generales("fracc", k+1)
        if df_temp is not None and len(df_temp.columns) >= k+1:
            st.subheader("🔧 Configuración de Columnas")
            factores = []
            cols_disponibles = df_temp.columns.tolist()
            for i in range(k):
                factor = st.selectbox(f"Factor {i+1}:", cols_disponibles if i == 0 else [c for c in cols_disponibles if c not in factores], key=f"factor_{i}_fracc")
                factores.append(factor)
            col_respuesta = st.selectbox("Respuesta (Y):", [c for c in cols_disponibles if c not in factores], key="respuesta_fracc")
            mostrar_informacion_datos(df_temp, "fracc")
            if st.button("🔍 Calcular", key="fracc_btn"):
                df = df_temp.copy()
                nombres = [f'F{i+1}' for i in range(k)] + ['Y']
                mapeo = {factores[i]: f'F{i+1}' for i in range(k)}
                mapeo[col_respuesta] = 'Y'
                for old, new in mapeo.items():
                    df.rename(columns={old: new}, inplace=True)
                df = df[nombres]
                for i in range(k):
                    df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
                factores_list = [f'F{i+1}' for i in range(k)]
                formula = generar_formula_interacciones(factores_list, nivel_seleccionado_fracc).replace('Y ~ ', 'Y ~ ')
                modelo, anova = analizar_anova(df, formula, f"Fraccionado 2^({k}-{p})", "fracc", "fracc")
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    
                    for factor in factores_list:
                        if len(df[factor].unique()) > 2:
                            st.subheader(f"🔬 Prueba de Tukey HSD para {factor}")
                            tukey = realizar_tukey(df, factor, 'Y')
                            if tukey is not None:
                                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                                st.dataframe(tukey_df)
                                for linea in interpretar_tukey(tukey):
                                    st.write(linea)
                    
                    # Gráficos de residuos - DIRECTO
                    mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "fracc", "Gráficos de Residuos - Fraccionado")
        elif df_temp is not None:
            st.warning(f"⚠️ Se necesitan {k+1} columnas para el análisis")
    
    with col2:
        if 'fracc' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['fracc']
            df = resultados['df']
            modelo = resultados['modelo']
            anova = resultados['anova']
            k = len([col for col in df.columns if col.startswith('F')])
            factores = [f'F{i+1}' for i in range(k)]
            p = k - int(np.log2(len(df)))
            if p < 1:
                p = 1
            st.subheader("📊 Visualización de Resultados")
            st.info(f"📌 Diseño Fraccionado 2^({k}-{p}) - {2**(k-p)} corridas")
            st.write("**Efectos principales por factor:**")
            n_filas = (len(factores) + 1) // 2
            fig1 = make_subplots(rows=n_filas, cols=2, subplot_titles=factores)
            for i in range(len(factores)):
                row = i // 2 + 1
                col = i % 2 + 1
                medias = df.groupby(factores[i])['Y'].mean().reset_index()
                fig1.add_trace(go.Bar(x=medias[factores[i]], y=medias['Y'], marker_color='#3498db'), row=row, col=col)
            fig1.update_layout(height=min(400, 200 * n_filas), showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
            if len(factores) >= 2:
                st.write("**Distribución por combinación de factores:**")
                df['Combinacion'] = ''
                for factor in factores:
                    df['Combinacion'] += df[factor].astype(str) + ','
                df['Combinacion'] = df['Combinacion'].str.rstrip(',')
                fig2 = px.box(df, x='Combinacion', y='Y', color='Combinacion', title="Distribución por Combinación de Factores")
                fig2.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig2, use_container_width=True)
            st.subheader("📝 Conclusiones del Análisis")
            conclusiones = generar_conclusiones_factorial(anova, modelo, df, factores, 'Y')
            for conclusion in conclusiones:
                if conclusion.startswith("✅") or conclusion.startswith("📈"):
                    st.success(conclusion)
                elif conclusion.startswith("❌"):
                    st.error(conclusion)
                else:
                    st.info(conclusion)
            if st.button("🗑️ Limpiar resultados", key="limpiar_fracc"):
                del st.session_state.resultados_anova['fracc']
                st.rerun()

# ============================================================
# PESTAÑA 9: MSR
# ============================================================
with tab9:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Metodología de Superficie de Respuesta")
        n_fact = st.number_input("Número de factores:", min_value=2, max_value=4, value=2, step=1, key="n_fact_msr")
        st.caption(f"Se esperan {n_fact+1} columnas")
        df_temp = obtener_datos_generales("msr", n_fact+1)
        if df_temp is not None and len(df_temp.columns) >= n_fact+1:
            st.subheader("🔧 Configuración del ANOVA")
            factores = []
            cols_disponibles = df_temp.columns.tolist()
            for i in range(n_fact):
                factor = st.selectbox(f"Factor {i+1}:", cols_disponibles if i == 0 else [c for c in cols_disponibles if c not in factores], key=f"factor_{i}_msr")
                factores.append(factor)
            col_respuesta = st.selectbox("Respuesta (Y):", [c for c in cols_disponibles if c not in factores], key="respuesta_msr")
            mostrar_informacion_datos(df_temp, "msr")
            if st.button("🔍 Calcular MSR", key="msr_btn"):
                try:
                    nf = int(n_fact)
                    df = df_temp.copy()
                    nombres = [f'F{i+1}' for i in range(nf)] + ['Y']
                    mapeo = {factores[i]: f'F{i+1}' for i in range(nf)}
                    mapeo[col_respuesta] = 'Y'
                    for old, new in mapeo.items():
                        df.rename(columns={old: new}, inplace=True)
                    df = df[nombres]
                    vars_list = [f'F{i+1}' for i in range(nf)]
                    formula = 'Y ~ ' + ' + '.join(vars_list)
                    cuadraticos = [f'I({v}**2)' for v in vars_list]
                    formula += ' + ' + ' + '.join(cuadraticos)
                    if nf >= 2:
                        interacciones = []
                        for i in range(nf):
                            for j in range(i+1, nf):
                                interacciones.append(f'F{i+1}:F{j+1}')
                        formula += ' + ' + ' + '.join(interacciones)
                    modelo, anova = analizar_anova(df, formula, "MSR", "msr", "msr")
                    if modelo is not None:
                        st.success("✅ ANOVA completado exitosamente")
                        
                        # Gráficos de residuos - DIRECTO
                        mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "msr", "Gráficos de Residuos - MSR")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        elif df_temp is not None:
            st.warning(f"⚠️ Se necesitan {n_fact+1} columnas para el análisis")
    
    with col2:
        if 'msr' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['msr']
            df = resultados['df']
            st.subheader("📊 Visualización de Resultados")
            if len(df.columns) == 3:
                fig = px.scatter_3d(df, x='F1', y='F2', z='Y', title="Superficie de Respuesta")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            if st.button("🗑️ Limpiar resultados", key="limpiar_msr"):
                del st.session_state.resultados_anova['msr']
                st.rerun()

# ============================================================
# PESTAÑA 10: TAGUCHI
# ============================================================
with tab10:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Robusto Taguchi")
        arreglo = st.selectbox("Arreglo Ortogonal:", ["L4 (2³)", "L8 (2⁷)", "L9 (3⁴)", "L16 (2¹⁵)", "L27 (3¹³)"], index=1, key="taguchi_arreglo")
        n_fact = {"L4 (2³)": 3, "L8 (2⁷)": 7, "L9 (3⁴)": 4, "L16 (2¹⁵)": 15, "L27 (3¹³)": 13}[arreglo]
        st.caption(f"Se esperan {n_fact+1} columnas")
        df_temp = obtener_datos_generales("taguchi", n_fact+1)
        if df_temp is not None and len(df_temp.columns) >= n_fact+1:
            st.subheader("🔧 Configuración del ANOVA")
            factores = []
            cols_disponibles = df_temp.columns.tolist()
            for i in range(min(n_fact, 8)):
                factor = st.selectbox(f"Factor {i+1}:", cols_disponibles if i == 0 else [c for c in cols_disponibles if c not in factores], key=f"factor_{i}_taguchi")
                factores.append(factor)
            if n_fact > 8:
                st.info(f"📌 Se seleccionarán automáticamente los factores restantes")
            col_respuesta = st.selectbox("Respuesta (Y):", [c for c in cols_disponibles if c not in factores], key="respuesta_taguchi")
            mostrar_informacion_datos(df_temp, "taguchi")
            if st.button("🔍 Calcular Taguchi", key="taguchi_btn"):
                df = df_temp.copy()
                nombres = [f'F{i+1}' for i in range(n_fact)] + ['Y']
                mapeo = {factores[i]: f'F{i+1}' for i in range(min(n_fact, 8))}
                restantes = [c for c in cols_disponibles if c not in factores and c != col_respuesta]
                for i in range(8, n_fact):
                    if restantes:
                        mapeo[restantes.pop(0)] = f'F{i+1}'
                mapeo[col_respuesta] = 'Y'
                for old, new in mapeo.items():
                    df.rename(columns={old: new}, inplace=True)
                df = df[nombres]
                for i in range(n_fact):
                    df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
                factores_list = [f'F{i+1}' for i in range(n_fact)]
                formula = 'Y ~ ' + ' + '.join(factores_list)
                modelo, anova = analizar_anova(df, formula, "Taguchi", "taguchi", "taguchi")
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    
                    for factor in factores_list:
                        if len(df[factor].unique()) > 2:
                            st.subheader(f"🔬 Prueba de Tukey HSD para {factor}")
                            tukey = realizar_tukey(df, factor, 'Y')
                            if tukey is not None:
                                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                                st.dataframe(tukey_df)
                                for linea in interpretar_tukey(tukey):
                                    st.write(linea)
                    
                    # Gráficos de residuos - DIRECTO
                    mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "taguchi", "Gráficos de Residuos - Taguchi")
        elif df_temp is not None:
            st.warning(f"⚠️ Se necesitan {n_fact+1} columnas para el análisis")
    
    with col2:
        if 'taguchi' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['taguchi']
            df = resultados['df']
            k = len([col for col in df.columns if col.startswith('F')])
            st.subheader("📊 Visualización de Resultados")
            n_mostrar = min(k, 6)
            n_filas = (n_mostrar + 1) // 2
            fig = make_subplots(rows=n_filas, cols=2, subplot_titles=[f'Factor {i+1}' for i in range(n_mostrar)])
            for i in range(n_mostrar):
                row = i // 2 + 1
                col = i % 2 + 1
                factor = f'F{i+1}'
                medias = df.groupby(factor)['Y'].mean().reset_index()
                fig.add_trace(go.Bar(x=medias[factor], y=medias['Y']), row=row, col=col)
            fig.update_layout(height=min(400, 200 * n_filas), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            if st.button("🗑️ Limpiar resultados", key="limpiar_taguchi"):
                del st.session_state.resultados_anova['taguchi']
                st.rerun()

# ============================================================
# PESTAÑA 11: OPTIMIZACIÓN
# ============================================================
with tab11:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Optimización Múltiple")
        st.caption("Se esperan 4 columnas: 2 factores + 2 respuestas")
        df_temp = obtener_datos_generales("omr", 4)
        if df_temp is not None and len(df_temp.columns) >= 4:
            st.subheader("🔧 Configuración de la Optimización")
            col_f1 = st.selectbox("Factor 1:", df_temp.columns.tolist(), key="f1_omr")
            cols_restantes = [c for c in df_temp.columns if c != col_f1]
            col_f2 = st.selectbox("Factor 2:", cols_restantes, key="f2_omr")
            cols_restantes = [c for c in cols_restantes if c != col_f2]
            col_y1 = st.selectbox("Respuesta 1 (maximizar):", cols_restantes, key="y1_omr")
            cols_restantes = [c for c in cols_restantes if c != col_y1]
            col_y2 = st.selectbox("Respuesta 2 (minimizar):", cols_restantes, key="y2_omr")
            mostrar_informacion_datos(df_temp, "omr")
            if st.button("🔍 Optimizar", key="omr_btn"):
                try:
                    df = df_temp.copy()
                    df.columns = ['F1' if col == col_f1 else 'F2' if col == col_f2 else 'Y1' if col == col_y1 else 'Y2' if col == col_y2 else col for col in df.columns]
                    df = df[['F1', 'F2', 'Y1', 'Y2']]
                    modelo1 = ols('Y1 ~ F1 + F2 + I(F1**2) + I(F2**2) + F1:F2', data=df).fit()
                    modelo2 = ols('Y2 ~ F1 + F2 + I(F1**2) + I(F2**2) + F1:F2', data=df).fit()
                    grid = pd.DataFrame([(f1, f2) for f1 in np.linspace(-1.5, 1.5, 20) for f2 in np.linspace(-1.5, 1.5, 20)], columns=['F1', 'F2'])
                    grid['Y1'] = modelo1.predict(grid)
                    grid['Y2'] = modelo2.predict(grid)
                    grid['D1'] = ((grid['Y1'] - grid['Y1'].min()) / (grid['Y1'].max() - grid['Y1'].min())).clip(0, 1)
                    grid['D2'] = (1 - (grid['Y2'] - grid['Y2'].min()) / (grid['Y2'].max() - grid['Y2'].min())).clip(0, 1)
                    grid['D'] = (grid['D1'] * grid['D2']) ** 0.5
                    optimo = grid.loc[grid['D'].idxmax()]
                    st.success("✅ Optimización completada")
                    st.write(f"**Óptimo:** F1={optimo['F1']:.3f}, F2={optimo['F2']:.3f}")
                    st.write(f"**Deseabilidad = {optimo['D']:.4f}**")
                    st.session_state['omr_resultado'] = {'grid': grid, 'optimo': optimo}
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        elif df_temp is not None:
            st.warning("⚠️ Se necesitan 4 columnas para el análisis")
    
    with col2:
        if 'omr_resultado' in st.session_state:
            grid = st.session_state['omr_resultado']['grid']
            optimo = st.session_state['omr_resultado']['optimo']
            pivot = grid.pivot(index='F1', columns='F2', values='D')
            fig = px.imshow(pivot, aspect="auto", color_continuous_scale='Blues')
            fig.add_scatter(x=[optimo['F2']], y=[optimo['F1']], mode='markers', marker=dict(color='red', size=15, symbol='x'))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            if st.button("🗑️ Limpiar resultados", key="limpiar_omr"):
                del st.session_state['omr_resultado']
                st.rerun()

# ============================================================
# PESTAÑA 12: ADITIVOS
# ============================================================
with tab12:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseños Añadidos")
        st.caption("Se esperan 4 columnas: 3 componentes + respuesta")
        df_temp = obtener_datos_generales("aditivos", 4)
        if df_temp is not None and len(df_temp.columns) >= 4:
            st.subheader("🔧 Configuración del ANOVA")
            col_c1 = st.selectbox("Componente 1:", df_temp.columns.tolist(), key="c1_aditivos")
            cols_restantes = [c for c in df_temp.columns if c != col_c1]
            col_c2 = st.selectbox("Componente 2:", cols_restantes, key="c2_aditivos")
            cols_restantes = [c for c in cols_restantes if c != col_c2]
            col_c3 = st.selectbox("Componente 3:", cols_restantes, key="c3_aditivos")
            cols_restantes = [c for c in cols_restantes if c != col_c3]
            col_respuesta = st.selectbox("Respuesta (Y):", cols_restantes, key="respuesta_aditivos")
            mostrar_informacion_datos(df_temp, "aditivos")
            if st.button("🔍 Calcular", key="aditivos_btn"):
                try:
                    df = df_temp.copy()
                    df.columns = ['C1' if col == col_c1 else 'C2' if col == col_c2 else 'C3' if col == col_c3 else 'Y' if col == col_respuesta else col for col in df.columns]
                    df = df[['C1', 'C2', 'C3', 'Y']]
                    formula = 'Y ~ C1 + C2 + C3'
                    modelo = ols(formula, data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ Modelo aditivo ajustado")
                    st.dataframe(anova_table)
                    st.write(f"**R² = {modelo.rsquared:.6f}**")
                    st.session_state['aditivos_resultado'] = {'df': df, 'modelo': modelo}
                    
                    for comp in ['C1', 'C2', 'C3']:
                        if len(df[comp].unique()) > 2:
                            st.subheader(f"🔬 Prueba de Tukey HSD para {comp}")
                            tukey = realizar_tukey(df, comp, 'Y')
                            if tukey is not None:
                                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                                st.dataframe(tukey_df)
                                for linea in interpretar_tukey(tukey):
                                    st.write(linea)
                    
                    # Gráficos de residuos - DIRECTO
                    mostrar_graficos_residuos_directo(modelo, df, 'C1', 'Y', "aditivos", "Gráficos de Residuos - Aditivos")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        elif df_temp is not None:
            st.warning("⚠️ Se necesitan 4 columnas para el análisis")
    
    with col2:
        if 'aditivos_resultado' in st.session_state:
            df = st.session_state['aditivos_resultado']['df']
            fig = make_subplots(rows=1, cols=3, subplot_titles=['C1', 'C2', 'C3'])
            for i in range(3):
                comp = f'C{i+1}'
                medias = df.groupby(comp)['Y'].mean().reset_index()
                fig.add_trace(go.Scatter(x=medias[comp], y=medias['Y'], mode='lines+markers'), row=1, col=i+1)
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            if st.button("🗑️ Limpiar resultados", key="limpiar_aditivos"):
                del st.session_state['aditivos_resultado']
                st.rerun()

# ============================================================
# PESTAÑA 13: SPLIT-PLOT
# ============================================================
with tab13:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Parcelas Divididas")
        st.caption("Se esperan 4 columnas: 3 factores + respuesta")
        df_temp = obtener_datos_generales("split", 4)
        if df_temp is not None and len(df_temp.columns) >= 4:
            st.subheader("🔧 Configuración del ANOVA")
            col_f1 = st.selectbox("Factor 1:", df_temp.columns.tolist(), key="f1_split")
            cols_restantes = [c for c in df_temp.columns if c != col_f1]
            col_f2 = st.selectbox("Factor 2:", cols_restantes, key="f2_split")
            cols_restantes = [c for c in cols_restantes if c != col_f2]
            col_f3 = st.selectbox("Factor 3:", cols_restantes, key="f3_split")
            cols_restantes = [c for c in cols_restantes if c != col_f3]
            col_respuesta = st.selectbox("Respuesta (Y):", cols_restantes, key="respuesta_split")
            mostrar_informacion_datos(df_temp, "split")
            if st.button("🔍 Calcular", key="split_btn"):
                try:
                    df = df_temp.copy()
                    df.columns = ['F1' if col == col_f1 else 'F2' if col == col_f2 else 'F3' if col == col_f3 else 'Y' if col == col_respuesta else col for col in df.columns]
                    df = df[['F1', 'F2', 'F3', 'Y']]
                    for col in ['F1', 'F2', 'F3']:
                        df[col] = df[col].astype('category')
                    formula = 'Y ~ F1 * F2 * F3'
                    modelo, anova = analizar_anova(df, formula, "Split-Plot", "split", "split")
                    if modelo is not None:
                        st.success("✅ ANOVA completado exitosamente")
                        
                        for factor in ['F1', 'F2', 'F3']:
                            if len(df[factor].unique()) > 2:
                                st.subheader(f"🔬 Prueba de Tukey HSD para {factor}")
                                tukey = realizar_tukey(df, factor, 'Y')
                                if tukey is not None:
                                    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                                    st.dataframe(tukey_df)
                                    for linea in interpretar_tukey(tukey):
                                        st.write(linea)
                        
                        # Gráficos de residuos - DIRECTO
                        mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "split", "Gráficos de Residuos - Split-Plot")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        elif df_temp is not None:
            st.warning("⚠️ Se necesitan 4 columnas para el análisis")
    
    with col2:
        if 'split' in st.session_state.resultados_anova:
            resultados = st.session_state.resultados_anova['split']
            df = resultados['df']
            st.subheader("📊 Visualización de Resultados")
            fig = px.box(df, x='F1', y='Y', color='F2', facet_col='F3')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            if st.button("🗑️ Limpiar resultados", key="limpiar_split"):
                del st.session_state.resultados_anova['split']
                st.rerun()

print("✅ app.py creado exitosamente!")
