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
if 'pruebas_multiples' not in st.session_state:
    st.session_state.pruebas_multiples = {}

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def selector_tipo_anova(key_suffix="", show_help=True):
    if show_help:
        st.markdown("""
        **📊 Tipos de ANOVA:**
        - **Tipo I (Secuencial):** El orden de entrada importa.
        - **Tipo II (Parcial):** Recomendado para diseños balanceados.
        - **Tipo III (Parcial con interacción):** Útil para diseños no balanceados.
        """)

    tipo_anova = st.radio(
        "Selecciona el tipo de ANOVA:",
        ["Tipo I (Secuencial)", "Tipo II (Parcial)", "Tipo III (Parcial con interacción)"],
        horizontal=True,
        key=f"tipo_anova_{key_suffix}"
    )

    tipo_map = {
        "Tipo I (Secuencial)": 1,
        "Tipo II (Parcial)": 2,
        "Tipo III (Parcial con interacción)": 3
    }
    return tipo_map[tipo_anova]


def analizar_anova_mejorada(df, formula, titulo="ANOVA", key_suffix="", tipo="dca", anova_type=2):
    try:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')

        modelo = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(modelo, typ=anova_type)

        if 'mean_sq' not in anova_table.columns:
            anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']

        rename_map = {
            'sum_sq': 'Sum Sq',
            'df': 'DF',
            'mean_sq': 'Mean Sq',
            'F': 'F-Ratio',
            'PR(>F)': 'P-Value'
        }
        anova_table = anova_table.rename(columns=rename_map)

        columnas_orden = [c for c in anova_table.columns if c not in ['Mean Sq']]
        if 'DF' in columnas_orden:
            idx_df = columnas_orden.index('DF')
            columnas_orden.insert(idx_df + 1, 'Mean Sq')
            anova_table = anova_table[columnas_orden]

        st.success(f"✅ {titulo} calculado (Tipo {anova_type})")
        st.dataframe(anova_table.style.format({
            'Sum Sq': '{:.4f}',
            'Mean Sq': '{:.4f}',
            'F-Ratio': '{:.4f}',
            'P-Value': '{:.6f}'
        }))

        st.write(f"**R² = {modelo.rsquared:.6f}**")
        st.write(f"**R² ajustado = {modelo.rsquared_adj:.6f}**")

        st.session_state.resultados_anova[key_suffix] = {
            'modelo': modelo,
            'anova': anova_table,
            'df': df,
            'tipo': tipo,
            'formula': formula,
            'anova_type': anova_type
        }

        return modelo, anova_table

    except Exception as e:
        st.error(f"❌ Error en ANOVA: {str(e)}")
        return None, None


def graficos_interaccion_multifactorial(df, factores, respuesta, max_pares=6):
    if len(factores) < 2:
        return None

    pares = list(combinations(factores, 2))[:max_pares]
    n_filas = (len(pares) + 1) // 2
    fig = make_subplots(rows=n_filas, cols=2,
                        subplot_titles=[f"{f1} × {f2}" for f1, f2 in pares])

    for idx, (f1, f2) in enumerate(pares):
        row = idx // 2 + 1
        col = idx % 2 + 1
        medias = df.groupby([f1, f2])[respuesta].mean().reset_index()
        for nivel in medias[f2].unique():
            datos_nivel = medias[medias[f2] == nivel]
            fig.add_trace(
                go.Scatter(x=datos_nivel[f1], y=datos_nivel[respuesta],
                           mode='lines+markers', name=f"{f2}={nivel}",
                           legendgroup=f"{f2}={nivel}", showlegend=(idx == 0)),
                row=row, col=col
            )
        fig.update_xaxes(title_text=f1, row=row, col=col)
        fig.update_yaxes(title_text=respuesta, row=row, col=col)

    fig.update_layout(height=min(400, 200 * n_filas), showlegend=True,
                      title_text="Gráficos de Interacción")
    return fig


def graficos_interaccion_heatmap(df, factores, respuesta):
    if len(factores) < 2:
        return None
    try:
        f1, f2 = factores[0], factores[1]
        pivot = df.pivot_table(index=f1, columns=f2, values=respuesta, aggfunc='mean')
        fig = px.imshow(pivot, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title=f"Mapa de Calor: {f1} × {f2}")
        fig.update_layout(height=400)
        return fig
    except:
        return None


def graficar_boxplot_por_factor(df, factores, respuesta):
    if not factores:
        return None
    n_factores = len(factores)
    fig = make_subplots(rows=1, cols=n_factores, subplot_titles=[f"{f}" for f in factores])
    for i, factor in enumerate(factores):
        fig.add_trace(
            go.Box(y=df[respuesta], x=df[factor], name=factor,
                   marker_color='#3498db', boxpoints='all', jitter=0.3, pointpos=-1.8),
            row=1, col=i+1
        )
        fig.update_xaxes(title_text=factor, row=1, col=i+1)
        fig.update_yaxes(title_text=respuesta, row=1, col=i+1)
    fig.update_layout(height=400, showlegend=False, title_text="Boxplot por Factor")
    return fig


def graficar_sumas_cuadrados(anova_table):
    try:
        df_sc = anova_table.reset_index().rename(columns={'index': 'Fuente'})
        col_sc = 'Sum Sq' if 'Sum Sq' in df_sc.columns else 'sum_sq' if 'sum_sq' in df_sc.columns else None
        if not col_sc:
            return None
        if 'Total' in df_sc['Fuente'].values:
            df_sc = df_sc[df_sc['Fuente'] != 'Total']
        fig = px.bar(df_sc, x='Fuente', y=col_sc,
                     title="Gráfico de Sumas de Cuadrados",
                     color='Fuente', text_auto='.2f')
        if df_sc[col_sc].max() / max(df_sc[col_sc].min(), 0.001) > 100:
            fig.update_yaxes(type="log")
        fig.update_layout(showlegend=False, height=400)
        fig.update_traces(textposition='outside')
        return fig
    except:
        return None


def graficar_verificacion_modelo_completa(modelo, df, factor, respuesta):
    if modelo is None:
        return None
    residuos = modelo.resid
    predichos = modelo.fittedvalues
    run_order = np.arange(1, len(residuos) + 1)

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Q-Q (Normalidad)", "Residuos vs Ajustados",
        "Residuos vs Orden", "Histograma de Residuos"
    ])

    from scipy.stats import probplot
    (osm, osr), (slope, intercept, r) = probplot(residuos, dist="norm", fit=True)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers',
                             marker=dict(color='blue', size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[min(osm), max(osm)],
                             y=[min(osm)*slope + intercept, max(osm)*slope + intercept],
                             mode='lines', line=dict(color='red', dash='dash')), row=1, col=1)

    fig.add_trace(go.Scatter(x=predichos, y=residuos, mode='markers',
                             marker=dict(color='green', size=6)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    fig.add_trace(go.Scatter(x=run_order, y=residuos, mode='lines+markers',
                             marker=dict(color='orange', size=6)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

    fig.add_trace(go.Histogram(x=residuos, marker_color='purple', nbinsx=15), row=2, col=2)

    fig.update_layout(height=700, showlegend=False,
                      title_text="Verificación de Supuestos")
    return fig


def graficar_interaccion_mejorada(df, f1, f2, respuesta):
    medias = df.groupby([f1, f2])[respuesta].mean().reset_index()
    fig = px.line(medias, x=f1, y=respuesta, color=f2, markers=True,
                  title=f"Interacción {f1} × {f2}")
    for i, row in medias.iterrows():
        fig.add_annotation(x=row[f1], y=row[respuesta],
                           text=f"{row[respuesta]:.2f}",
                           showarrow=False, yshift=10, font=dict(size=10))
    fig.update_layout(height=400)
    return fig


# ============================================================
# PRUEBAS MÚLTIPLES
# ============================================================

def realizar_tukey(df, factor, respuesta, alpha=0.05):
    try:
        df[factor] = df[factor].astype('category')
        return pairwise_tukeyhsd(df[respuesta], df[factor], alpha=alpha)
    except:
        return None


def interpretar_tukey(tukey):
    interpretacion = []
    if tukey is None:
        return ["❌ No se pudo realizar la prueba de Tukey"]
    try:
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        significativas = tukey_df[tukey_df['reject'] == True]
        if len(significativas) == 0:
            interpretacion.append("📊 **Tukey HSD:** No hay diferencias significativas")
            return interpretacion
        interpretacion.append(f"📊 **Tukey HSD:** {len(significativas)} diferencias significativas")
        for _, row in significativas.iterrows():
            interpretacion.append(f"  • {row['group1']} vs {row['group2']}: Diff = {row['meandiff']:.4f} (p = {row['p-adj']:.6f})")
    except Exception as e:
        interpretacion.append(f"❌ Error: {str(e)}")
    return interpretacion


# ============================================================
# FUNCIÓN LSD CORREGIDA - Recibe MSE y GL del modelo completo
# ============================================================

def realizar_lsd(df, factor, respuesta, mse, gl_error, alpha=0.05):
    """
    Realiza prueba LSD usando el MSE y GL del error del modelo completo.
    """
    try:
        if mse is None or gl_error is None:
            return None

        medias = df.groupby(factor)[respuesta].mean()
        grupos = medias.index.tolist()
        t_crit = stats.t.ppf(1 - alpha/2, gl_error)

        resultados = []
        for i in range(len(grupos)):
            for j in range(i+1, len(grupos)):
                g1, g2 = grupos[i], grupos[j]
                diff = medias[g1] - medias[g2]
                n1 = len(df[df[factor] == g1])
                n2 = len(df[df[factor] == g2])
                se_diff = np.sqrt(mse * (1/n1 + 1/n2))
                t_val = diff / se_diff
                p_val = 2 * (1 - stats.t.cdf(abs(t_val), gl_error))
                lsd = t_crit * se_diff
                significativo = p_val < alpha
                resultados.append({
                    'group1': g1, 'group2': g2, 'meandiff': diff,
                    'p-adj': p_val, 'lower': diff - lsd, 'upper': diff + lsd,
                    'reject': significativo
                })

        return pd.DataFrame(resultados)
    except Exception as e:
        st.error(f"❌ Error en LSD: {str(e)}")
        return None


def interpretar_lsd(lsd_df):
    interpretacion = []
    if lsd_df is None or len(lsd_df) == 0:
        return ["❌ No se pudo realizar la prueba de LSD"]
    try:
        significativas = lsd_df[lsd_df['reject'] == True]
        if len(significativas) == 0:
            interpretacion.append("📊 **LSD:** No hay diferencias significativas")
            return interpretacion
        interpretacion.append(f"📊 **LSD:** {len(significativas)} diferencias significativas")
        for _, row in significativas.iterrows():
            interpretacion.append(f"  • {row['group1']} vs {row['group2']}: Diff = {row['meandiff']:.4f} (p = {row['p-adj']:.6f})")
    except Exception as e:
        interpretacion.append(f"❌ Error: {str(e)}")
    return interpretacion


def extraer_mse_gl(anova_table):
    """Extrae el MSE y GL del error del modelo completo desde la tabla ANOVA."""
    try:
        if 'Residual' in anova_table.index:
            return anova_table.loc['Residual', 'Mean Sq'], anova_table.loc['Residual', 'DF']
        else:
            return anova_table['Mean Sq'].iloc[-1], anova_table['DF'].iloc[-1]
    except:
        return None, None


# ============================================================
# GRÁFICOS DE RESIDUOS - DIRECTO (CON PANELES DE NORMALIDAD MEJORADOS)
# ============================================================

def mostrar_graficos_residuos_directo(modelo, df, factor=None, respuesta=None, key_suffix="", titulo="Residuos"):
    if modelo is None:
        st.warning("⚠️ No hay modelo para generar gráficos de residuos")
        return
    
    st.subheader(f"📈 {titulo}")
    
    with st.spinner("Generando gráficos de verificación del modelo..."):
        # 1. Gráficos de residuos (Q-Q, Homocedasticidad, Independencia, Histograma)
        fig = graficar_verificacion_modelo_completa(modelo, df, factor, respuesta)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # 2. Pruebas de normalidad con PANELES GRANDES
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
# DISEÑO INTERACTIVO
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
            if resp_data.get('valores'):
                valores = resp_data['valores']
                df[resp_name] = (valores * (len(df) // len(valores) + 1))[:len(df)]
            else:
                df[resp_name] = np.random.normal(10, 2, len(df))
        return df
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def mostrar_definicion_factores():
    st.subheader("🔧 Definición de Factores")
    n_factores = st.number_input("Número de factores:", min_value=1, max_value=10,
                                  value=3, step=1, key="n_factores_def")
    st.markdown("---")
    factores_def = {}
    for i in range(n_factores):
        st.subheader(f"📌 Factor {chr(65+i)}")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            nombre = st.text_input(f"Nombre:", value=f"Factor_{chr(65+i)}", key=f"nombre_factor_{i}")
        with col2:
            n_niveles = st.number_input(f"Niveles:", min_value=2, max_value=10, value=3, step=1, key=f"niveles_factor_{i}")
        with col3:
            tipo = st.selectbox("Tipo:", ["Cualitativo", "Cuantitativo"], key=f"tipo_factor_{i}")
        niveles = []
        cols = st.columns(min(n_niveles, 5))
        for j in range(n_niveles):
            with cols[j % len(cols)]:
                nivel = st.text_input(f"Nivel {j+1}:", value=str(j+1), key=f"nivel_{i}_{j}")
                niveles.append(nivel)
        factores_def[nombre] = {'niveles': niveles, 'tipo': tipo, 'n_niveles': n_niveles}
        st.markdown("---")
    return factores_def


def mostrar_definicion_respuestas():
    st.subheader("📊 Definición de Respuestas")
    n_respuestas = st.number_input("Número de respuestas:", min_value=1, max_value=5,
                                    value=1, step=1, key="n_respuestas_def")
    st.markdown("---")
    respuestas_def = {}
    for i in range(n_respuestas):
        st.subheader(f"📈 Respuesta {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            nombre = st.text_input(f"Nombre:", value=f"Respuesta_{i+1}", key=f"nombre_respuesta_{i}")
        with col2:
            n_valores = st.number_input("Valores predefinidos:", min_value=0, max_value=50, value=0, step=1, key=f"n_valores_respuesta_{i}")
        if n_valores > 0:
            valores = []
            cols = st.columns(min(n_valores, 5))
            for j in range(n_valores):
                with cols[j % len(cols)]:
                    valor = st.text_input(f"Valor {j+1}:", value=str(np.random.uniform(5, 15)), key=f"valor_respuesta_{i}_{j}")
                    try:
                        valores.append(float(valor))
                    except:
                        valores.append(10.0)
            respuestas_def[nombre] = {'valores': valores}
        else:
            respuestas_def[nombre] = {'valores': None}
        st.markdown("---")
    return respuestas_def


def mostrar_configuracion_diseno():
    st.subheader("⚙️ Configuración")
    col1, col2 = st.columns(2)
    with col1:
        replicas = st.number_input("Réplicas:", min_value=1, max_value=5, value=1, step=1, key="replicas_diseno")
    with col2:
        randomizar = st.checkbox("Randomizar", value=True, key="randomizar_diseno")
    return replicas, randomizar


def mostrar_resumen_diseno(df, factores_def, respuestas_def):
    st.subheader("📋 Resumen del Diseño")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Corridas", len(df))
    with col2:
        st.metric("Factores", len(factores_def))
    with col3:
        st.metric("Respuestas", len(respuestas_def))
    st.dataframe(df.head(10))


# ============================================================
# CONCLUSIONES
# ============================================================

def generar_conclusiones_anova(anova_table, modelo, df, factor, respuesta, alpha=0.05):
    conclusiones = []
    p_valor = None
    for col in anova_table.columns:
        if 'P-Value' in col:
            p_valor = anova_table[col].iloc[0]
            break
    if p_valor is None:
        return ["⚠️ No se obtuvo p-valor"], None

    if p_valor < alpha:
        conclusiones.append(f"✅ **Conclusión:** Diferencia significativa (p = {p_valor:.6f})")
    else:
        conclusiones.append(f"❌ **Conclusión:** Sin diferencia significativa (p = {p_valor:.6f})")

    medias = df.groupby(factor)[respuesta].agg(['mean', 'std', 'count']).reset_index()
    medias.columns = [factor, 'Media', 'Desv', 'n']
    medias_ordenadas = medias.sort_values('Media', ascending=False)

    for _, row in medias_ordenadas.iterrows():
        conclusiones.append(f"  • {row[factor]}: Media = {row['Media']:.4f} ± {row['Desv']:.4f} (n={int(row['n'])})")

    mejor = medias_ordenadas.iloc[0]
    peor = medias_ordenadas.iloc[-1]
    conclusiones.append(f"🏆 **Mejor:** {mejor[factor]} (Media = {mejor['Media']:.4f})")
    conclusiones.append(f"📉 **Peor:** {peor[factor]} (Media = {peor['Media']:.4f})")

    cv = (df[respuesta].std() / df[respuesta].mean()) * 100
    conclusiones.append(f"📈 **CV:** {cv:.2f}%")

    conclusiones.append(f"📐 **R² = {modelo.rsquared:.6f}**")
    return conclusiones, medias_ordenadas


def generar_conclusiones_factorial(anova_table, modelo, df, factores, respuesta, alpha=0.05):
    conclusiones = []
    for i, row in anova_table.iterrows():
        p_valor = None
        for col in anova_table.columns:
            if 'P-Value' in col:
                p_valor = row[col]
                break
        if p_valor is None:
            continue
        if p_valor < alpha:
            conclusiones.append(f"✅ **{i}:** Efecto significativo (p = {p_valor:.6f})")
        else:
            conclusiones.append(f"❌ **{i}:** No significativo (p = {p_valor:.6f})")
    conclusiones.append(f"📐 **R² = {modelo.rsquared:.6f}**")
    return conclusiones


# ============================================================
# CARGA DE DATOS
# ============================================================

def detectar_separador(texto):
    lineas = [l.strip() for l in texto.split('\n') if l.strip()]
    if not lineas:
        return ','
    separadores = [',', ';', '\t', '|', ' ']
    conteos = {sep: sum(l.count(sep) for l in lineas) for sep in separadores}
    mejor_sep = max(conteos, key=conteos.get)
    return mejor_sep if conteos[mejor_sep] > 0 else ','


def limpiar_nombres_columnas(df):
    df.columns = [c.strip().replace(' ', '_').replace(';', '').replace(',', '') for c in df.columns]
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
        df = limpiar_nombres_columnas(df)
        df = convertir_tipos(df)
        return df
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def cargar_datos_desde_url(url, separador=None):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
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
        df = limpiar_nombres_columnas(df)
        df = convertir_tipos(df)
        return df
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def procesar_datos_texto(texto, n_columnas=None):
    try:
        separador = detectar_separador(texto)
        lineas = [l.strip() for l in texto.split('\n') if l.strip()]
        primera_linea = lineas[0]
        valores = primera_linea.split(separador)
        tiene_encabezado = False
        try:
            for v in valores:
                float(v.strip())
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
                    datos.append([float(v) for v in valores])
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
        st.error(f"❌ Error: {str(e)}")
        return None


def obtener_datos_generales(key_suffix, n_columnas_esperadas=None):
    st.subheader("📥 Carga de Datos")
    tipo_carga = st.radio("Método:", ["✏️ Manual", "📁 CSV", "🔗 URL"],
                          horizontal=True, key=f"tipo_carga_{key_suffix}")
    df_resultado = None

    if tipo_carga == "✏️ Manual":
        if n_columnas_esperadas:
            st.caption(f"Formato: {n_columnas_esperadas} columnas")
        datos_texto = st.text_area("Datos:", value="", height=200,
                                    key=f"text_area_{key_suffix}",
                                    placeholder="A,23\nA,25\nB,30")
        if datos_texto and st.button("📊 Procesar", key=f"procesar_text_{key_suffix}"):
            df_resultado = procesar_datos_texto(datos_texto, n_columnas_esperadas)
            if df_resultado is not None:
                st.success(f"✅ {len(df_resultado)} filas cargadas")
                st.dataframe(df_resultado.head())
                st.session_state.datos_cargados[key_suffix] = df_resultado

    elif tipo_carga == "📁 CSV":
        archivo = st.file_uploader("Archivo:", type=['csv', 'txt'], key=f"csv_upload_{key_suffix}")
        if archivo is not None:
            if st.button("📊 Cargar", key=f"cargar_csv_{key_suffix}"):
                df_resultado = cargar_datos_desde_csv(archivo)
                if df_resultado is not None:
                    st.success(f"✅ {len(df_resultado)} filas")
                    st.dataframe(df_resultado.head())
                    st.session_state.datos_cargados[key_suffix] = df_resultado

    elif tipo_carga == "🔗 URL":
        url = st.text_input("URL:", key=f"url_input_{key_suffix}")
        if url and st.button("📊 Cargar", key=f"cargar_url_{key_suffix}"):
            df_resultado = cargar_datos_desde_url(url)
            if df_resultado is not None:
                st.success(f"✅ {len(df_resultado)} filas")
                st.dataframe(df_resultado.head())
                st.session_state.datos_cargados[key_suffix] = df_resultado

    if key_suffix in st.session_state.datos_cargados:
        df_guardado = st.session_state.datos_cargados[key_suffix]
        st.info(f"📊 Datos previos: {len(df_guardado)} filas")
        return df_guardado

    return df_resultado


def mostrar_informacion_datos(df, key_suffix):
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filas", len(df))
        with col2:
            st.metric("Columnas", len(df.columns))


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

tab_interactivo, tab1, tab2, tab3, tab4, tab5, tab6, tab6_resultados, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "🎯 Diseño Interactivo", "📊 ANOVA Un Factor", "📊 Diseño en Bloques",
    "📐 Cuadrado Latino", "📐 Greco-Latino", "🎯 Factorial Dos Factores",
    "🎯 Factorial Tres Factores", "🔬 Pruebas F2³", "📊 Diseño 2^k",
    "📊 Fraccionado", "📈 MSR", "⚙️ Taguchi", "🎯 Optimización",
    "➕ Aditivos", "📦 Split-Plot"
])

# ============================================================
# TAB INTERACTIVO
# ============================================================
with tab_interactivo:
    st.header("🎯 Diseño Experimental Interactivo")
    col1, col2 = st.columns([2, 1])
    with col1:
        factores_def = mostrar_definicion_factores()
        respuestas_def = mostrar_definicion_respuestas()
        replicas, randomizar = mostrar_configuracion_diseno()
        if st.button("🚀 Generar Diseño"):
            df_diseno = crear_diseno_completo(factores_def, respuestas_def, replicas, randomizar)
            if df_diseno is not None:
                st.session_state.diseno_actual = {'df': df_diseno, 'factores': factores_def, 'respuestas': respuestas_def}
                st.success(f"✅ {len(df_diseno)} corridas generadas")
    with col2:
        if 'diseno_actual' in st.session_state and st.session_state.diseno_actual:
            df = st.session_state.diseno_actual['df']
            st.dataframe(df.head())
            if st.button("🗑️ Limpiar"):
                st.session_state.diseno_actual = None
                st.rerun()

# ============================================================
# PESTAÑA 1: ANOVA UN FACTOR (ACTUALIZADA CON VISUALIZACIONES)
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Completamente Aleatorizado")
        df_temp = obtener_datos_generales("dca", 2)
        if df_temp is not None and len(df_temp.columns) >= 2:
            st.subheader("🔧 Configuración del ANOVA")

            tipo_anova = selector_tipo_anova("dca")

            col_factor = st.selectbox("Selecciona la columna del factor (Tratamiento):", df_temp.columns.tolist(), key="factor_dca")
            col_respuesta = st.selectbox("Selecciona la columna de respuesta (Valor):", [col for col in df_temp.columns if col != col_factor], key="respuesta_dca")
            mostrar_informacion_datos(df_temp, "dca")
            if st.button("🔍 Calcular ANOVA", key="dca_btn"):
                df = df_temp.copy()
                df.columns = ['Tratamiento' if col == col_factor else 'Valor' if col == col_respuesta else col for col in df.columns]
                df = df[['Tratamiento', 'Valor']]
                df['Tratamiento'] = df['Tratamiento'].astype('category')
                formula = 'Valor ~ C(Tratamiento)'
                modelo, anova = analizar_anova_mejorada(df, formula, "ANOVA Un Factor", "dca", "dca", tipo_anova)
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    st.session_state.pruebas_multiples['dca'] = {'df': df, 'modelo': modelo, 'anova': anova}

                    # Gráfico de Sumas de Cuadrados
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc:
                        st.plotly_chart(fig_sc, use_container_width=True)

                    # Gráficos de residuos
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

            # --- Boxplot por Tratamiento ---
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento',
                         title="Distribución por Tratamiento")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

            # --- Gráfico de Medias por Tratamiento ---
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor',
                          title="Medias por Tratamiento",
                          color='Tratamiento', text_auto=True)
            fig2.update_layout(showlegend=False, height=300)
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

            # --- PRUEBAS MÚLTIPLES CON FORMULARIO ---
            st.subheader("🔬 Pruebas Múltiples")
            with st.form(key="form_pruebas_dca"):
                metodo = st.radio("Selecciona el método:",
                                  ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"],
                                  horizontal=True, key="metodo_dca_form")
                submitted = st.form_submit_button("Calcular Prueba")

                if submitted:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    if metodo == "Tukey HSD":
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None:
                            st.session_state.pruebas_multiples['dca_resultado'] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, 'Tratamiento', 'Valor', mse_c, gl_c)
                        if lsd_df is not None:
                            st.session_state.pruebas_multiples['dca_resultado'] = ('lsd', lsd_df, mse_c, gl_c)

            # Mostrar resultados de la prueba
            if 'dca_resultado' in st.session_state.pruebas_multiples:
                res = st.session_state.pruebas_multiples['dca_resultado']
                if res[0] == 'tukey':
                    tukey_df = pd.DataFrame(data=res[1].summary().data[1:], columns=res[1].summary().data[0])
                    st.dataframe(tukey_df)
                    for linea in interpretar_tukey(res[1]):
                        st.write(linea)
                else:
                    st.write(f"**MSE = {res[2]:.4f}, GL Error = {int(res[3])}**")
                    st.dataframe(res[1])
                    for linea in interpretar_lsd(res[1]):
                        st.write(linea)

            # --- PANEL DE CONCLUSIONES ---
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
                if 'dca' in st.session_state.resultados_anova:
                    del st.session_state.resultados_anova['dca']
                if 'dca_resultado' in st.session_state.pruebas_multiples:
                    del st.session_state.pruebas_multiples['dca_resultado']
                st.rerun()

# ============================================================
# PESTAÑA 2: DISEÑO EN BLOQUES (ACTUALIZADA CON GRÁFICOS)
# ============================================================
with tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño en Bloques")
        df_temp = obtener_datos_generales("dbca", 3)
        if df_temp is not None and len(df_temp.columns) >= 3:
            st.subheader("🔧 Configuración del ANOVA")

            tipo_anova = selector_tipo_anova("dbca")

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
                modelo, anova = analizar_anova_mejorada(df, formula, "ANOVA en Bloques", "dbca", "dbca", tipo_anova)
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    st.session_state.pruebas_multiples['dbca'] = {'df': df, 'modelo': modelo, 'anova': anova}

                    # Gráfico de Sumas de Cuadrados
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc:
                        st.plotly_chart(fig_sc, use_container_width=True)

                    # Gráficos de residuos
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

            # --- Boxplot por Tratamiento ---
            st.write("**Boxplot por Tratamiento:**")
            fig_box_trat = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento',
                                  title="Distribución por Tratamiento")
            fig_box_trat.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_box_trat, use_container_width=True)

            # --- Boxplot por Bloque ---
            st.write("**Boxplot por Bloque:**")
            fig_box_bloque = px.box(df, x='Bloque', y='Valor', color='Bloque',
                                    title="Distribución por Bloque")
            fig_box_bloque.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_box_bloque, use_container_width=True)

            # --- Boxplot facetado por Bloque ---
            st.write("**Distribución por Tratamiento y Bloque:**")
            fig_facet = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento', facet_col='Bloque')
            fig_facet.update_layout(height=350)
            st.plotly_chart(fig_facet, use_container_width=True)

            # --- Medias por Tratamiento ---
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor', title="Medias por Tratamiento",
                          color='Tratamiento', text_auto=True)
            fig2.update_layout(showlegend=False, height=300)
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

            # --- PRUEBAS MÚLTIPLES CON FORMULARIO ---
            st.subheader("🔬 Pruebas Múltiples para Tratamientos")
            with st.form(key="form_pruebas_dbca"):
                metodo = st.radio("Selecciona el método:", ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"], horizontal=True, key="metodo_dbca_form")
                submitted = st.form_submit_button("Calcular Prueba")

                if submitted:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    if metodo == "Tukey HSD":
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None:
                            st.session_state.pruebas_multiples['dbca_resultado'] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, 'Tratamiento', 'Valor', mse_c, gl_c)
                        if lsd_df is not None:
                            st.session_state.pruebas_multiples['dbca_resultado'] = ('lsd', lsd_df, mse_c, gl_c)

            if 'dbca_resultado' in st.session_state.pruebas_multiples:
                res = st.session_state.pruebas_multiples['dbca_resultado']
                if res[0] == 'tukey':
                    tukey_df = pd.DataFrame(data=res[1].summary().data[1:], columns=res[1].summary().data[0])
                    st.dataframe(tukey_df)
                    for linea in interpretar_tukey(res[1]):
                        st.write(linea)
                else:
                    st.write(f"**MSE = {res[2]:.4f}, GL Error = {int(res[3])}**")
                    st.dataframe(res[1])
                    for linea in interpretar_lsd(res[1]):
                        st.write(linea)

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
                if 'dbca' in st.session_state.resultados_anova:
                    del st.session_state.resultados_anova['dbca']
                if 'dbca_resultado' in st.session_state.pruebas_multiples:
                    del st.session_state.pruebas_multiples['dbca_resultado']
                st.rerun()

# ============================================================
# PESTAÑA 3: CUADRADO LATINO (ACTUALIZADA CON VISUALIZACIONES)
# ============================================================
with tab3:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Cuadrado Latino")
        df_temp = obtener_datos_generales("cl", 4)
        if df_temp is not None and len(df_temp.columns) >= 4:
            st.subheader("🔧 Configuración del ANOVA")

            tipo_anova = selector_tipo_anova("cl")

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
                modelo, anova = analizar_anova_mejorada(df, formula, "Cuadrado Latino", "cl", "cl", tipo_anova)
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    st.session_state.pruebas_multiples['cl'] = {'df': df, 'modelo': modelo, 'anova': anova}

                    # Gráfico de Sumas de Cuadrados
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc:
                        st.plotly_chart(fig_sc, use_container_width=True)

                    # Gráficos de residuos
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

            # --- Boxplot por Tratamiento ---
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento',
                         title="Distribución por Tratamiento")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

            # --- Gráfico de Medias por Tratamiento ---
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor',
                          title="Medias por Tratamiento",
                          color='Tratamiento', text_auto=True)
            fig2.update_layout(showlegend=False, height=300)
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

            # --- Heatmap Fila × Columna ---
            pivot = df.pivot(index='Fila', columns='Columna', values='Valor')
            fig3 = px.imshow(pivot, text_auto=True, aspect="auto",
                             color_continuous_scale='Blues',
                             title="Mapa de Calor: Fila × Columna")
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)

            # --- PRUEBAS MÚLTIPLES CON FORMULARIO ---
            st.subheader("🔬 Pruebas Múltiples para Tratamientos")
            with st.form(key="form_pruebas_cl"):
                metodo = st.radio("Selecciona el método:",
                                  ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"],
                                  horizontal=True, key="metodo_cl_form")
                submitted = st.form_submit_button("Calcular Prueba")

                if submitted:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    if metodo == "Tukey HSD":
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None:
                            st.session_state.pruebas_multiples['cl_resultado'] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, 'Tratamiento', 'Valor', mse_c, gl_c)
                        if lsd_df is not None:
                            st.session_state.pruebas_multiples['cl_resultado'] = ('lsd', lsd_df, mse_c, gl_c)

            # Mostrar resultados
            if 'cl_resultado' in st.session_state.pruebas_multiples:
                res = st.session_state.pruebas_multiples['cl_resultado']
                if res[0] == 'tukey':
                    tukey_df = pd.DataFrame(data=res[1].summary().data[1:], columns=res[1].summary().data[0])
                    st.dataframe(tukey_df)
                    for linea in interpretar_tukey(res[1]):
                        st.write(linea)
                else:
                    st.write(f"**MSE = {res[2]:.4f}, GL Error = {int(res[3])}**")
                    st.dataframe(res[1])
                    for linea in interpretar_lsd(res[1]):
                        st.write(linea)

            # --- PANEL DE CONCLUSIONES ---
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
                if 'cl' in st.session_state.resultados_anova:
                    del st.session_state.resultados_anova['cl']
                if 'cl_resultado' in st.session_state.pruebas_multiples:
                    del st.session_state.pruebas_multiples['cl_resultado']
                st.rerun()

# ============================================================
# PESTAÑA 4: CUADRADO GRECO-LATINO (ACTUALIZADA CON GRÁFICOS)
# ============================================================
with tab4:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Cuadrado Greco-Latino")
        df_temp = obtener_datos_generales("cgl", 5)
        if df_temp is not None and len(df_temp.columns) >= 5:
            st.subheader("🔧 Configuración del ANOVA")

            tipo_anova = selector_tipo_anova("cgl")

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
                modelo, anova = analizar_anova_mejorada(df, formula, "Greco-Latino", "cgl", "cgl", tipo_anova)
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    st.session_state.pruebas_multiples['cgl'] = {'df': df, 'modelo': modelo, 'anova': anova}

                    # Gráfico de Sumas de Cuadrados
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc:
                        st.plotly_chart(fig_sc, use_container_width=True)

                    # Gráficos de residuos
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

            # --- Boxplot por Tratamiento ---
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento')
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

            # --- Medias por Tratamiento ---
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor', title="Medias por Tratamiento",
                          color='Tratamiento', text_auto=True)
            fig2.update_layout(showlegend=False, height=300)
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

            # --- Heatmap Fila × Columna ---
            pivot = df.pivot(index='Fila', columns='Columna', values='Valor')
            fig3 = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='Greens',
                             title="Mapa de Calor: Fila × Columna")
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)

            # --- Heatmap Tratamiento × Greco ---
            try:
                pivot_tg = df.pivot_table(index='Tratamiento', columns='Greco', values='Valor', aggfunc='mean')
                fig4 = px.imshow(pivot_tg, text_auto=True, aspect="auto", color_continuous_scale='Blues',
                                 title="Mapa de Calor: Tratamiento × Greco")
                fig4.update_layout(height=350)
                st.plotly_chart(fig4, use_container_width=True)
            except:
                pass

            # --- PRUEBAS MÚLTIPLES: TRATAMIENTO ---
            st.subheader("🔬 Pruebas Múltiples - Tratamiento")
            with st.form(key="form_pruebas_cgl_trat"):
                metodo_trat = st.radio("Selecciona el método:", ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"], horizontal=True, key="metodo_cgl_trat_form")
                submitted_trat = st.form_submit_button("Calcular Prueba para Tratamiento")

                if submitted_trat:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    if metodo_trat == "Tukey HSD":
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None:
                            st.session_state.pruebas_multiples['cgl_trat_resultado'] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, 'Tratamiento', 'Valor', mse_c, gl_c)
                        if lsd_df is not None:
                            st.session_state.pruebas_multiples['cgl_trat_resultado'] = ('lsd', lsd_df, mse_c, gl_c)

            if 'cgl_trat_resultado' in st.session_state.pruebas_multiples:
                res = st.session_state.pruebas_multiples['cgl_trat_resultado']
                if res[0] == 'tukey':
                    tukey_df = pd.DataFrame(data=res[1].summary().data[1:], columns=res[1].summary().data[0])
                    st.dataframe(tukey_df)
                    for linea in interpretar_tukey(res[1]):
                        st.write(linea)
                else:
                    st.write(f"**MSE = {res[2]:.4f}, GL Error = {int(res[3])}**")
                    st.dataframe(res[1])
                    for linea in interpretar_lsd(res[1]):
                        st.write(linea)

            # --- PRUEBAS MÚLTIPLES: GRECO ---
            st.subheader("🔬 Pruebas Múltiples - Greco")
            with st.form(key="form_pruebas_cgl_greco"):
                metodo_greco = st.radio("Selecciona el método:", ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"], horizontal=True, key="metodo_cgl_greco_form")
                submitted_greco = st.form_submit_button("Calcular Prueba para Greco")

                if submitted_greco:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    if metodo_greco == "Tukey HSD":
                        tukey = realizar_tukey(df, 'Greco', 'Valor')
                        if tukey is not None:
                            st.session_state.pruebas_multiples['cgl_greco_resultado'] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, 'Greco', 'Valor', mse_c, gl_c)
                        if lsd_df is not None:
                            st.session_state.pruebas_multiples['cgl_greco_resultado'] = ('lsd', lsd_df, mse_c, gl_c)

            if 'cgl_greco_resultado' in st.session_state.pruebas_multiples:
                res = st.session_state.pruebas_multiples['cgl_greco_resultado']
                if res[0] == 'tukey':
                    tukey_df = pd.DataFrame(data=res[1].summary().data[1:], columns=res[1].summary().data[0])
                    st.dataframe(tukey_df)
                    for linea in interpretar_tukey(res[1]):
                        st.write(linea)
                else:
                    st.write(f"**MSE = {res[2]:.4f}, GL Error = {int(res[3])}**")
                    st.dataframe(res[1])
                    for linea in interpretar_lsd(res[1]):
                        st.write(linea)

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
                for key in ['cgl', 'cgl_trat_resultado', 'cgl_greco_resultado']:
                    if key in st.session_state.resultados_anova:
                        del st.session_state.resultados_anova[key]
                    if key in st.session_state.pruebas_multiples:
                        del st.session_state.pruebas_multiples[key]
                st.rerun()

# ============================================================
# TAB 5: FACTORIAL 2²
# ============================================================
with tab5:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Factorial 2²")
        df_temp = obtener_datos_generales("f22", 3)
        if df_temp is not None and len(df_temp.columns) >= 3:
            tipo_anova = selector_tipo_anova("f22")
            col_a = st.selectbox("Factor A:", df_temp.columns.tolist(), key="a_f22")
            cr = [c for c in df_temp.columns if c != col_a]
            col_b = st.selectbox("Factor B:", cr, key="b_f22")
            cr = [c for c in cr if c != col_b]
            col_resp = st.selectbox("Respuesta:", cr, key="r_f22")
            if st.button("🔍 Calcular", key="f22_btn"):
                df = df_temp.copy()
                df.columns = ['A' if c == col_a else 'B' if c == col_b else 'Y' if c == col_resp else c for c in df.columns]
                df = df[['A', 'B', 'Y']]
                df['A'] = df['A'].astype('category')
                df['B'] = df['B'].astype('category')
                modelo, anova = analizar_anova_mejorada(df, 'Y ~ A * B', "F2²", "f22", "f22", tipo_anova)
                if modelo is not None:
                    st.session_state.pruebas_multiples['f22'] = {'df': df, 'modelo': modelo, 'anova': anova}
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    mostrar_graficos_residuos_directo(modelo, df, 'A', 'Y', "f22", "Residuos")
    with col2:
        if 'f22' in st.session_state.resultados_anova:
            res = st.session_state.resultados_anova['f22']
            df, anova = res['df'], res['anova']
            st.write("**Boxplot por Factor:**")
            fig = graficar_boxplot_por_factor(df, ['A', 'B'], 'Y')
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.write("**Interacción:**")
            fig_i = graficar_interaccion_mejorada(df, 'A', 'B', 'Y')
            if fig_i: st.plotly_chart(fig_i, use_container_width=True)
            
            st.subheader("🔬 Pruebas Múltiples")
            with st.form(key="form_f22"):
                metodo = st.radio("Método:", ["Tukey HSD", "LSD"], horizontal=True, key="metodo_f22")
                factores_prueba = st.multiselect("Factores:", ['A', 'B'], default=['A', 'B'], key="fact_f22")
                submitted = st.form_submit_button("Calcular")
                if submitted:
                    mse, gl = extraer_mse_gl(anova)
                    resultados = {}
                    for factor in factores_prueba:
                        if metodo == "Tukey HSD" and len(df[factor].unique()) > 2:
                            tukey = realizar_tukey(df, factor, 'Y')
                            if tukey is not None:
                                resultados[factor] = ('tukey', tukey)
                        elif metodo == "LSD":
                            lsd_df = realizar_lsd(df, factor, 'Y', mse, gl)
                            if lsd_df is not None:
                                resultados[factor] = ('lsd', lsd_df, mse, gl)
                    st.session_state.pruebas_multiples['f22_resultado'] = resultados
            if 'f22_resultado' in st.session_state.pruebas_multiples:
                for factor, r in st.session_state.pruebas_multiples['f22_resultado'].items():
                    st.markdown(f"**Factor {factor}:**")
                    if r[0] == 'tukey':
                        st.dataframe(pd.DataFrame(data=r[1].summary().data[1:], columns=r[1].summary().data[0]))
                        for l in interpretar_tukey(r[1]): st.write(l)
                    else:
                        st.write(f"MSE = {r[2]:.4f}, GL = {int(r[3])}")
                        st.dataframe(r[1])
                        for l in interpretar_lsd(r[1]): st.write(l)

# ============================================================
# PESTAÑA 6: FACTORIAL 2³ (ANOVA + GRÁFICOS + CONCLUSIONES)
# ============================================================
with tab6:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Factorial 2³")
        df_temp = obtener_datos_generales("f23", 4)
        if df_temp is not None and len(df_temp.columns) >= 4:
            st.subheader("🔧 Configuración del ANOVA")

            tipo_anova = selector_tipo_anova("f23")

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
                modelo, anova = analizar_anova_mejorada(df, formula, "Factorial 2³", "f23", "f23", tipo_anova)
                if modelo is not None:
                    st.success("✅ ANOVA completado exitosamente")
                    st.session_state.pruebas_multiples['f23'] = {'df': df, 'modelo': modelo, 'anova': anova}

                    # Gráfico de Sumas de Cuadrados
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc:
                        st.plotly_chart(fig_sc, use_container_width=True)

                    # ============================================================
                    # GRÁFICOS DE INTERACCIÓN
                    # ============================================================
                    st.subheader("📊 Gráficos de Interacción")
                    factores_list = ['A', 'B', 'C']
                    num_pares = len(list(combinations(factores_list, 2)))
                    max_pares_mostrar = st.slider(
                        "Número de pares de interacción a mostrar:",
                        min_value=1,
                        max_value=min(num_pares, 8),
                        value=min(3, num_pares),
                        key="num_pares_interaccion_f23"
                    )
                    fig_interaccion = graficos_interaccion_multifactorial(
                        df, factores_list, 'Y', max_pares_mostrar
                    )
                    if fig_interaccion is not None:
                        st.plotly_chart(fig_interaccion, use_container_width=True)

                    st.subheader("📊 Mapa de Calor de Interacciones")
                    fig_heatmap = graficos_interaccion_heatmap(df, factores_list, 'Y')
                    if fig_heatmap is not None:
                        st.plotly_chart(fig_heatmap, use_container_width=True)

                    # Gráficos de residuos
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

            # --- Boxplot por Factor ---
            st.write("**Boxplot por Factor:**")
            fig_box = graficar_boxplot_por_factor(df, ['A', 'B', 'C'], 'Y')
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True)

            # --- Gráfico de Interacción Mejorado (A x B) ---
            st.write("**Gráfico de Interacción Mejorado (A x B):**")
            fig_inter = graficar_interaccion_mejorada(df, 'A', 'B', 'Y')
            if fig_inter:
                st.plotly_chart(fig_inter, use_container_width=True)

            # --- Boxplot por Combinación ---
            df['Combinacion'] = df['A'].astype(str) + ',' + df['B'].astype(str) + ',' + df['C'].astype(str)
            fig1 = px.box(df, x='Combinacion', y='Y', color='Combinacion',
                          title="Distribución por Combinación de Factores")
            fig1.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig1, use_container_width=True)

            # --- Gráfico de Barras de Medias ---
            medias = df.groupby(['A', 'B', 'C'])['Y'].mean().reset_index()
            fig3 = px.bar(medias, x='A', y='Y', color='B', facet_col='C',
                          title="Medias por Combinación de Factores",
                          barmode='group', text_auto=True)
            fig3.update_layout(height=350)
            fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)

            # ============================================================
            # CONCLUSIONES DEL ANÁLISIS (panel de colores)
            # ============================================================
            st.subheader("📝 Conclusiones del Análisis")
            conclusiones = generar_conclusiones_factorial(anova, modelo, df, ['A', 'B', 'C'], 'Y')
            for conclusion in conclusiones:
                if conclusion.startswith("✅"):
                    st.success(conclusion)
                elif conclusion.startswith("❌"):
                    st.error(conclusion)
                else:
                    st.info(conclusion)

            if st.button("🗑️ Limpiar resultados", key="limpiar_f23"):
                for key in ['f23', 'f23_resultado_tab']:
                    if key in st.session_state.resultados_anova:
                        del st.session_state.resultados_anova[key]
                    if key in st.session_state.pruebas_multiples:
                        del st.session_state.pruebas_multiples[key]
                st.rerun()

# ============================================================
# TAB 6_RESULTADOS: PRUEBAS MÚLTIPLES F2³ (CORREGIDO)
# ============================================================
with tab6_resultados:
    st.header("🔬 Pruebas Múltiples - Factorial 2³")
    st.markdown("Esta pestaña muestra los resultados de Tukey HSD y LSD para el Factorial 2³.")

    if 'f23' in st.session_state.resultados_anova:
        res = st.session_state.resultados_anova['f23']
        df, anova = res['df'], res['anova']
        mse_c, gl_c = extraer_mse_gl(anova)

        st.info(f"📊 **MSE del modelo completo:** {mse_c:.4f} | **GL Error:** {int(gl_c)}")

        st.markdown("---")
        st.subheader("⚙️ Configuración")

        metodo = st.radio(
            "Método:",
            ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"],
            horizontal=True,
            key="met_f23_final"
        )

        st.write("**Factores para probar:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            usar_a = st.checkbox("Factor A", value=True, key="chk_a_f23_final")
        with col2:
            usar_b = st.checkbox("Factor B", value=True, key="chk_b_f23_final")
        with col3:
            usar_c = st.checkbox("Factor C", value=True, key="chk_c_f23_final")

        factores_prueba = []
        if usar_a: factores_prueba.append('A')
        if usar_b: factores_prueba.append('B')
        if usar_c: factores_prueba.append('C')

        if st.button("Calcular Pruebas Múltiples", key="btn_calc_f23_final"):
            if not factores_prueba:
                st.warning("⚠️ Selecciona al menos un factor.")
            else:
                resultados_multiples = {}
                for factor in factores_prueba:
                    n_niveles = len(df[factor].unique())
                    if metodo == "Tukey HSD":
                        if n_niveles > 2:
                            tukey = realizar_tukey(df, factor, 'Y')
                            if tukey is not None:
                                resultados_multiples[factor] = ('tukey', tukey)
                        else:
                            st.info(f"ℹ️ **{factor}:** Tukey no aplica (solo {n_niveles} niveles). Usa LSD.")
                    else:
                        # LSD SIEMPRE se ejecuta, incluso con 2 niveles
                        lsd_df = realizar_lsd(df, factor, 'Y', mse_c, gl_c)
                        if lsd_df is not None:
                            resultados_multiples[factor] = ('lsd', lsd_df, mse_c, gl_c)
                st.session_state.pruebas_multiples['f23_resultado_tab'] = resultados_multiples

        if 'f23_resultado_tab' in st.session_state.pruebas_multiples:
            res_dict = st.session_state.pruebas_multiples['f23_resultado_tab']
            if not res_dict:
                st.warning("⚠️ No se generaron resultados.")
            else:
                for factor, r in res_dict.items():
                    st.markdown(f"#### 📌 Resultados para Factor {factor}")
                    if r[0] == 'tukey':
                        st.dataframe(pd.DataFrame(data=r[1].summary().data[1:], columns=r[1].summary().data[0]))
                        for l in interpretar_tukey(r[1]): st.write(l)
                    else:
                        st.write(f"**MSE = {r[2]:.4f}, GL Error = {int(r[3])}**")
                        st.dataframe(r[1])
                        for l in interpretar_lsd(r[1]): st.write(l)
    else:
        st.warning("⚠️ Primero calcula el ANOVA en la pestaña 'Factorial Tres Factores'.")

# ============================================================
# TAB 7: DISEÑO 2^k (simplificado)
# ============================================================
with tab7:
    st.subheader("📌 Diseño 2^k")
    k = st.number_input("Factores (k):", min_value=2, max_value=8, value=3, step=1, key="k_2k")
    st.caption(f"Se esperan {k+1} columnas.")
    df_temp = obtener_datos_generales("d2k", k+1)
    if df_temp is not None and len(df_temp.columns) >= k+1:
        tipo_anova = selector_tipo_anova("d2k")
        factores = []
        cols = df_temp.columns.tolist()
        for i in range(k):
            f = st.selectbox(f"Factor {i+1}:", cols if i == 0 else [c for c in cols if c not in factores], key=f"f_{i}_2k")
            factores.append(f)
        col_r = st.selectbox("Respuesta:", [c for c in cols if c not in factores], key="r_2k")
        if st.button("🔍 Calcular", key="2k_btn"):
            df = df_temp.copy()
            nombres = [f'F{i+1}' for i in range(k)] + ['Y']
            mapeo = {factores[i]: f'F{i+1}' for i in range(k)}
            mapeo[col_r] = 'Y'
            for o, n in mapeo.items(): df.rename(columns={o: n}, inplace=True)
            df = df[nombres]
            for i in range(k): df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
            factores_list = [f'F{i+1}' for i in range(k)]
            formula = 'Y ~ ' + ' * '.join(factores_list)
            modelo, anova = analizar_anova_mejorada(df, formula, f"2^{k}", "d2k", "d2k", tipo_anova)
            if modelo is not None:
                st.session_state.pruebas_multiples['d2k'] = {'df': df, 'modelo': modelo, 'anova': anova}
                fig_sc = graficar_sumas_cuadrados(anova)
                if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "d2k", "Residuos")

# ============================================================
# TAB 8: FRACCIONADO
# ============================================================
with tab8:
    st.subheader("📌 Fraccionado 2^(k-p)")
    k = st.number_input("k:", min_value=3, max_value=8, value=4, step=1, key="k_fr")
    p = st.number_input("p:", min_value=1, max_value=min(3, k-1), value=1, step=1, key="p_fr")
    st.caption(f"Se esperan {k+1} columnas.")
    df_temp = obtener_datos_generales("fr", k+1)
    if df_temp is not None and len(df_temp.columns) >= k+1:
        tipo_anova = selector_tipo_anova("fr")
        factores = []
        cols = df_temp.columns.tolist()
        for i in range(k):
            f = st.selectbox(f"Factor {i+1}:", cols if i == 0 else [c for c in cols if c not in factores], key=f"f_{i}_fr")
            factores.append(f)
        col_r = st.selectbox("Respuesta:", [c for c in cols if c not in factores], key="r_fr")
        if st.button("🔍 Calcular", key="fr_btn"):
            df = df_temp.copy()
            nombres = [f'F{i+1}' for i in range(k)] + ['Y']
            mapeo = {factores[i]: f'F{i+1}' for i in range(k)}
            mapeo[col_r] = 'Y'
            for o, n in mapeo.items(): df.rename(columns={o: n}, inplace=True)
            df = df[nombres]
            for i in range(k): df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
            factores_list = [f'F{i+1}' for i in range(k)]
            formula = 'Y ~ ' + ' * '.join(factores_list)
            modelo, anova = analizar_anova_mejorada(df, formula, f"2^({k}-{p})", "fr", "fr", tipo_anova)
            if modelo is not None:
                st.session_state.pruebas_multiples['fr'] = {'df': df, 'modelo': modelo, 'anova': anova}
                fig_sc = graficar_sumas_cuadrados(anova)
                if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "fr", "Residuos")

# ============================================================
# TAB 9: MSR
# ============================================================
with tab9:
    st.subheader("📌 MSR")
    n_f = st.number_input("Factores:", min_value=2, max_value=4, value=2, step=1, key="nf_msr")
    df_temp = obtener_datos_generales("msr", n_f+1)
    if df_temp is not None and len(df_temp.columns) >= n_f+1:
        tipo_anova = selector_tipo_anova("msr")
        factores = []
        cols = df_temp.columns.tolist()
        for i in range(n_f):
            f = st.selectbox(f"Factor {i+1}:", cols if i == 0 else [c for c in cols if c not in factores], key=f"f_{i}_msr")
            factores.append(f)
        col_r = st.selectbox("Respuesta:", [c for c in cols if c not in factores], key="r_msr")
        if st.button("🔍 Calcular", key="msr_btn"):
            df = df_temp.copy()
            nombres = [f'F{i+1}' for i in range(n_f)] + ['Y']
            mapeo = {factores[i]: f'F{i+1}' for i in range(n_f)}
            mapeo[col_r] = 'Y'
            for o, n in mapeo.items(): df.rename(columns={o: n}, inplace=True)
            df = df[nombres]
            vars_list = [f'F{i+1}' for i in range(n_f)]
            formula = 'Y ~ ' + ' + '.join(vars_list) + ' + ' + ' + '.join([f'I({v}**2)' for v in vars_list])
            if n_f >= 2:
                formula += ' + ' + ' + '.join([f'F{i+1}:F{j+1}' for i in range(n_f) for j in range(i+1, n_f)])
            modelo, anova = analizar_anova_mejorada(df, formula, "MSR", "msr", "msr", tipo_anova)
            if modelo is not None:
                mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "msr", "Residuos")

# ============================================================
# TAB 10: TAGUCHI
# ============================================================
with tab10:
    st.subheader("📌 Taguchi")
    arreglo = st.selectbox("Arreglo:", ["L4 (2³)", "L8 (2⁷)", "L9 (3⁴)", "L16 (2¹⁵)", "L27 (3¹³)"], index=1, key="tag")
    n_f = {"L4 (2³)": 3, "L8 (2⁷)": 7, "L9 (3⁴)": 4, "L16 (2¹⁵)": 15, "L27 (3¹³)": 13}[arreglo]
    df_temp = obtener_datos_generales("tag", n_f+1)
    if df_temp is not None and len(df_temp.columns) >= n_f+1:
        tipo_anova = selector_tipo_anova("tag")
        factores = []
        cols = df_temp.columns.tolist()
        for i in range(min(n_f, 8)):
            f = st.selectbox(f"Factor {i+1}:", cols if i == 0 else [c for c in cols if c not in factores], key=f"f_{i}_tag")
            factores.append(f)
        col_r = st.selectbox("Respuesta:", [c for c in cols if c not in factores], key="r_tag")
        if st.button("🔍 Calcular", key="tag_btn"):
            df = df_temp.copy()
            nombres = [f'F{i+1}' for i in range(n_f)] + ['Y']
            mapeo = {factores[i]: f'F{i+1}' for i in range(min(n_f, 8))}
            restantes = [c for c in cols if c not in factores and c != col_r]
            for i in range(8, n_f):
                if restantes: mapeo[restantes.pop(0)] = f'F{i+1}'
            mapeo[col_r] = 'Y'
            for o, n in mapeo.items(): df.rename(columns={o: n}, inplace=True)
            df = df[nombres]
            for i in range(n_f): df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
            factores_list = [f'F{i+1}' for i in range(n_f)]
            formula = 'Y ~ ' + ' + '.join(factores_list)
            modelo, anova = analizar_anova_mejorada(df, formula, "Taguchi", "tag", "tag", tipo_anova)
            if modelo is not None:
                mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "tag", "Residuos")

# ============================================================
# TAB 11: OPTIMIZACIÓN
# ============================================================
with tab11:
    st.subheader("📌 Optimización Múltiple")
    df_temp = obtener_datos_generales("omr", 4)
    if df_temp is not None and len(df_temp.columns) >= 4:
        col_f1 = st.selectbox("Factor 1:", df_temp.columns.tolist(), key="f1_omr")
        cr = [c for c in df_temp.columns if c != col_f1]
        col_f2 = st.selectbox("Factor 2:", cr, key="f2_omr")
        cr = [c for c in cr if c != col_f2]
        col_y1 = st.selectbox("Respuesta 1 (max):", cr, key="y1_omr")
        cr = [c for c in cr if c != col_y1]
        col_y2 = st.selectbox("Respuesta 2 (min):", cr, key="y2_omr")
        if st.button("🔍 Optimizar", key="omr_btn"):
            try:
                df = df_temp.copy()
                df.columns = ['F1' if c == col_f1 else 'F2' if c == col_f2 else 'Y1' if c == col_y1 else 'Y2' if c == col_y2 else c for c in df.columns]
                df = df[['F1', 'F2', 'Y1', 'Y2']]
                m1 = ols('Y1 ~ F1 + F2 + I(F1**2) + I(F2**2) + F1:F2', data=df).fit()
                m2 = ols('Y2 ~ F1 + F2 + I(F1**2) + I(F2**2) + F1:F2', data=df).fit()
                grid = pd.DataFrame([(f1, f2) for f1 in np.linspace(-1.5, 1.5, 20) for f2 in np.linspace(-1.5, 1.5, 20)], columns=['F1', 'F2'])
                grid['Y1'] = m1.predict(grid)
                grid['Y2'] = m2.predict(grid)
                grid['D'] = (((grid['Y1'] - grid['Y1'].min()) / (grid['Y1'].max() - grid['Y1'].min())) * (1 - (grid['Y2'] - grid['Y2'].min()) / (grid['Y2'].max() - grid['Y2'].min()))) ** 0.5
                opt = grid.loc[grid['D'].idxmax()]
                st.success(f"Óptimo: F1={opt['F1']:.3f}, F2={opt['F2']:.3f}, D={opt['D']:.4f}")
            except Exception as e:
                st.error(f"❌ {e}")

# ============================================================
# TAB 12: ADITIVOS
# ============================================================
with tab12:
    st.subheader("📌 Diseños Añadidos")
    df_temp = obtener_datos_generales("ad", 4)
    if df_temp is not None and len(df_temp.columns) >= 4:
        tipo_anova = selector_tipo_anova("ad")
        c1 = st.selectbox("Comp 1:", df_temp.columns.tolist(), key="c1_ad")
        cr = [c for c in df_temp.columns if c != c1]
        c2 = st.selectbox("Comp 2:", cr, key="c2_ad")
        cr = [c for c in cr if c != c2]
        c3 = st.selectbox("Comp 3:", cr, key="c3_ad")
        cr = [c for c in cr if c != c3]
        col_r = st.selectbox("Respuesta:", cr, key="r_ad")
        if st.button("🔍 Calcular", key="ad_btn"):
            df = df_temp.copy()
            df.columns = ['C1' if c == c1 else 'C2' if c == c2 else 'C3' if c == c3 else 'Y' if c == col_r else c for c in df.columns]
            df = df[['C1', 'C2', 'C3', 'Y']]
            modelo, anova = analizar_anova_mejorada(df, 'Y ~ C1 + C2 + C3', "Aditivos", "ad", "ad", tipo_anova)
            if modelo is not None:
                mostrar_graficos_residuos_directo(modelo, df, 'C1', 'Y', "ad", "Residuos")

# ============================================================
# TAB 13: SPLIT-PLOT
# ============================================================
with tab13:
    st.subheader("📌 Split-Plot")
    df_temp = obtener_datos_generales("sp", 4)
    if df_temp is not None and len(df_temp.columns) >= 4:
        tipo_anova = selector_tipo_anova("sp")
        f1 = st.selectbox("Factor 1:", df_temp.columns.tolist(), key="f1_sp")
        cr = [c for c in df_temp.columns if c != f1]
        f2 = st.selectbox("Factor 2:", cr, key="f2_sp")
        cr = [c for c in cr if c != f2]
        f3 = st.selectbox("Factor 3:", cr, key="f3_sp")
        cr = [c for c in cr if c != f3]
        col_r = st.selectbox("Respuesta:", cr, key="r_sp")
        if st.button("🔍 Calcular", key="sp_btn"):
            df = df_temp.copy()
            df.columns = ['F1' if c == f1 else 'F2' if c == f2 else 'F3' if c == f3 else 'Y' if c == col_r else c for c in df.columns]
            df = df[['F1', 'F2', 'F3', 'Y']]
            for c in ['F1', 'F2', 'F3']: df[c] = df[c].astype('category')
            modelo, anova = analizar_anova_mejorada(df, 'Y ~ F1 * F2 * F3', "Split-Plot", "sp", "sp", tipo_anova)
            if modelo is not None:
                fig_sc = graficar_sumas_cuadrados(anova)
                if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "sp", "Residuos")

print("✅ app.py creado exitosamente!")
