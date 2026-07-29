import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.stats import ttest_ind, ttest_rel, f_oneway, chi2_contingency, pearsonr, spearmanr, mannwhitneyu, wilcoxon, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
import io
import csv

# ============================================================
# FUNCIONES PARA LECTURA ROBUSTA DE CSV
# ============================================================

def detectar_separador(archivo):
    """Detecta automáticamente el separador del archivo CSV"""
    try:
        contenido = archivo.getvalue().decode('utf-8')
        lineas = contenido.split('\n')[:5]

        separadores = [',', ';', '\t', '|']
        for sep in separadores:
            for linea in lineas:
                if sep in linea and len(linea.split(sep)) > 1:
                    return sep
        return ','
    except:
        return ','

def leer_csv_robusto(archivo):
    """Lee un archivo CSV con múltiples estrategias"""
    try:
        contenido = archivo.getvalue()
        separador = detectar_separador(archivo)

        try:
            df = pd.read_csv(io.BytesIO(contenido), sep=separador, encoding='utf-8')
            if len(df.columns) > 1 and df.shape[0] > 0:
                return df
        except:
            pass

        try:
            df = pd.read_csv(io.BytesIO(contenido), sep=separador, encoding='utf-8',
                            engine='python', on_bad_lines='skip')
            if len(df.columns) > 1 and df.shape[0] > 0:
                return df
        except:
            pass

        try:
            texto = contenido.decode('utf-8')
            lineas = [line.strip() for line in texto.split('\n') if line.strip()]
            sep = detectar_separador(io.BytesIO(contenido))

            datos = []
            for linea in lineas:
                if sep in linea:
                    fila = linea.split(sep)
                    datos.append([x.strip() for x in fila])

            if datos:
                cabecera = datos[0]
                datos_fila = datos[1:]
                max_cols = max([len(fila) for fila in datos_fila]) if datos_fila else len(cabecera)

                datos_fila_limpios = []
                for fila in datos_fila:
                    while len(fila) < max_cols:
                        fila.append('')
                    datos_fila_limpios.append(fila[:max_cols])

                df = pd.DataFrame(datos_fila_limpios, columns=cabecera[:max_cols])
                return df
        except:
            pass

        try:
            df = pd.read_csv(io.BytesIO(contenido), sep=None, engine='python',
                            on_bad_lines='skip', encoding='utf-8')
            if len(df.columns) > 1 and df.shape[0] > 0:
                return df
        except:
            pass

        return None

    except Exception as e:
        return None

# ============================================================
# FUNCIONES DE REGRESIÓN AVANZADA
# ============================================================

def regresion_curve_fit(x, y):
    """Regresión lineal usando optimize.curve_fit"""
    try:
        def modelo_lineal(x, a, b):
            return a + b * x

        popt, pcov = optimize.curve_fit(modelo_lineal, x, y)
        a, b = popt

        y_pred = modelo_lineal(x, a, b)
        residuos = y - y_pred
        ss_res = np.sum(residuos**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot)
        mse = np.mean(residuos**2)

        return {
            'metodo': 'curve_fit',
            'intercepto': a,
            'pendiente': b,
            'r2': r2,
            'mse': mse,
            'y_pred': y_pred,
            'residuos': residuos,
            'a_std': np.sqrt(pcov[0, 0]),
            'b_std': np.sqrt(pcov[1, 1])
        }
    except Exception as e:
        return None

def regresion_ols(x, y):
    """Regresión lineal usando Mínimos Cuadrados (statsmodels)"""
    try:
        X = sm.add_constant(x)
        modelo = sm.OLS(y, X).fit()

        b = modelo.params.iloc[1]
        a = modelo.params.iloc[0]

        r2 = modelo.rsquared
        mse = modelo.mse_resid

        y_pred = modelo.predict(X)
        residuos = modelo.resid

        return {
            'metodo': 'OLS',
            'intercepto': a,
            'pendiente': b,
            'r2': r2,
            'mse': mse,
            'y_pred': y_pred,
            'residuos': residuos,
            'p_value_intercepto': modelo.pvalues.iloc[0],
            'p_value_pendiente': modelo.pvalues.iloc[1]
        }
    except Exception as e:
        return None

def regresion_sklearn(x, y):
    """Regresión lineal usando Scikit-learn"""
    try:
        X = x.values.reshape(-1, 1) if isinstance(x, pd.Series) else x.reshape(-1, 1)
        y_values = y.values if isinstance(y, pd.Series) else y

        model = LinearRegression()
        model.fit(X, y_values)

        b = model.coef_[0]
        a = model.intercept_

        y_pred = model.predict(X)
        residuos = y_values - y_pred

        r2 = r2_score(y_values, y_pred)
        mse = mean_squared_error(y_values, y_pred)

        return {
            'metodo': 'Scikit-learn',
            'intercepto': a,
            'pendiente': b,
            'r2': r2,
            'mse': mse,
            'y_pred': y_pred,
            'residuos': residuos,
            'modelo': model
        }
    except Exception as e:
        return None

def regresion_bootstrap(x, y, n_bootstrap=1000, ci=0.95):
    """Regresión lineal con Bootstrap"""
    try:
        def modelo_lineal(x, a, b):
            return a + b * x

        n = len(x)
        bootstrap_a = []
        bootstrap_b = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            bootstrap_x = x.iloc[indices] if isinstance(x, pd.Series) else x[indices]
            bootstrap_y = y.iloc[indices] if isinstance(y, pd.Series) else y[indices]

            try:
                popt, _ = optimize.curve_fit(modelo_lineal, bootstrap_x, bootstrap_y)
                bootstrap_a.append(popt[0])
                bootstrap_b.append(popt[1])
            except:
                pass

        mean_a = np.mean(bootstrap_a)
        mean_b = np.mean(bootstrap_b)

        lower_a = np.percentile(bootstrap_a, (1-ci)/2 * 100)
        upper_a = np.percentile(bootstrap_a, (1+ci)/2 * 100)
        lower_b = np.percentile(bootstrap_b, (1-ci)/2 * 100)
        upper_b = np.percentile(bootstrap_b, (1+ci)/2 * 100)

        y_pred = modelo_lineal(x, mean_a, mean_b)
        residuos = y - y_pred
        ss_res = np.sum(residuos**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot)
        mse = np.mean(residuos**2)

        return {
            'metodo': 'Bootstrap',
            'intercepto': mean_a,
            'pendiente': mean_b,
            'intercepto_ci': (lower_a, upper_a),
            'pendiente_ci': (lower_b, upper_b),
            'r2': r2,
            'mse': mse,
            'y_pred': y_pred,
            'residuos': residuos,
            'bootstrap_a': bootstrap_a,
            'bootstrap_b': bootstrap_b,
            'n_bootstrap': n_bootstrap,
            'ci': ci
        }
    except Exception as e:
        return None

def graficar_comparacion_regresiones(x, y, resultados):
    """Genera gráfico de comparación de los 3 métodos de regresión"""
    try:
        fig = make_subplots(rows=1, cols=3,
                           subplot_titles=['Optimize.curve_fit', 'Minimos Cuadrados (OLS)', 'Scikit-learn'])

        scatter = go.Scatter(x=x, y=y, mode='markers', name='Datos',
                            marker=dict(color='blue', opacity=0.6))

        x_plot = np.linspace(x.min(), x.max(), 100)

        if 'curve_fit' in resultados and resultados['curve_fit']:
            res = resultados['curve_fit']
            y_plot = res['intercepto'] + res['pendiente'] * x_plot
            fig.add_trace(scatter, row=1, col=1)
            fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines',
                                    name='curve_fit', line=dict(color='red', width=2)),
                         row=1, col=1)
            fig.add_annotation(x=0.05, y=0.95, xref="x domain", yref="y domain",
                              text=f"R² = {res['r2']:.4f}", showarrow=False,
                              row=1, col=1)

        if 'ols' in resultados and resultados['ols']:
            res = resultados['ols']
            y_plot = res['intercepto'] + res['pendiente'] * x_plot
            fig.add_trace(scatter, row=1, col=2)
            fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines',
                                    name='OLS', line=dict(color='green', width=2)),
                         row=1, col=2)
            fig.add_annotation(x=0.05, y=0.95, xref="x domain", yref="y domain",
                              text=f"R² = {res['r2']:.4f}", showarrow=False,
                              row=1, col=2)

        if 'sklearn' in resultados and resultados['sklearn']:
            res = resultados['sklearn']
            y_plot = res['intercepto'] + res['pendiente'] * x_plot
            fig.add_trace(scatter, row=1, col=3)
            fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines',
                                    name='Scikit-learn', line=dict(color='orange', width=2)),
                         row=1, col=3)
            fig.add_annotation(x=0.05, y=0.95, xref="x domain", yref="y domain",
                              text=f"R² = {res['r2']:.4f}", showarrow=False,
                              row=1, col=3)

        fig.update_layout(height=450, showlegend=False, title_text="Comparacion de Metodos de Regresion")
        fig.update_xaxes(title_text="X")
        fig.update_yaxes(title_text="Y")

        return fig
    except Exception as e:
        return None

def graficar_bootstrap_regresion(x, y, resultado_bootstrap):
    """Genera gráfico de regresión con Bootstrap"""
    try:
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=['Regresion con Bootstrap', 'Distribucion Bootstrap'])

        x_plot = np.linspace(x.min(), x.max(), 100)
        y_plot = resultado_bootstrap['intercepto'] + resultado_bootstrap['pendiente'] * x_plot

        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Datos',
                                marker=dict(color='blue', opacity=0.6)),
                     row=1, col=1)

        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines',
                                name='Bootstrap', line=dict(color='purple', width=3)),
                     row=1, col=1)

        if 'intercepto_ci' in resultado_bootstrap:
            y_lower = resultado_bootstrap['intercepto_ci'][0] + resultado_bootstrap['pendiente_ci'][0] * x_plot
            y_upper = resultado_bootstrap['intercepto_ci'][1] + resultado_bootstrap['pendiente_ci'][1] * x_plot

            fig.add_trace(go.Scatter(x=np.concatenate([x_plot, x_plot[::-1]]),
                                    y=np.concatenate([y_lower, y_upper[::-1]]),
                                    fill='toself', fillcolor='rgba(128, 0, 128, 0.2)',
                                    line=dict(color='rgba(128, 0, 128, 0)'),
                                    name=f'IC {resultado_bootstrap["ci"]*100:.0f}%'),
                         row=1, col=1)

        fig.add_annotation(x=0.05, y=0.95, xref="x domain", yref="y domain",
                          text=f"R² = {resultado_bootstrap['r2']:.4f}", showarrow=False,
                          row=1, col=1)

        fig.add_trace(go.Histogram(x=resultado_bootstrap['bootstrap_a'],
                                  name='Intercepto', opacity=0.7, nbinsx=30),
                     row=1, col=2)
        fig.add_trace(go.Histogram(x=resultado_bootstrap['bootstrap_b'],
                                  name='Pendiente', opacity=0.7, nbinsx=30),
                     row=1, col=2)

        fig.update_layout(height=500, showlegend=True, title_text="Regresion Lineal con Bootstrap")
        fig.update_xaxes(title_text="Coeficiente", row=1, col=2)
        fig.update_yaxes(title_text="Frecuencia", row=1, col=2)

        return fig
    except Exception as e:
        return None

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Analizador de Datos Completo",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analizador de Datos - Completo")
st.markdown("---")

# ============================================================
# INICIALIZAR SESSION STATE
# ============================================================

if 'data' not in st.session_state:
    st.session_state.data = None
if 'form_data' not in st.session_state:
    st.session_state.form_data = []

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📝 Ingreso de Datos",
    "📊 Estadisticas Descriptivas",
    "📈 Estadistica Inferencial",
    "🎯 Analisis Multivariable (PCA)",
    "📐 Regresion Lineal",
    "🔄 Bootstrap",
    "🤖 Machine Learning",
    "📊 Regresion Avanzada",
    "📖 Guia"
])

# ============================================================
# PESTAÑA 1: INGRESO DE DATOS
# ============================================================

with tab1:
    st.subheader("📝 Ingreso de Datos")

    opcion_ingreso = st.radio(
        "Selecciona la forma de ingresar datos:",
        ["📋 Ingreso Manual", "📂 Importar CSV"],
        key="opcion_ingreso_1"
    )

    if opcion_ingreso == "📂 Importar CSV":
        st.subheader("📂 Importar Archivo CSV")

        with st.expander("💡 Consejos para importar CSV"):
            st.markdown("""
            **Formatos soportados:**
            - Separadores: `,` `;` `\t` `|`
            - Codificacion: UTF-8, Latin-1

            **Recomendaciones:**
            - Guarda el archivo en formato CSV estandar
            - Usa Excel → Guardar como → CSV (delimitado por comas)
            - Verifica que todas las filas tengan el mismo numero de columnas
            """)

        archivo = st.file_uploader("Selecciona un archivo CSV", type=['csv'], key="archivo_csv_1")

        if archivo is not None:
            with st.spinner("📖 Leyendo archivo..."):
                df = leer_csv_robusto(archivo)

                if df is not None:
                    df = df.dropna(axis=1, how='all')
                    df = df.dropna(axis=0, how='all')

                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except:
                            pass

                    if len(df) > 0 and len(df.columns) > 1:
                        st.session_state.data = df
                        st.success(f"✅ Datos importados: {df.shape[0]} filas, {df.shape[1]} columnas")
                        st.dataframe(df.head(10))

                        with st.expander("📊 Informacion del Archivo"):
                            col_info = pd.DataFrame({
                                'Columna': df.columns,
                                'Tipo': df.dtypes.astype(str),
                                'Valores no nulos': df.count().values,
                                'Valores nulos': df.isna().sum().values
                            })
                            st.dataframe(col_info)
                    else:
                        st.error("❌ Error: El archivo no tiene datos validos.")
                else:
                    st.error("❌ Error: No se pudo leer el archivo.")

    else:
        st.subheader("📋 Formulario de Ingreso de Datos")

        with st.form("formulario_datos"):
            col1, col2 = st.columns(2)

            with col1:
                nombre = st.text_input("👤 Nombre", placeholder="Ingresa tu nombre", key="nombre_1")
                edad = st.number_input("🎂 Edad", min_value=15, max_value=80, value=18, key="edad_1")
                sexo = st.selectbox("👤 Sexo", ["", "Masculino", "Femenino", "Prefiero no decir"], key="sexo_1")
                ciudad = st.text_input("📍 Ciudad", placeholder="Bogota", key="ciudad_1")
                programa = st.text_input("🎓 Programa", placeholder="Ingenieria de Sistemas", key="programa_1")
                semestre = st.number_input("📚 Semestre", min_value=1, max_value=12, value=1, key="semestre_1")

            with col2:
                puntuacion = st.number_input("⭐ Puntuacion", min_value=0, max_value=100, value=50, key="puntuacion_1")
                horas_estudio = st.number_input("📖 Horas de estudio/semana", min_value=0, max_value=50, value=10, key="horas_estudio_1")
                estrato = st.selectbox("🏠 Estrato", ["", "1", "2", "3", "4", "5", "6"], key="estrato_1")
                transporte = st.selectbox("🚌 Transporte", ["", "Publico", "Bicicleta", "Carro", "Moto", "A pie"], key="transporte_1")
                nivel_estres = st.slider("😰 Nivel de Estres", 1, 10, 5, key="nivel_estres_1")
                motivacion = st.slider("💪 Motivacion Academica", 1, 10, 7, key="motivacion_1")

            st.subheader("📊 Variables Adicionales")
            col3, col4 = st.columns(2)

            with col3:
                autoestima = st.slider("💫 Autoestima", 1, 10, 6, key="autoestima_1")
                ansiedad = st.slider("😟 Ansiedad", 1, 10, 5, key="ansiedad_1")
                apoyo_familiar = st.slider("👨‍👩‍👧‍👦 Apoyo Familiar", 1, 10, 7, key="apoyo_familiar_1")
                habitos_estudio = st.slider("📚 Habitos de Estudio", 1, 10, 6, key="habitos_estudio_1")

            with col4:
                habilidades_cognitivas = st.slider("🧠 Habilidades Cognitivas", 1, 10, 7, key="habilidades_cognitivas_1")
                creatividad = st.slider("🎨 Creatividad", 1, 10, 6, key="creatividad_1")
                liderazgo = st.slider("👥 Liderazgo", 1, 10, 5, key="liderazgo_1")
                trabajo_equipo = st.slider("🤝 Trabajo en Equipo", 1, 10, 7, key="trabajo_equipo_1")

            st.subheader("📈 Variables de Rendimiento")
            col5, col6 = st.columns(2)

            with col5:
                nota_promedio = st.number_input("📊 Nota Promedio", min_value=0.0, max_value=5.0, value=3.5, step=0.1, key="nota_promedio_1")
                asistencia = st.slider("✅ Asistencia (%)", 0, 100, 85, key="asistencia_1")

            with col6:
                participacion = st.slider("🎯 Participacion en Clase", 1, 10, 6, key="participacion_1")
                rendimiento = st.slider("📈 Rendimiento Academico", 1, 10, 7, key="rendimiento_1")

            st.subheader("📌 Variables Personalizadas")
            st.caption("Agrega variables adicionales que quieras analizar")

            num_variables_extra = st.number_input("Numero de variables extra", min_value=0, max_value=5, value=0, key="num_variables_extra_1")

            variables_extra = {}
            for i in range(num_variables_extra):
                col7, col8 = st.columns(2)
                with col7:
                    nombre_var = st.text_input(f"Nombre variable {i+1}", key=f"var_name_{i}")
                with col8:
                    valor_var = st.text_input(f"Valor {i+1}", key=f"var_val_{i}")
                if nombre_var and valor_var:
                    try:
                        variables_extra[nombre_var] = float(valor_var)
                    except:
                        variables_extra[nombre_var] = valor_var

            enviado = st.form_submit_button("📤 Agregar Datos")

            if enviado:
                if not nombre:
                    st.warning("⚠️ El nombre es obligatorio")
                else:
                    nuevo_registro = {
                        'nombre': nombre,
                        'edad': edad,
                        'sexo': sexo,
                        'ciudad': ciudad,
                        'programa': programa,
                        'semestre': semestre,
                        'puntuacion': puntuacion,
                        'horas_estudio': horas_estudio,
                        'estrato': estrato,
                        'transporte': transporte,
                        'nivel_estres': nivel_estres,
                        'motivacion': motivacion,
                        'autoestima': autoestima,
                        'ansiedad': ansiedad,
                        'apoyo_familiar': apoyo_familiar,
                        'habitos_estudio': habitos_estudio,
                        'habilidades_cognitivas': habilidades_cognitivas,
                        'creatividad': creatividad,
                        'liderazgo': liderazgo,
                        'trabajo_equipo': trabajo_equipo,
                        'nota_promedio': nota_promedio,
                        'asistencia': asistencia,
                        'participacion': participacion,
                        'rendimiento': rendimiento
                    }

                    for k, v in variables_extra.items():
                        nuevo_registro[k] = v

                    st.session_state.form_data.append(nuevo_registro)

                    if len(st.session_state.form_data) > 0:
                        st.session_state.data = pd.DataFrame(st.session_state.form_data)

                    st.success(f"✅ Datos agregados correctamente. Total: {len(st.session_state.form_data)} registros")

        if st.session_state.data is not None and len(st.session_state.data) > 0:
            st.subheader("📊 Datos Ingresados")
            st.dataframe(st.session_state.data)
            st.info(f"📌 Total de registros: {len(st.session_state.data)}")

# ============================================================
# PESTAÑA 2: ESTADÍSTICAS DESCRIPTIVAS
# ============================================================

with tab2:
    st.subheader("📊 Estadisticas Descriptivas")

    if st.session_state.data is None or len(st.session_state.data) == 0:
        st.warning("⚠️ No hay datos cargados.")
    else:
        df = st.session_state.data
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Variables Numericas")
            if len(num_cols) > 0:
                try:
                    stats_desc = df[num_cols].describe().T
                    stats_desc['varianza'] = df[num_cols].var()
                    stats_desc['asimetria'] = df[num_cols].skew()
                    stats_desc['curtosis'] = df[num_cols].kurtosis()
                    st.dataframe(stats_desc)

                    if len(num_cols) >= 2:
                        st.subheader("📈 Matriz de Correlacion")
                        corr_matrix = df[num_cols].corr()
                        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                             title="Matriz de Correlacion",
                                             color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig_corr, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Error: {e}")

        with col2:
            st.subheader("📊 Variables Categoricas")
            if len(cat_cols) > 0:
                for col in cat_cols:
                    st.write(f"**{col}:**")
                    try:
                        freq = df[col].value_counts()
                        st.dataframe(freq)

                        if len(freq) > 1:
                            fig_bar = px.bar(x=freq.index, y=freq.values,
                                            title=f"Distribucion de {col}",
                                            labels={'x': col, 'y': 'Frecuencia'})
                            st.plotly_chart(fig_bar, use_container_width=True)
                    except:
                        st.write(f"  No se pudo procesar la columna {col}")

# ============================================================
# PESTAÑA 3: ESTADÍSTICA INFERENCIAL - COMPLETA
# ============================================================

with tab3:
    st.subheader("📈 Estadistica Inferencial")

    if st.session_state.data is None or len(st.session_state.data) == 0:
        st.warning("⚠️ No hay datos cargados.")
    else:
        df = st.session_state.data
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # ============================================================
        # 1. PRUEBA T PARA MUESTRAS INDEPENDIENTES
        # ============================================================
        if len(cat_cols) > 0 and len(num_cols) > 0:
            st.subheader("🔬 Prueba t para Muestras Independientes")

            col1, col2, col3 = st.columns(3)

            with col1:
                var_grupo = st.selectbox("Variable de grupo:", cat_cols, key="t_grupo_3")

            with col2:
                var_valor = st.selectbox("Variable de valor:", num_cols, key="t_valor_3")

            with col3:
                if st.button("🔍 Ejecutar Prueba t", key="btn_t_3"):
                    try:
                        grupos = df[var_grupo].unique()
                        if len(grupos) == 2:
                            grupo1 = df[df[var_grupo] == grupos[0]][var_valor].dropna()
                            grupo2 = df[df[var_grupo] == grupos[1]][var_valor].dropna()
                            if len(grupo1) > 1 and len(grupo2) > 1:
                                stat, p_val = ttest_ind(grupo1, grupo2)
                                st.write(f"**Estadistico t:** {stat:.4f}")
                                st.write(f"**p-valor:** {p_val:.6f}")
                                st.write(f"**Media {grupos[0]}:** {grupo1.mean():.2f} (n={len(grupo1)})")
                                st.write(f"**Media {grupos[1]}:** {grupo2.mean():.2f} (n={len(grupo2)})")
                                if p_val < 0.05:
                                    st.success("✅ Diferencia significativa (p < 0.05)")
                                else:
                                    st.warning("❌ No hay diferencia significativa (p >= 0.05)")
                            else:
                                st.warning("⚠️ Grupos con pocos datos")
                        else:
                            st.warning(f"⚠️ La variable tiene {len(grupos)} grupos. Se necesitan 2.")
                    except Exception as e:
                        st.warning(f"⚠️ Error: {e}")

        # ============================================================
        # 2. PRUEBA T PARA MUESTRAS RELACIONADAS (PAREADAS)
        # ============================================================
        if len(num_cols) >= 2:
            st.subheader("🔬 Prueba t para Muestras Relacionadas (Pareadas)")

            col1, col2, col3 = st.columns(3)

            with col1:
                var1_pareada = st.selectbox("Variable 1:", num_cols, key="t_pareada_var1")

            with col2:
                var2_pareada = st.selectbox("Variable 2:", [col for col in num_cols if col != var1_pareada], key="t_pareada_var2")

            with col3:
                if st.button("🔍 Ejecutar Prueba t Pareada", key="btn_t_pareada"):
                    try:
                        datos1 = df[var1_pareada].dropna()
                        datos2 = df[var2_pareada].dropna()

                        # Alinear datos
                        idx = datos1.index.intersection(datos2.index)
                        datos1 = datos1.loc[idx]
                        datos2 = datos2.loc[idx]

                        if len(datos1) > 1:
                            stat, p_val = ttest_rel(datos1, datos2)
                            st.write(f"**Estadistico t:** {stat:.4f}")
                            st.write(f"**p-valor:** {p_val:.6f}")
                            st.write(f"**Media var1:** {datos1.mean():.2f}")
                            st.write(f"**Media var2:** {datos2.mean():.2f}")
                            st.write(f"**Diferencia media:** {(datos1.mean() - datos2.mean()):.2f}")
                            if p_val < 0.05:
                                st.success("✅ Diferencia significativa (p < 0.05)")
                            else:
                                st.warning("❌ No hay diferencia significativa (p >= 0.05)")
                        else:
                            st.warning("⚠️ Se necesitan al menos 2 pares de datos.")
                    except Exception as e:
                        st.warning(f"⚠️ Error: {e}")

        # ============================================================
        # 3. ANOVA DE UN FACTOR
        # ============================================================
        if len(cat_cols) > 0 and len(num_cols) > 0:
            st.subheader("📊 ANOVA de un Factor")

            col1, col2, col3 = st.columns(3)

            with col1:
                var_grupo_anova = st.selectbox("Variable de grupo:", cat_cols, key="anova_grupo_3")

            with col2:
                var_valor_anova = st.selectbox("Variable de valor:", num_cols, key="anova_valor_3")

            with col3:
                if st.button("🔍 Ejecutar ANOVA", key="btn_anova_3"):
                    try:
                        grupos = df[var_grupo_anova].unique()
                        if len(grupos) >= 2:
                            datos_grupos = []
                            nombres_grupos = []
                            for g in grupos:
                                datos = df[df[var_grupo_anova] == g][var_valor_anova].dropna()
                                if len(datos) > 1:
                                    datos_grupos.append(datos)
                                    nombres_grupos.append(g)

                            if len(datos_grupos) >= 2:
                                stat, p_val = f_oneway(*datos_grupos)
                                st.write(f"**Estadistico F:** {stat:.4f}")
                                st.write(f"**p-valor:** {p_val:.6f}")
                                st.write(f"**Grupos:** {', '.join([str(g) for g in nombres_grupos])}")

                                for i, g in enumerate(nombres_grupos):
                                    st.write(f"**Media {g}:** {datos_grupos[i].mean():.2f} (n={len(datos_grupos[i])})")

                                if p_val < 0.05:
                                    st.success("✅ Diferencia significativa entre grupos (p < 0.05)")
                                else:
                                    st.warning("❌ No hay diferencia significativa entre grupos (p >= 0.05)")
                            else:
                                st.warning("⚠️ Grupos con pocos datos")
                        else:
                            st.warning(f"⚠️ La variable tiene {len(grupos)} grupos. Se necesitan al menos 2.")
                    except Exception as e:
                        st.warning(f"⚠️ Error: {e}")

        # ============================================================
        # 4. PRUEBA CHI-CUADRADO DE INDEPENDENCIA
        # ============================================================
        if len(cat_cols) >= 2:
            st.subheader("📊 Prueba Chi-Cuadrado de Independencia")

            col1, col2, col3 = st.columns(3)

            with col1:
                var1_chi = st.selectbox("Variable 1:", cat_cols, key="chi_var1_3")

            with col2:
                var2_chi = st.selectbox("Variable 2:", [col for col in cat_cols if col != var1_chi], key="chi_var2_3")

            with col3:
                if st.button("🔍 Ejecutar Chi-Cuadrado", key="btn_chi_3"):
                    try:
                        tabla = pd.crosstab(df[var1_chi], df[var2_chi])
                        if tabla.shape[0] > 1 and tabla.shape[1] > 1:
                            chi2, p_val, dof, expected = chi2_contingency(tabla)

                            st.write(f"**Chi-Cuadrado:** {chi2:.4f}")
                            st.write(f"**p-valor:** {p_val:.6f}")
                            st.write(f"**Grados de libertad:** {dof}")

                            st.write("**Tabla de Contingencia:**")
                            st.dataframe(tabla)

                            if p_val < 0.05:
                                st.success("✅ Hay dependencia entre las variables (p < 0.05)")
                            else:
                                st.warning("❌ No hay evidencia de dependencia (p >= 0.05)")
                        else:
                            st.warning("⚠️ La tabla de contingencia es muy pequeña.")
                    except Exception as e:
                        st.warning(f"⚠️ Error: {e}")

        # ============================================================
        # 5. PRUEBA DE MANN-WHITNEY (Alternativa no paramétrica a t)
        # ============================================================
        if len(cat_cols) > 0 and len(num_cols) > 0:
            st.subheader("🔬 Prueba de Mann-Whitney (Alternativa no paramétrica)")

            col1, col2, col3 = st.columns(3)

            with col1:
                var_grupo_mw = st.selectbox("Variable de grupo:", cat_cols, key="mw_grupo_3")

            with col2:
                var_valor_mw = st.selectbox("Variable de valor:", num_cols, key="mw_valor_3")

            with col3:
                if st.button("🔍 Ejecutar Mann-Whitney", key="btn_mw_3"):
                    try:
                        grupos = df[var_grupo_mw].unique()
                        if len(grupos) == 2:
                            grupo1 = df[df[var_grupo_mw] == grupos[0]][var_valor_mw].dropna()
                            grupo2 = df[df[var_grupo_mw] == grupos[1]][var_valor_mw].dropna()
                            if len(grupo1) > 1 and len(grupo2) > 1:
                                stat, p_val = mannwhitneyu(grupo1, grupo2)
                                st.write(f"**Estadistico U:** {stat:.4f}")
                                st.write(f"**p-valor:** {p_val:.6f}")
                                st.write(f"**Mediana {grupos[0]}:** {grupo1.median():.2f}")
                                st.write(f"**Mediana {grupos[1]}:** {grupo2.median():.2f}")
                                if p_val < 0.05:
                                    st.success("✅ Diferencia significativa (p < 0.05)")
                                else:
                                    st.warning("❌ No hay diferencia significativa (p >= 0.05)")
                            else:
                                st.warning("⚠️ Grupos con pocos datos")
                        else:
                            st.warning(f"⚠️ La variable tiene {len(grupos)} grupos. Se necesitan 2.")
                    except Exception as e:
                        st.warning(f"⚠️ Error: {e}")

        # ============================================================
        # 6. PRUEBA DE WILCOXON (Alternativa no paramétrica a t pareada)
        # ============================================================
        if len(num_cols) >= 2:
            st.subheader("🔬 Prueba de Wilcoxon (Alternativa no paramétrica pareada)")

            col1, col2, col3 = st.columns(3)

            with col1:
                var1_wilcox = st.selectbox("Variable 1:", num_cols, key="wilcox_var1_3")

            with col2:
                var2_wilcox = st.selectbox("Variable 2:", [col for col in num_cols if col != var1_wilcox], key="wilcox_var2_3")

            with col3:
                if st.button("🔍 Ejecutar Wilcoxon", key="btn_wilcox_3"):
                    try:
                        datos1 = df[var1_wilcox].dropna()
                        datos2 = df[var2_wilcox].dropna()

                        idx = datos1.index.intersection(datos2.index)
                        datos1 = datos1.loc[idx]
                        datos2 = datos2.loc[idx]

                        if len(datos1) > 1:
                            stat, p_val = wilcoxon(datos1, datos2)
                            st.write(f"**Estadistico W:** {stat:.4f}")
                            st.write(f"**p-valor:** {p_val:.6f}")
                            st.write(f"**Mediana var1:** {datos1.median():.2f}")
                            st.write(f"**Mediana var2:** {datos2.median():.2f}")
                            if p_val < 0.05:
                                st.success("✅ Diferencia significativa (p < 0.05)")
                            else:
                                st.warning("❌ No hay diferencia significativa (p >= 0.05)")
                        else:
                            st.warning("⚠️ Se necesitan al menos 2 pares de datos.")
                    except Exception as e:
                        st.warning(f"⚠️ Error: {e}")

        # ============================================================
        # 7. PRUEBA DE KRUSKAL-WALLIS (Alternativa no paramétrica a ANOVA)
        # ============================================================
        if len(cat_cols) > 0 and len(num_cols) > 0:
            st.subheader("📊 Prueba de Kruskal-Wallis (Alternativa no paramétrica a ANOVA)")

            col1, col2, col3 = st.columns(3)

            with col1:
                var_grupo_kw = st.selectbox("Variable de grupo:", cat_cols, key="kw_grupo_3")

            with col2:
                var_valor_kw = st.selectbox("Variable de valor:", num_cols, key="kw_valor_3")

            with col3:
                if st.button("🔍 Ejecutar Kruskal-Wallis", key="btn_kw_3"):
                    try:
                        grupos = df[var_grupo_kw].unique()
                        if len(grupos) >= 2:
                            datos_grupos = []
                            nombres_grupos = []
                            for g in grupos:
                                datos = df[df[var_grupo_kw] == g][var_valor_kw].dropna()
                                if len(datos) > 1:
                                    datos_grupos.append(datos)
                                    nombres_grupos.append(g)

                            if len(datos_grupos) >= 2:
                                stat, p_val = kruskal(*datos_grupos)
                                st.write(f"**Estadistico H:** {stat:.4f}")
                                st.write(f"**p-valor:** {p_val:.6f}")
                                st.write(f"**Grupos:** {', '.join([str(g) for g in nombres_grupos])}")

                                for i, g in enumerate(nombres_grupos):
                                    st.write(f"**Mediana {g}:** {datos_grupos[i].median():.2f} (n={len(datos_grupos[i])})")

                                if p_val < 0.05:
                                    st.success("✅ Diferencia significativa entre grupos (p < 0.05)")
                                else:
                                    st.warning("❌ No hay diferencia significativa entre grupos (p >= 0.05)")
                            else:
                                st.warning("⚠️ Grupos con pocos datos")
                        else:
                            st.warning(f"⚠️ La variable tiene {len(grupos)} grupos. Se necesitan al menos 2.")
                    except Exception as e:
                        st.warning(f"⚠️ Error: {e}")

# ============================================================
# PESTAÑA 4: PCA
# ============================================================

with tab4:
    st.subheader("🎯 Analisis de Componentes Principales (PCA)")

    if st.session_state.data is None or len(st.session_state.data) == 0:
        st.warning("⚠️ No hay datos cargados.")
    else:
        df = st.session_state.data
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numericas.")
        else:
            vars_seleccionadas = st.multiselect(
                "Selecciona las variables para PCA:",
                num_cols,
                default=num_cols[:min(5, len(num_cols))],
                key="vars_pca_4"
            )

            if len(vars_seleccionadas) >= 2:
                n_componentes = st.slider("Numero de componentes:", 2, min(5, len(vars_seleccionadas)), 2, key="n_componentes_4")

                if st.button("🔍 Ejecutar PCA", key="btn_pca_4"):
                    try:
                        X = df[vars_seleccionadas].dropna()
                        if len(X) < 3:
                            st.warning("⚠️ Se necesitan al menos 3 observaciones.")
                        else:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)

                            n_comp = min(n_componentes, X.shape[1], X.shape[0]-1)
                            pca = PCA(n_components=n_comp)
                            componentes = pca.fit_transform(X_scaled)

                            var_exp = pca.explained_variance_ratio_
                            var_acum = np.cumsum(var_exp)

                            st.write("**Varianza explicada:**")
                            for i, v in enumerate(var_exp):
                                st.write(f"PC{i+1}: {v*100:.2f}%")

                            st.write("**Varianza acumulada:**")
                            for i, v in enumerate(var_acum):
                                st.write(f"PC{i+1}: {v*100:.2f}%")

                            loadings = pd.DataFrame(
                                pca.components_.T,
                                columns=[f'PC{i+1}' for i in range(n_comp)],
                                index=vars_seleccionadas
                            )
                            st.subheader("📊 Loadings")
                            st.dataframe(loadings)

                            if n_comp >= 2:
                                componentes_df = pd.DataFrame(
                                    componentes,
                                    columns=[f'PC{i+1}' for i in range(n_comp)]
                                )
                                fig_pca = px.scatter(
                                    componentes_df, x='PC1', y='PC2',
                                    title='Proyeccion en Componentes Principales',
                                    labels={'PC1': f'PC1 ({var_exp[0]*100:.1f}%)',
                                           'PC2': f'PC2 ({var_exp[1]*100:.1f}%)'}
                                )
                                st.plotly_chart(fig_pca, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Error: {e}")

# ============================================================
# PESTAÑA 5: REGRESIÓN LINEAL
# ============================================================

with tab5:
    st.subheader("📐 Regresion Lineal - Minimos Cuadrados")

    if st.session_state.data is None or len(st.session_state.data) == 0:
        st.warning("⚠️ No hay datos cargados.")
    else:
        df = st.session_state.data
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numericas.")
        else:
            target = st.selectbox("🎯 Variable objetivo (Y):", num_cols, key="target_5")
            features = st.multiselect(
                "📊 Variables predictoras (X):",
                [col for col in num_cols if col != target],
                default=[col for col in num_cols if col != target][:min(2, len(num_cols)-1)],
                key="features_5"
            )

            if len(features) >= 1:
                if st.button("🔍 Ejecutar Regresion", key="btn_regresion_5"):
                    try:
                        X = df[features].dropna()
                        y = df[target].loc[X.index].dropna()

                        if len(X) < 10:
                            st.warning("⚠️ Se necesitan al menos 10 observaciones.")
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )

                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)

                            st.write(f"**R²:** {r2:.4f}")
                            st.write(f"**MSE:** {mse:.4f}")
                            st.write(f"**Intercepto:** {model.intercept_:.4f}")

                            for var, coef in zip(features, model.coef_):
                                st.write(f"  {var}: {coef:.4f}")

                            fig_pred = go.Figure()
                            fig_pred.add_trace(go.Scatter(
                                x=y_test,
                                y=y_pred,
                                mode='markers',
                                name='Predicciones'
                            ))
                            fig_pred.add_trace(go.Scatter(
                                x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode='lines',
                                name='Ideal',
                                line=dict(color='red', dash='dash')
                            ))
                            fig_pred.update_layout(
                                title='Valores Reales vs Predichos',
                                height=400
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Error: {e}")

# ============================================================
# PESTAÑA 6: BOOTSTRAP
# ============================================================

with tab6:
    st.subheader("🔄 Bootstrap - Intervalos de Confianza")

    if st.session_state.data is None or len(st.session_state.data) == 0:
        st.warning("⚠️ No hay datos cargados.")
    else:
        df = st.session_state.data
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) == 0:
            st.warning("⚠️ No hay variables numericas.")
        else:
            var_bootstrap = st.selectbox("Selecciona la variable:", num_cols, key="var_bootstrap_6")
            n_bootstrap = st.number_input("Numero de remuestreos:", min_value=100, max_value=10000, value=1000, step=100, key="n_bootstrap_6")
            ci = st.slider("Nivel de confianza:", 0.80, 0.99, 0.95, 0.01, key="ci_6")

            if st.button("🔍 Ejecutar Bootstrap", key="btn_bootstrap_6"):
                try:
                    datos = df[var_bootstrap].dropna()
                    if len(datos) < 5:
                        st.warning("⚠️ Se necesitan al menos 5 datos.")
                    else:
                        np.random.seed(42)
                        medias = []
                        for _ in range(n_bootstrap):
                            muestra = np.random.choice(datos, size=len(datos), replace=True)
                            medias.append(np.mean(muestra))

                        lower = np.percentile(medias, (1-ci)/2 * 100)
                        upper = np.percentile(medias, (1+ci)/2 * 100)

                        st.write(f"**Media:** {np.mean(datos):.4f}")
                        st.write(f"**IC {ci*100:.0f}%:** [{lower:.4f}, {upper:.4f}]")

                        fig_boot = go.Figure()
                        fig_boot.add_trace(go.Histogram(x=medias, nbinsx=30, name='Medias Bootstrap'))
                        fig_boot.add_vline(x=lower, line_dash="dash", line_color="red")
                        fig_boot.add_vline(x=upper, line_dash="dash", line_color="red")
                        fig_boot.add_vline(x=np.mean(datos), line_color="blue")
                        fig_boot.update_layout(title='Distribucion de Medias Bootstrap', height=400)
                        st.plotly_chart(fig_boot, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Error: {e}")

# ============================================================
# PESTAÑA 7: MACHINE LEARNING
# ============================================================

with tab7:
    st.subheader("🤖 Machine Learning")

    if st.session_state.data is None or len(st.session_state.data) == 0:
        st.warning("⚠️ No hay datos cargados.")
    else:
        df = st.session_state.data
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        tipo_ml = st.selectbox(
            "Tipo de problema:",
            ["Regresion", "Clasificacion"],
            key="tipo_ml_7"
        )

        if tipo_ml == "Regresion":
            if len(num_cols) < 2:
                st.warning("⚠️ Se necesitan al menos 2 variables numericas.")
            else:
                target = st.selectbox("🎯 Variable objetivo (Y):", num_cols, key="ml_target_reg_7")
                features = st.multiselect(
                    "📊 Variables predictoras:",
                    [col for col in num_cols if col != target],
                    default=[col for col in num_cols if col != target][:min(2, len(num_cols)-1)],
                    key="ml_features_reg_7"
                )

                if len(features) >= 1:
                    modelo_ml = st.selectbox(
                        "Modelo:",
                        ["Random Forest", "Regresion Lineal"],
                        key="ml_model_reg_7"
                    )

                    if st.button("🔍 Entrenar Modelo", key="btn_ml_reg_7"):
                        try:
                            X = df[features].dropna()
                            y = df[target].loc[X.index].dropna()

                            if len(X) < 10:
                                st.warning("⚠️ Se necesitan al menos 10 observaciones.")
                            else:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=42
                                )

                                if modelo_ml == "Random Forest":
                                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                                else:
                                    model = LinearRegression()

                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                                st.write(f"**R²:** {r2_score(y_test, y_pred):.4f}")
                                st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.4f}")

                                fig_ml = go.Figure()
                                fig_ml.add_trace(go.Scatter(
                                    x=y_test,
                                    y=y_pred,
                                    mode='markers',
                                    name='Predicciones'
                                ))
                                fig_ml.add_trace(go.Scatter(
                                    x=[y_test.min(), y_test.max()],
                                    y=[y_test.min(), y_test.max()],
                                    mode='lines',
                                    name='Ideal',
                                    line=dict(color='red', dash='dash')
                                ))
                                fig_ml.update_layout(title='Valores Reales vs Predichos', height=400)
                                st.plotly_chart(fig_ml, use_container_width=True)

                                if hasattr(model, 'feature_importances_'):
                                    importancia = dict(zip(features, model.feature_importances_))
                                    fig_imp = go.Figure()
                                    fig_imp.add_trace(go.Bar(
                                        x=list(importancia.values()),
                                        y=list(importancia.keys()),
                                        orientation='h'
                                    ))
                                    fig_imp.update_layout(title='Importancia de Caracteristicas', height=300)
                                    st.plotly_chart(fig_imp, use_container_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ Error: {e}")

# ============================================================
# PESTAÑA 8: REGRESIÓN AVANZADA
# ============================================================

with tab8:
    st.subheader("📊 Regresion Lineal Avanzada - Comparacion de Metodos")

    if st.session_state.data is None or len(st.session_state.data) == 0:
        st.warning("⚠️ No hay datos cargados.")
    else:
        df = st.session_state.data
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) < 2:
            st.warning("⚠️ Se necesitan al menos 2 variables numericas.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                var_x = st.selectbox("📊 Variable Independiente (X):", num_cols, key="reg_x_8")

            with col2:
                var_y = st.selectbox("📊 Variable Dependiente (Y):",
                                    [col for col in num_cols if col != var_x], key="reg_y_8")

            st.subheader("🔄 Configuracion de Bootstrap")

            col3, col4 = st.columns(2)
            with col3:
                n_bootstrap = st.number_input("Numero de remuestreos:", min_value=100, max_value=10000, value=1000, step=100, key="n_bootstrap_8")
            with col4:
                ci_bootstrap = st.slider("Nivel de confianza:", 0.80, 0.99, 0.95, 0.01, key="ci_bootstrap_8")

            if st.button("🔍 Ejecutar Regresion Avanzada", key="btn_reg_avanzada_8"):
                try:
                    df_clean = df[[var_x, var_y]].dropna()
                    x = df_clean[var_x]
                    y = df_clean[var_y]

                    if len(x) < 5:
                        st.warning("⚠️ Se necesitan al menos 5 observaciones.")
                    else:
                        with st.spinner("🔄 Calculando regresiones..."):
                            res_curve = regresion_curve_fit(x, y)
                            res_ols = regresion_ols(x, y)
                            res_sklearn = regresion_sklearn(x, y)
                            res_bootstrap = regresion_bootstrap(x, y, n_bootstrap, ci_bootstrap)

                            resultados = {
                                'curve_fit': res_curve,
                                'ols': res_ols,
                                'sklearn': res_sklearn,
                                'bootstrap': res_bootstrap
                            }

                            st.subheader("📊 Comparacion de Metodos")

                            comparacion = pd.DataFrame({
                                'Metodo': [],
                                'Intercepto': [],
                                'Pendiente': [],
                                'R²': [],
                                'MSE': []
                            })

                            for key, res in resultados.items():
                                if res:
                                    comparacion = pd.concat([comparacion, pd.DataFrame({
                                        'Metodo': [res['metodo']],
                                        'Intercepto': [res['intercepto']],
                                        'Pendiente': [res['pendiente']],
                                        'R²': [res['r2']],
                                        'MSE': [res['mse']]
                                    })], ignore_index=True)

                            if len(comparacion) > 0:
                                st.dataframe(comparacion.style.format({
                                    'Intercepto': '{:.4f}',
                                    'Pendiente': '{:.4f}',
                                    'R²': '{:.4f}',
                                    'MSE': '{:.4f}'
                                }))

                            fig_comparacion = graficar_comparacion_regresiones(x, y, {
                                'curve_fit': res_curve,
                                'ols': res_ols,
                                'sklearn': res_sklearn
                            })

                            if fig_comparacion:
                                st.plotly_chart(fig_comparacion, use_container_width=True)

                            if res_bootstrap:
                                st.subheader("🔄 Resultados de Bootstrap")
                                st.write(f"**Intercepto promedio:** {res_bootstrap['intercepto']:.4f}")
                                st.write(f"**Pendiente promedio:** {res_bootstrap['pendiente']:.4f}")
                                st.write(f"**IC {res_bootstrap['ci']*100:.0f}% Intercepto:** [{res_bootstrap['intercepto_ci'][0]:.4f}, {res_bootstrap['intercepto_ci'][1]:.4f}]")
                                st.write(f"**IC {res_bootstrap['ci']*100:.0f}% Pendiente:** [{res_bootstrap['pendiente_ci'][0]:.4f}, {res_bootstrap['pendiente_ci'][1]:.4f}]")

                                fig_bootstrap = graficar_bootstrap_regresion(x, y, res_bootstrap)
                                if fig_bootstrap:
                                    st.plotly_chart(fig_bootstrap, use_container_width=True)

                                if st.button("📥 Descargar Coeficientes Bootstrap", key="btn_descargar_bootstrap_8"):
                                    df_bootstrap = pd.DataFrame({
                                        'Intercepto': res_bootstrap['bootstrap_a'],
                                        'Pendiente': res_bootstrap['bootstrap_b']
                                    })
                                    csv = df_bootstrap.to_csv(index=False)
                                    st.download_button(
                                        label="Descargar CSV",
                                        data=csv,
                                        file_name="coeficientes_bootstrap.csv",
                                        mime="text/csv",
                                        key="download_bootstrap_8"
                                    )
                except Exception as e:
                    st.warning(f"⚠️ Error: {e}")

# ============================================================
# PESTAÑA 9: GUÍA
# ============================================================

with tab9:
    st.write("""
    # 📖 Guia de Uso - Analizador de Datos Completo

    ## 📊 Estadistica Inferencial - Pruebas Disponibles

    ### Pruebas Paramétricas
    1. **Prueba t para muestras independientes**: Compara medias entre 2 grupos
    2. **Prueba t para muestras relacionadas (pareadas)**: Compara medias en mediciones repetidas
    3. **ANOVA de un factor**: Compara medias entre 3 o más grupos

    ### Pruebas No Paramétricas
    4. **Mann-Whitney**: Alternativa no paramétrica a la t independiente
    5. **Wilcoxon**: Alternativa no paramétrica a la t pareada
    6. **Kruskal-Wallis**: Alternativa no paramétrica a ANOVA

    ### Pruebas de Asociación
    7. **Chi-Cuadrado de Independencia**: Analiza relación entre variables categóricas

    ## 📊 Regresion Avanzada

    ### 3 Metodos de Regresion
    1. **Optimize.curve_fit**: Usa scipy.optimize.curve_fit
    2. **Minimos Cuadrados (OLS)**: Usa statsmodels
    3. **Scikit-learn**: Usa LinearRegression de sklearn

    ### Bootstrap
    - **Remuestreo**: Genera multiples muestras con reemplazo
    - **Coeficientes**: Distribucion de los coeficientes
    - **Intervalos de Confianza**: Estimacion robusta

    ## 📝 Ingreso de Datos
    - **Ingreso Manual**: Completa el formulario
    - **Importar CSV**: Sube un archivo CSV

    ## 📊 Estadisticas Descriptivas
    - Variables numericas y categoricas
    - Matriz de correlacion

    ## 🎯 Analisis Multivariable (PCA)
    - Reduccion de dimensionalidad

    ## 🤖 Machine Learning
    - Regresion y Clasificacion
    - Random Forest, Regresion Logistica
    """)
