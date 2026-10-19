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

# Variables exclusivas para la pestaña Fraccionado
if 'fr_df' not in st.session_state: st.session_state.fr_df = None
if 'fr_analizado' not in st.session_state: st.session_state.fr_analizado = False
if 'fr_colapsado' not in st.session_state: st.session_state.fr_colapsado = None

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
        horizontal=True, key=f"tipo_anova_{key_suffix}"
    )
    tipo_map = {"Tipo I (Secuencial)": 1, "Tipo II (Parcial)": 2, "Tipo III (Parcial con interacción)": 3}
    return tipo_map[tipo_anova]

def analizar_anova_mejorada(df, formula, titulo="ANOVA", key_suffix="", tipo="dca", anova_type=2):
    try:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
        modelo = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(modelo, typ=anova_type)
        if 'mean_sq' not in anova_table.columns:
            anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
        rename_map = {'sum_sq': 'Sum Sq', 'df': 'DF', 'mean_sq': 'Mean Sq', 'F': 'F-Ratio', 'PR(>F)': 'P-Value'}
        anova_table = anova_table.rename(columns=rename_map)
        columnas_orden = [c for c in anova_table.columns if c not in ['Mean Sq']]
        if 'DF' in columnas_orden:
            idx_df = columnas_orden.index('DF')
            columnas_orden.insert(idx_df + 1, 'Mean Sq')
            anova_table = anova_table[columnas_orden]
            
        # Guardamos el resultado
        st.session_state.resultados_anova[key_suffix] = {
            'modelo': modelo, 'anova': anova_table, 'df': df, 'tipo': tipo,
            'formula': formula, 'anova_type': anova_type, 'titulo': titulo
        }
        
        st.success(f"✅ {titulo} calculado (Tipo {anova_type})")
        st.dataframe(anova_table.style.format({
            'Sum Sq': '{:.4f}', 'Mean Sq': '{:.4f}', 'F-Ratio': '{:.4f}', 'P-Value': '{:.6f}'
        }, na_rep="None"))
        st.write(f"**R² = {modelo.rsquared:.6f}**")
        st.write(f"**R² ajustado = {modelo.rsquared_adj:.6f}**")
        return modelo, anova_table
    except Exception as e:
        st.error(f"❌ Error en ANOVA: {str(e)}")
        return None, None

def mostrar_tabla_anova(key_suffix):
    """Imprime de forma persistente la tabla ANOVA guardada en session_state."""
    if key_suffix in st.session_state.resultados_anova:
        res = st.session_state.resultados_anova[key_suffix]
        st.success(f"✅ {res.get('titulo', 'ANOVA')} calculado (Tipo {res.get('anova_type', 2)})")
        st.dataframe(res['anova'].style.format({
            'Sum Sq': '{:.4f}', 'Mean Sq': '{:.4f}', 'F-Ratio': '{:.4f}', 'P-Value': '{:.6f}'
        }, na_rep="None"))
        st.write(f"**R² = {res['modelo'].rsquared:.6f}** | **R² ajustado = {res['modelo'].rsquared_adj:.6f}**")

def graficos_interaccion_multifactorial(df, factores, respuesta, max_pares=6):
    if len(factores) < 2:
        return None
    pares = list(combinations(factores, 2))[:max_pares]
    n_filas = (len(pares) + 1) // 2
    fig = make_subplots(rows=n_filas, cols=2, subplot_titles=[f"{f1} × {f2}" for f1, f2 in pares])
    for idx, (f1, f2) in enumerate(pares):
        row = idx // 2 + 1
        col = idx % 2 + 1
        medias = df.groupby([f1, f2])[respuesta].mean().reset_index()
        for nivel in medias[f2].unique():
            datos_nivel = medias[medias[f2] == nivel]
            fig.add_trace(go.Scatter(x=datos_nivel[f1], y=datos_nivel[respuesta],
                                     mode='lines+markers', name=f"{f2}={nivel}",
                                     legendgroup=f"{f2}={nivel}", showlegend=(idx == 0)),
                          row=row, col=col)
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
                        color_continuous_scale='RdBu_r', title=f"Mapa de Calor: {f1} × {f2}")
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
        fig = px.bar(df_sc, x='Fuente', y=col_sc, title="Gráfico de Sumas de Cuadrados",
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
    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', marker=dict(color='blue', size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[min(osm), max(osm)],
                             y=[min(osm)*slope + intercept, max(osm)*slope + intercept],
                             mode='lines', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=predichos, y=residuos, mode='markers', marker=dict(color='green', size=6)), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_trace(go.Scatter(x=run_order, y=residuos, mode='lines+markers', marker=dict(color='orange', size=6)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_trace(go.Histogram(x=residuos, marker_color='purple', nbinsx=15), row=2, col=2)
    fig.update_layout(height=700, showlegend=False, title_text="Verificación de Supuestos")
    return fig

def graficar_interaccion_mejorada(df, f1, f2, respuesta):
    medias = df.groupby([f1, f2])[respuesta].mean().reset_index()
    fig = px.line(medias, x=f1, y=respuesta, color=f2, markers=True, title=f"Interacción {f1} × {f2}")
    for i, row in medias.iterrows():
        fig.add_annotation(x=row[f1], y=row[respuesta], text=f"{row[respuesta]:.2f}",
                           showarrow=False, yshift=10, font=dict(size=10))
    fig.update_layout(height=400)
    return fig

def graficar_efectos_e_interacciones(df, factores, resp):
    """Genera gráficos de líneas para Efectos Principales e Interacciones de 2 vías."""
    st.markdown("---")
    st.subheader("📈 Gráficos de Efectos Principales")
    cols = st.columns(min(len(factores), 3))
    for i, f in enumerate(factores):
        with cols[i % len(cols)]:
            medias = df.groupby(f)[resp].mean().reset_index()
            fig = px.line(medias, x=f, y=resp, markers=True, title=f"Efecto Principal: {f}")
            fig.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
    if len(factores) >= 2:
        st.subheader("🔄 Gráficos de Interacciones (Pares)")
        pares = list(combinations(factores, 2))
        if len(pares) > 6:
            st.caption("Mostrando las primeras 6 interacciones...")
            pares = pares[:6]
        
        cols_int = st.columns(2)
        for i, (f1, f2) in enumerate(pares):
            with cols_int[i % 2]:
                medias = df.groupby([f1, f2])[resp].mean().reset_index()
                fig = px.line(medias, x=f1, y=resp, color=f2, markers=True, title=f"Interacción: {f1} × {f2}")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# ⭐ NUEVO: SUMMARY STATISTICS (ESTILO STATGRAPHICS)
# ============================================================

def calcular_summary_statistics(modelo, df, factores, respuesta, alpha=0.05):
    from scipy.stats import t as t_dist
    try:
        anova_table = sm.stats.anova_lm(modelo, typ=2)
        if 'Residual' in anova_table.index:
            mse = anova_table.loc['Residual', 'sum_sq'] / anova_table.loc['Residual', 'df']
            gl_error = anova_table.loc['Residual', 'df']
        else:
            mse = modelo.mse_resid
            gl_error = modelo.df_resid
    except Exception:
        mse = modelo.mse_resid
        gl_error = modelo.df_resid

    t_crit = t_dist.ppf(1 - alpha / 2, gl_error) if gl_error > 0 else np.nan

    filas = []
    grand_mean = df[respuesta].mean()
    n_total = len(df)
    se_grand = np.sqrt(mse / n_total) if pd.notna(mse) and mse > 0 else np.nan
    filas.append({
        'Level': 'GRAND MEAN', 'Count': n_total, 'Mean': grand_mean,
        'Stnd. Error': se_grand,
        'Lower Limit': grand_mean - t_crit * se_grand if pd.notna(t_crit) else np.nan,
        'Upper Limit': grand_mean + t_crit * se_grand if pd.notna(t_crit) else np.nan,
    })

    for factor in factores:
        filas.append({'Level': factor, 'Count': None, 'Mean': None,
                      'Stnd. Error': None, 'Lower Limit': None, 'Upper Limit': None})
        try:
            niveles = sorted(df[factor].dropna().unique(), key=lambda x: (str(type(x)), str(x)))
        except Exception:
            niveles = list(df[factor].dropna().unique())
        if len(niveles) > 30:
            filas.append({'Level': '   (factor continuo)', 'Count': None, 'Mean': None,
                          'Stnd. Error': None, 'Lower Limit': None, 'Upper Limit': None})
            continue
        for nivel in niveles:
            sub = df[df[factor] == nivel]
            n = len(sub)
            if n == 0:
                continue
            media = sub[respuesta].mean()
            se = np.sqrt(mse / n) if pd.notna(mse) and mse > 0 else np.nan
            filas.append({
                'Level': str(nivel), 'Count': n, 'Mean': media, 'Stnd. Error': se,
                'Lower Limit': media - t_crit * se if pd.notna(t_crit) else np.nan,
                'Upper Limit': media + t_crit * se if pd.notna(t_crit) else np.nan,
            })

    return pd.DataFrame(filas), mse, gl_error, t_crit

def mostrar_summary_statistics(modelo, df, factores, respuesta, key_suffix="", alpha=0.05, incluir_descriptivas=True):
    st.subheader("📋 Summary Statistics — Least Squares Means (95% CI)")
    try:
        tabla, mse, gl_error, t_crit = calcular_summary_statistics(modelo, df, factores, respuesta, alpha)
        tabla_fmt = tabla.copy()
        for col in ['Mean', 'Stnd. Error', 'Lower Limit', 'Upper Limit']:
            tabla_fmt[col] = tabla_fmt[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) and isinstance(x, (int, float, np.floating)) else "")
        tabla_fmt['Count'] = tabla_fmt['Count'].apply(lambda x: str(int(x)) if pd.notna(x) and x is not None and x != "" else "")

        st.caption(
            f"**MSE = {mse:.6f}  |  GL Error = {int(gl_error)}  |  "
            f"Nivel de confianza = {(1-alpha)*100:.1f}%  |  t crítico = {t_crit:.4f}**"
        )
        st.dataframe(tabla_fmt, use_container_width=True, hide_index=True)

        csv = tabla.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar tabla LS Means (CSV)", data=csv, file_name=f"lsmeans_{key_suffix}.csv", mime="text/csv", key=f"dl_ss_{key_suffix}")

        if incluir_descriptivas:
            with st.expander("📈 Estadísticas descriptivas (media, mediana, moda, cuartiles, CV)", expanded=False):
                filas_desc = []
                for factor in factores:
                    try:
                        niveles = sorted(df[factor].dropna().unique(), key=lambda x: (str(type(x)), str(x)))
                    except Exception:
                        niveles = list(df[factor].dropna().unique())
                    if len(niveles) > 30: continue
                    for nivel in niveles:
                        s = df[df[factor] == nivel][respuesta]
                        moda_vals = s.mode()
                        filas_desc.append({
                            'Factor': factor, 'Nivel': str(nivel), 'N': len(s),
                            'Media': s.mean(), 'Mediana': s.median(),
                            'Moda': moda_vals.iloc[0] if len(moda_vals) > 0 else np.nan,
                            'Desv.Est.': s.std(), 'Varianza': s.var(),
                            'Mínimo': s.min(), 'Q1': s.quantile(0.25),
                            'Q3': s.quantile(0.75), 'Máximo': s.max(),
                            'CV (%)': (s.std() / s.mean()) * 100 if s.mean() != 0 else np.nan,
                        })
                s = df[respuesta]
                moda_vals = s.mode()
                filas_desc.insert(0, {
                    'Factor': 'GLOBAL', 'Nivel': '—', 'N': len(s),
                    'Media': s.mean(), 'Mediana': s.median(),
                    'Moda': moda_vals.iloc[0] if len(moda_vals) > 0 else np.nan,
                    'Desv.Est.': s.std(), 'Varianza': s.var(),
                    'Mínimo': s.min(), 'Q1': s.quantile(0.25),
                    'Q3': s.quantile(0.75), 'Máximo': s.max(),
                    'CV (%)': (s.std() / s.mean()) * 100 if s.mean() != 0 else np.nan,
                })
                df_desc = pd.DataFrame(filas_desc)
                st.dataframe(
                    df_desc.style.format({
                        'Media': '{:.4f}', 'Mediana': '{:.4f}', 'Moda': '{:.4f}',
                        'Desv.Est.': '{:.4f}', 'Varianza': '{:.4f}',
                        'Mínimo': '{:.4f}', 'Q1': '{:.4f}', 'Q3': '{:.4f}',
                        'Máximo': '{:.4f}', 'CV (%)': '{:.2f}'
                    }),
                    use_container_width=True, hide_index=True
                )
                st.download_button("📥 Descargar descriptivas (CSV)", data=df_desc.to_csv(index=False).encode('utf-8'), file_name=f"descriptivas_{key_suffix}.csv", mime="text/csv", key=f"dl_desc_{key_suffix}")
    except Exception as e:
        st.warning(f"⚠️ No se pudieron calcular las estadísticas: {e}")

# ============================================================
# PRUEBAS MÚLTIPLES Y RESIDUOS
# ============================================================

def realizar_tukey(df, factor, respuesta, alpha=0.05):
    try:
        df[factor] = df[factor].astype('category')
        return pairwise_tukeyhsd(df[respuesta], df[factor], alpha=alpha)
    except: return None

def interpretar_tukey(tukey):
    interpretacion = []
    if tukey is None: return ["❌ No se pudo realizar la prueba de Tukey"]
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

def realizar_lsd(df, factor, respuesta, mse, gl_error, alpha=0.05):
    try:
        if mse is None or gl_error is None or gl_error == 0: return None
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
    except Exception as e: return None

def interpretar_lsd(lsd_df):
    interpretacion = []
    if lsd_df is None or len(lsd_df) == 0: return ["❌ No se pudo realizar la prueba de LSD"]
    try:
        significativas = lsd_df[lsd_df['reject'] == True]
        if len(significativas) == 0:
            interpretacion.append("📊 **LSD:** No hay diferencias significativas")
            return interpretacion
        interpretacion.append(f"📊 **LSD:** {len(significativas)} diferencias significativas")
        for _, row in significativas.iterrows():
            interpretacion.append(f"  • {row['group1']} vs {row['group2']}: Diff = {row['meandiff']:.4f} (p = {row['p-adj']:.6f})")
    except Exception as e: interpretacion.append(f"❌ Error: {str(e)}")
    return interpretacion

def extraer_mse_gl(anova_table):
    try:
        if 'Residual' in anova_table.index: return anova_table.loc['Residual', 'Mean Sq'], anova_table.loc['Residual', 'DF']
        else: return anova_table['Mean Sq'].iloc[-1], anova_table['DF'].iloc[-1]
    except: return None, None

def mostrar_graficos_residuos_directo(modelo, df, factor=None, respuesta=None, key_suffix="", titulo="Residuos"):
    if modelo is None: return
    st.subheader(f"📈 {titulo}")
    with st.spinner("Generando gráficos de verificación del modelo..."):
        fig = graficar_verificacion_modelo_completa(modelo, df, factor, respuesta)
        if fig: st.plotly_chart(fig, use_container_width=True)
        st.subheader("📊 Pruebas de Normalidad de Residuos")
        from scipy.stats import shapiro, anderson
        residuos = modelo.resid
        col1, col2 = st.columns(2)
        with col1:
            try:
                shapiro_stat, shapiro_p = shapiro(residuos)
                st.metric("Shapiro-Wilk", f"p = {shapiro_p:.6f}")
                if shapiro_p > 0.05: st.success("✅ Los residuos siguen distribución normal (p > 0.05)")
                else: st.warning("⚠️ Los residuos NO siguen distribución normal (p ≤ 0.05)")
            except: st.warning("⚠️ No se pudo realizar la prueba de Shapiro-Wilk")
        with col2:
            try:
                anderson_result = anderson(residuos)
                st.metric("Anderson-Darling", f"Stat = {anderson_result.statistic:.4f}")
                if anderson_result.statistic < anderson_result.critical_values[2]: st.success("✅ Los residuos siguen distribución normal (al 5%)")
                else: st.warning("⚠️ Los residuos NO siguen distribución normal (al 5%)")
            except: st.warning("⚠️ No se pudo realizar la prueba de Anderson-Darling")

# ============================================================
# ⭐ NUEVO: FUNCIONES GLOBALES PARA PARETO, DANIEL Y EXCLUIR
# ============================================================

def mostrar_pareto_y_daniel(modelo, anova, resp, n_corridas):
    st.markdown("---")
    st.subheader("📊 Gráficos para Selección de Efectos (Pareto y Daniel)")
    try:
        def clean_name(name):
            return name.replace('C(', '').replace(')', '').replace('Factor_', '').replace(':', '')
        coefs = modelo.params.drop('Intercept', errors='ignore')
        names = [clean_name(x) for x in coefs.index]
        effects = 2 * coefs.values
        gl_error = anova.loc['Residual', 'DF'] if 'Residual' in anova.index else 0
        
        if gl_error > 0:
            mse = anova.loc['Residual', 'Mean Sq']
            se = np.sqrt(mse * (4 / n_corridas))
            limite = stats.t.ppf(0.975, gl_error) * se
        else:
            s0 = 1.5 * np.median(np.abs(effects))
            reduced = effects[np.abs(effects) < 2.5 * s0]
            pse = 1.5 * np.median(np.abs(reduced)) if len(reduced) > 0 else 1.5 * np.median(np.abs(effects))
            limite = 2.0 * pse
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Diagrama de Pareto**")
            df_pareto = pd.DataFrame({'Efecto': names, 'Magnitud': np.abs(effects)}).sort_values(by='Magnitud', ascending=True)
            colors = ['#2b8cbe' if m > limite else '#a6bddb' for m in df_pareto['Magnitud']]
            fig_pareto = go.Figure(go.Bar(x=df_pareto['Magnitud'], y=df_pareto['Efecto'], orientation='h', marker_color=colors))
            fig_pareto.add_vline(x=limite, line_dash="dash", line_color="red")
            fig_pareto.update_layout(xaxis_title="Efecto (|E|)", height=400, showlegend=False)
            st.plotly_chart(fig_pareto, use_container_width=True)
        with col2:
            st.markdown("**Gráfico de Daniel**")
            idx = np.argsort(effects)
            sorted_effects = effects[idx]
            n = len(sorted_effects)
            z = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            fig_daniel = go.Figure()
            fig_daniel.add_trace(go.Scatter(x=sorted_effects, y=z, mode='markers+text', text=[names[i] for i in idx], textposition="top center"))
            mask = np.abs(sorted_effects) < limite
            slope, intercept = np.polyfit(sorted_effects[mask], z[mask], 1) if sum(mask) > 1 else np.polyfit(sorted_effects, z, 1)
            fig_daniel.add_trace(go.Scatter(x=[min(sorted_effects), max(sorted_effects)], y=[min(sorted_effects) * slope + intercept, max(sorted_effects) * slope + intercept], mode='lines', line_color='blue'))
            fig_daniel.update_layout(xaxis_title="Efectos Estandarizados", yaxis_title="Proporción", height=400, showlegend=False)
            st.plotly_chart(fig_daniel, use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudieron generar los gráficos: {e}")

def ui_excluir_efectos(df, resp, anova, tipo_anova, key_suffix):
    st.markdown("---")
    st.subheader("🛠️ Refinamiento de Efectos (Excluir No Significativos)")
    todos_efectos = [idx for idx in anova.index if idx != 'Residual']
    efectos_seleccionados = st.multiselect("Efectos a MANTENER en el modelo final:", todos_efectos, default=todos_efectos, key=f"excluir_efectos_{key_suffix}")
    
    if st.button("🔄 Recalcular Modelo Reducido", key=f"btn_recalcular_{key_suffix}"):
        if len(efectos_seleccionados) == 0:
            st.warning("⚠️ Debes mantener al menos un efecto.")
        else:
            formula_reducida = f"{resp} ~ " + " + ".join(efectos_seleccionados)
            analizar_anova_mejorada(df, formula_reducida, "Modelo Reducido (Mejor ANOVA)", f"{key_suffix}_red", "reducido", tipo_anova)
            st.rerun()

# ============================================================
# CARGA DE DATOS, DISEÑO Y CONCLUSIONES
# ============================================================

def crear_diseno_completo(factores_dict, respuestas_dict, n_replicas=1, randomize=True):
    try:
        niveles_factores = [factores_dict[f]['niveles'] for f in factores_dict]
        nombres_factores = list(factores_dict.keys())
        combinaciones = list(product(*niveles_factores))
        df = pd.DataFrame(combinaciones, columns=nombres_factores)
        if n_replicas > 1: df = pd.concat([df] * n_replicas, ignore_index=True)
        if randomize: df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        for resp_name, resp_data in respuestas_dict.items():
            df[resp_name] = np.nan
        return df
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def generar_conclusiones_anova(anova_table, modelo, df, factor, respuesta, alpha=0.05):
    conclusiones = []
    p_valor = None
    for col in anova_table.columns:
        if 'P-Value' in col: p_valor = anova_table[col].iloc[0]; break
    if p_valor is None: return ["⚠️ No se obtuvo p-valor"], None
    if p_valor < alpha: conclusiones.append(f"✅ **Conclusión:** Diferencia significativa (p = {p_valor:.6f})")
    else: conclusiones.append(f"❌ **Conclusión:** Sin diferencia significativa (p = {p_valor:.6f})")
    medias = df.groupby(factor)[respuesta].agg(['mean', 'std', 'count']).reset_index()
    medias.columns = [factor, 'Media', 'Desv', 'n']
    medias_ordenadas = medias.sort_values('Media', ascending=False)
    for _, row in medias_ordenadas.iterrows():
        conclusiones.append(f"  • {row[factor]}: Media = {row['Media']:.4f} ± {row['Desv']:.4f} (n={int(row['n'])})")
    mejor, peor = medias_ordenadas.iloc[0], medias_ordenadas.iloc[-1]
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
            if 'P-Value' in col: p_valor = row[col]; break
        if p_valor is None: continue
        if p_valor < alpha: conclusiones.append(f"✅ **{i}:** Efecto significativo (p = {p_valor:.6f})")
        else: conclusiones.append(f"❌ **{i}:** No significativo (p = {p_valor:.6f})")
    conclusiones.append(f"📐 **R² = {modelo.rsquared:.6f}**")
    return conclusiones

def detectar_separador(texto):
    lineas = [l.strip() for l in texto.split('\n') if l.strip()]
    if not lineas: return ','
    conteos = {sep: sum(l.count(sep) for l in lineas) for sep in [',', ';', '\t', '|', ' ']}
    mejor_sep = max(conteos, key=conteos.get)
    return mejor_sep if conteos[mejor_sep] > 0 else ','

def limpiar_nombres_columnas(df):
    df.columns = [c.strip().replace(' ', '_').replace(';', '').replace(',', '') for c in df.columns]
    return df

def convertir_tipos(df):
    for col in df.columns:
        try: df[col] = pd.to_numeric(df[col])
        except: df[col] = df[col].astype('category')
    return df

def cargar_datos_desde_csv(archivo, separador=None):
    try:
        contenido = archivo.getvalue().decode('utf-8')
        if separador is None: separador = detectar_separador(contenido)
        try: df = pd.read_csv(io.StringIO(contenido), sep=separador)
        except:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(contenido), sep=sep)
                    if len(df.columns) > 1: break
                except: continue
        return convertir_tipos(limpiar_nombres_columnas(df))
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def cargar_datos_desde_url(url, separador=None):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        if separador is None: separador = detectar_separador(content)
        try: df = pd.read_csv(io.StringIO(content), sep=separador)
        except:
            for sep in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep)
                    if len(df.columns) > 1: break
                except: continue
        return convertir_tipos(limpiar_nombres_columnas(df))
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def procesar_datos_texto(texto, n_columnas=None):
    try:
        separador = detectar_separador(texto)
        lineas = [l.strip() for l in texto.split('\n') if l.strip()]
        if not lineas: return None
        valores = lineas[0].split(separador)
        tiene_encabezado = False
        try:
            for v in valores: float(v.strip())
        except: tiene_encabezado = True
        datos, columnas = [], None
        if tiene_encabezado:
            columnas = [v.strip() for v in valores]
            lineas = lineas[1:]
        for linea in lineas:
            valores = [v.strip() for v in linea.split(separador) if v.strip()]
            if n_columnas is None or len(valores) == n_columnas:
                try: datos.append([float(v) for v in valores])
                except: datos.append(valores)
        if not datos: return None
        df = pd.DataFrame(datos)
        if columnas and len(columnas) == len(df.columns): df.columns = columnas
        else: df.columns = [f'V{i+1}' for i in range(len(df.columns))]
        return convertir_tipos(limpiar_nombres_columnas(df))
    except: return None

def obtener_datos_generales(key_suffix, n_columnas_esperadas=None):
    st.subheader("📥 Carga de Datos")
    tipo_carga = st.radio("Método:", ["✏️ Manual", "📁 CSV", "🔗 URL"], horizontal=True, key=f"tipo_carga_{key_suffix}")
    df_resultado = None
    if tipo_carga == "✏️ Manual":
        if n_columnas_esperadas: st.caption(f"Formato: {n_columnas_esperadas} columnas")
        datos_texto = st.text_area("Datos:", value="", height=200, key=f"text_area_{key_suffix}", placeholder="A,23\nA,25\nB,30")
        if datos_texto and st.button("📊 Procesar", key=f"procesar_text_{key_suffix}"):
            df_resultado = procesar_datos_texto(datos_texto, n_columnas_esperadas)
            if df_resultado is not None:
                st.success(f"✅ {len(df_resultado)} filas cargadas")
                st.dataframe(df_resultado.head())
                st.session_state.datos_cargados[key_suffix] = df_resultado
    elif tipo_carga == "📁 CSV":
        archivo = st.file_uploader("Archivo:", type=['csv', 'txt'], key=f"csv_upload_{key_suffix}")
        if archivo is not None and st.button("📊 Cargar", key=f"cargar_csv_{key_suffix}"):
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
        with col1: st.metric("Filas", len(df))
        with col2: st.metric("Columnas", len(df.columns))

# ============================================================
# INTERFAZ PRINCIPAL - CREAR PESTAÑAS PRIMERO
# ============================================================

tab_interactivo, tab1, tab2, tab3, tab4, tab5, tab6, tab6_resultados, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "🎯 Diseño Interactivo", "📊 ANOVA Un Factor", "📊 Diseño en Bloques",
    "📐 Cuadrado Latino", "📐 Greco-Latino", "🎯 Factorial Dos Factores",
    "🎯 Factorial Tres Factores", "🔬 Pruebas F2³", "📊 Diseño 2^k",
    "📊 Fraccionado", "📈 MSR", "⚙️ Taguchi", "🎯 Optimización",
    "➕ Aditivos", "📦 Split-Plot"
])

# ============================================================
# TAB INTERACTIVO (WIZARD CON CARGA CSV/URL Y GRÁFICOS)
# ============================================================
with tab_interactivo:
    st.header("🎯 Wizard de Diseño Experimental (Estilo Statgraphics)")
    st.markdown("Crea tu diseño paso a paso, define los factores, y la aplicación analizará automáticamente.")

    if 'wiz_paso' not in st.session_state: st.session_state.wiz_paso = 1
    if 'wiz_tipo_diseno' not in st.session_state: st.session_state.wiz_tipo_diseno = None
    if 'wiz_factores' not in st.session_state: st.session_state.wiz_factores = {}
    if 'wiz_respuesta' not in st.session_state: st.session_state.wiz_respuesta = None
    if 'wiz_datos' not in st.session_state: st.session_state.wiz_datos = None
    if 'wiz_config' not in st.session_state: st.session_state.wiz_config = {}

    st.progress((st.session_state.wiz_paso - 1) / 6)
    st.caption(f"**Paso {st.session_state.wiz_paso} de 7**")

    # PASO 1
    if st.session_state.wiz_paso == 1:
        st.subheader("📋 Paso 1: Tipo de Diseño")
        opciones_diseno = {
            "🔍 Screening (Factorial 2^k)": "screening", "📈 Superficie de Respuesta (MSR)": "rsm",
            "🔢 Factorial Multinivel": "multilevel", "🎯 Un Solo Factor (DCA)": "single_factor",
            "📊 Un Solo Factor con Bloques (DBCA)": "single_factor_block",
            "🔲 Cuadrado Latino (DCL)": "latin_square", "🔳 Cuadrado Greco-Latino (DCGL)": "greco_latin",
        }
        tipo = st.radio("**Design Class:**", list(opciones_diseno.keys()), key="wiz_tipo_radio")
        if st.button("➡️ Siguiente Paso", key="wiz_next_1"):
            st.session_state.wiz_tipo_diseno = opciones_diseno[tipo]
            st.session_state.wiz_paso = 2
            st.rerun()

    # PASO 2
    elif st.session_state.wiz_paso == 2:
        st.subheader("📋 Paso 2: Definición de Factores")
        tipo = st.session_state.wiz_tipo_diseno
        n_factores = st.number_input("**Número de Factores:**", min_value=1, max_value=8, value=1 if tipo in ['single_factor', 'single_factor_block', 'latin_square', 'greco_latin'] else 3, step=1, key="wiz_n_factores")
        factores_def = {}
        for i in range(n_factores):
            with st.expander(f"📌 Factor {chr(65+i)}", expanded=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1: nombre = st.text_input(f"Nombre del Factor {chr(65+i)}:", value=f"Factor_{chr(65+i)}", key=f"wiz_nombre_{i}")
                with col2: n_niveles = st.number_input(f"Niveles:", min_value=2, max_value=10, value=2 if tipo in ['screening', 'rsm'] else 3, step=1, key=f"wiz_niveles_{i}")
                with col3: tipo_var = st.selectbox("Tipo:", ["Continuo", "Categórico"], key=f"wiz_tipo_{i}")
                niveles = []
                if tipo_var == "Continuo" and n_niveles == 2:
                    col_low, col_high = st.columns(2)
                    with col_low: low = st.text_input("Low:", value="-1.0", key=f"wiz_low_{i}")
                    with col_high: high = st.text_input("High:", value="1.0", key=f"wiz_high_{i}")
                    niveles = [low, high]
                else:
                    cols = st.columns(min(n_niveles, 5))
                    for j in range(n_niveles):
                        with cols[j % len(cols)]: niveles.append(st.text_input(f"Nivel {j+1}:", value=str(j+1), key=f"wiz_nivel_{i}_{j}"))
                factores_def[nombre] = {'niveles': niveles, 'tipo': tipo_var, 'n_niveles': n_niveles}
        if tipo == 'single_factor_block':
            with st.expander("📌 Factor de Bloque", expanded=True):
                nombre_bloque = st.text_input("Nombre del Bloque:", value="Bloque", key="wiz_nombre_bloque")
                n_niveles_bloque = st.number_input("Niveles del Bloque:", min_value=2, max_value=10, value=3, step=1, key="wiz_niveles_bloque")
                niveles_bloque = []
                cols = st.columns(min(n_niveles_bloque, 5))
                for j in range(n_niveles_bloque):
                    with cols[j % len(cols)]: niveles_bloque.append(st.text_input(f"Bloque {j+1}:", value=f"B{j+1}", key=f"wiz_bloque_{j}"))
                factores_def[nombre_bloque] = {'niveles': niveles_bloque, 'tipo': 'Categórico', 'n_niveles': n_niveles_bloque, 'es_bloque': True}
        if tipo == 'latin_square':
            with st.expander("📌 Filas y Columnas (Bloqueos)", expanded=True):
                col1, col2 = st.columns(2)
                with col1: n_filas = st.number_input("Filas:", min_value=2, max_value=10, value=3, step=1, key="wiz_n_filas")
                with col2: n_cols = st.number_input("Columnas:", min_value=2, max_value=10, value=3, step=1, key="wiz_n_cols")
        if tipo == 'greco_latin':
            with st.expander("📌 Filas, Columnas y Greco (Bloqueos)", expanded=True):
                col1, col2 = st.columns(2)
                with col1: n_filas = st.number_input("Filas:", min_value=2, max_value=10, value=3, step=1, key="wiz_n_filas_g")
                with col2: n_cols = st.number_input("Columnas:", min_value=2, max_value=10, value=3, step=1, key="wiz_n_cols_g")
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("⬅️ Atrás", key="wiz_back_2"): st.session_state.wiz_paso = 1; st.rerun()
        with col_btn2:
            if st.button("➡️ Siguiente Paso", key="wiz_next_2"):
                st.session_state.wiz_factores = factores_def; st.session_state.wiz_paso = 3; st.rerun()

    # PASO 3
    elif st.session_state.wiz_paso == 3:
        st.subheader("📋 Paso 3: Definición de la Respuesta")
        nombre_resp = st.text_input("**Nombre de la Variable Respuesta:**", value="Respuesta", key="wiz_nombre_resp")
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("⬅️ Atrás", key="wiz_back_3"): st.session_state.wiz_paso = 2; st.rerun()
        with col_btn2:
            if st.button("➡️ Siguiente Paso", key="wiz_next_3"):
                st.session_state.wiz_respuesta = nombre_resp; st.session_state.wiz_paso = 4; st.rerun()

    # PASO 4
    elif st.session_state.wiz_paso == 4:
        st.subheader("📋 Paso 4: Opciones del Diseño")
        col1, col2 = st.columns(2)
        with col1:
            n_replicas = st.number_input("**Réplicas:**", min_value=1, max_value=10, value=1, step=1, key="wiz_replicas")
            puntos_centro = st.number_input("**Puntos Centrales:**", min_value=0, max_value=10, value=0, step=1, key="wiz_puntos_centro")
        with col2:
            randomizar = st.checkbox("**Randomizar el orden de corrida**", value=True, key="wiz_randomizar")
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("⬅️ Atrás", key="wiz_back_4"): st.session_state.wiz_paso = 3; st.rerun()
        with col_btn2:
            if st.button("🎲 Generar Diseño", key="wiz_generar"):
                st.session_state.wiz_config = {'n_replicas': n_replicas, 'puntos_centro': puntos_centro, 'randomizar': randomizar}
                st.session_state.wiz_datos = None; st.session_state.wiz_paso = 5; st.rerun()

    # PASO 5
    elif st.session_state.wiz_paso == 5:
        st.subheader("📋 Paso 5: Tabla del Diseño")
        if st.session_state.wiz_datos is None:
            factores, resp, config, tipo = st.session_state.wiz_factores, st.session_state.wiz_respuesta, st.session_state.wiz_config, st.session_state.wiz_tipo_diseno
            try:
                if tipo in ['screening', 'multilevel', 'rsm']:
                    df_diseno = crear_diseno_completo(factores, {resp: {'valores': None}}, n_replicas=config['n_replicas'], randomize=config['randomizar'])
                elif tipo in ['single_factor', 'single_factor_block']:
                    factor_principal = [f for f, v in factores.items() if not v.get('es_bloque', False)]
                    if factor_principal:
                        f = factor_principal[0]; niveles_f = factores[f]['niveles']; n_bloque, niveles_bloque = None, None
                        if tipo == 'single_factor_block':
                            factores_bloque = [f for f, v in factores.items() if v.get('es_bloque', False)]
                            if factores_bloque: n_bloque, niveles_bloque = factores_bloque[0], factores[factores_bloque[0]]['niveles']
                        filas = []
                        for nivel in niveles_f:
                            if n_bloque:
                                for blq in niveles_bloque: filas.append({f: nivel, n_bloque: blq, resp: None})
                            else:
                                for _ in range(config['n_replicas']): filas.append({f: nivel, resp: None})
                        df_diseno = pd.DataFrame(filas)
                        if config['randomizar']: df_diseno = df_diseno.sample(frac=1, random_state=42).reset_index(drop=True)
                elif tipo in ['latin_square', 'greco_latin']:
                    factor_principal = [f for f, v in factores.items() if not v.get('es_bloque', False)]
                    if factor_principal:
                        f = factor_principal[0]; niveles_f = factores[f]['niveles']; n = len(niveles_f); filas = []
                        for i in range(n):
                            for j in range(n):
                                fila = {'Fila': f'F{i+1}', 'Columna': f'C{j+1}', f: niveles_f[(i + j) % n], resp: None}
                                if tipo == 'greco_latin': fila['Greco'] = chr(65 + (i + 2*j) % n)
                                filas.append(fila)
                        df_diseno = pd.DataFrame(filas)
                else:
                    df_diseno = crear_diseno_completo(factores, {resp: {'valores': None}}, n_replicas=config['n_replicas'], randomize=config['randomizar'])
                if config.get('puntos_centro', 0) > 0:
                    for _ in range(config['puntos_centro']):
                        fila_centro = {f: v['niveles'][0] for f, v in factores.items() if not v.get('es_bloque', False)}
                        fila_centro[resp] = None
                        df_diseno = pd.concat([df_diseno, pd.DataFrame([fila_centro])], ignore_index=True)
                st.session_state.wiz_datos = df_diseno
            except Exception as e: st.error(f"❌ Error al generar el diseño: {str(e)}")
        
        if st.session_state.wiz_datos is not None:
            df_diseno = st.session_state.wiz_datos
            st.success(f"✅ Diseño generado: {len(df_diseno)} corridas")
            st.dataframe(df_diseno, use_container_width=True)
            st.download_button("📥 Descargar Diseño (vacío)", data=df_diseno.to_csv(index=False).encode('utf-8'), file_name="diseno_vacio.csv", mime="text/csv")
            st.subheader("✏️ Editar datos manualmente")
            df_editado = st.data_editor(df_diseno, num_rows="fixed", key="wiz_editor")
            
            st.markdown("---")
            st.subheader("📁 O cargar datos completos desde CSV / URL")
            col_up1, col_up2 = st.columns(2)
            with col_up1:
                archivo = st.file_uploader("Sube tu CSV:", type=['csv', 'txt'], key="wiz_upload_csv")
                if archivo is not None:
                    df_subido = cargar_datos_desde_csv(archivo)
                    if df_subido is not None:
                        if st.button("✅ Usar CSV", key="wiz_usar_csv"):
                            st.session_state.wiz_datos = df_subido
                            st.session_state.wiz_paso = 6
                            st.rerun()
            with col_up2:
                url = st.text_input("O ingresa una URL:", key="wiz_upload_url")
                if url and st.button("✅ Usar URL", key="wiz_usar_url"):
                    df_subido = cargar_datos_desde_url(url)
                    if df_subido is not None:
                        st.session_state.wiz_datos = df_subido
                        st.session_state.wiz_paso = 6
                        st.rerun()

            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("⬅️ Atrás", key="wiz_back_5"): st.session_state.wiz_paso = 4; st.rerun()
            with col_btn2:
                if st.button("➡️ Analizar Datos", key="wiz_analizar"):
                    st.session_state.wiz_datos = df_editado; st.session_state.wiz_paso = 6; st.rerun()

    # PASO 6
    elif st.session_state.wiz_paso == 6:
        st.subheader("📋 Paso 6: Configuración del Análisis")
        df_diseno = st.session_state.wiz_datos
        resp = st.session_state.wiz_respuesta
        if df_diseno is None or len(df_diseno) == 0:
            st.error("❌ No hay datos. Vuelve al paso anterior.")
            if st.button("⬅️ Volver"): st.session_state.wiz_paso = 5; st.rerun()
        elif df_diseno[resp].isnull().all():
            st.warning("⚠️ La columna de respuesta está vacía.")
            if st.button("⬅️ Volver"): st.session_state.wiz_paso = 5; st.rerun()
        else:
            tipo_anova = selector_tipo_anova("wizard")
            st.session_state.wiz_tipo_anova = tipo_anova
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("⬅️ Atrás", key="wiz_back_6"): st.session_state.wiz_paso = 5; st.rerun()
            with col_btn2:
                if st.button("🚀 Ejecutar Análisis", key="wiz_ejecutar"): st.session_state.wiz_paso = 7; st.rerun()

    # PASO 7 (CON PARETO, DANIEL Y MODELO REDUCIDO)
    elif st.session_state.wiz_paso == 7:
        st.subheader("📊 Paso 7: Análisis y Resultados")
        df_diseno, resp, tipo, factores_def = st.session_state.wiz_datos, st.session_state.wiz_respuesta, st.session_state.wiz_tipo_diseno, st.session_state.wiz_factores
        factores_reales = [f for f in factores_def.keys() if f in df_diseno.columns]
        factor_principal = [f for f in factores_reales if not factores_def[f].get('es_bloque', False)]
        df_analisis = df_diseno.copy().reset_index(drop=True)
        try:
            tipo_anova = st.session_state.get('wiz_tipo_anova', 1)
            if tipo in ['screening', 'multilevel', 'rsm']:
                for f in factores_reales: df_analisis[f] = df_analisis[f].astype('category')
                if tipo == 'rsm':
                    vars_list = factor_principal
                    formula = f'{resp} ~ ' + ' + '.join(vars_list) + ' + ' + ' + '.join([f'I({v}**2)' for v in vars_list])
                    if len(vars_list) >= 2: formula += ' + ' + ' + '.join([f'{vars_list[i]}:{vars_list[j]}' for i in range(len(vars_list)) for j in range(i+1, len(vars_list))])
                else: formula = f'{resp} ~ ' + ' * '.join(factor_principal)
                modelo, anova = analizar_anova_mejorada(df_analisis, formula, "ANOVA Wizard", "wizard", "wizard", tipo_anova)
                
                if modelo is not None:
                    mostrar_tabla_anova("wizard")
                    mostrar_summary_statistics(modelo, df_analisis, factor_principal, resp, "wizard", tipo_anova)
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    
                    # ⭐ NUEVO: Gráficos de Efectos e Interacciones + Pareto/Daniel
                    graficar_efectos_e_interacciones(df_analisis, factor_principal, resp)
                    mostrar_pareto_y_daniel(modelo, anova, resp, len(df_analisis))
                    ui_excluir_efectos(df_analisis, resp, anova, tipo_anova, "wizard")
                    
                    # ⭐ MOSTRAR MEJOR ANOVA
                    if 'wizard_red' in st.session_state.resultados_anova:
                        st.markdown("### 🌟 Mejor ANOVA (Modelo Reducido)")
                        mostrar_tabla_anova("wizard_red")
                    
                    mostrar_graficos_residuos_directo(modelo, df_analisis, factor_principal[0], resp, "wizard", "Gráficos de Residuos")
                    if len(factor_principal) >= 1:
                        st.subheader("📊 Boxplots por Factor")
                        fig_box = graficar_boxplot_por_factor(df_analisis, factor_principal, resp)
                        if fig_box: st.plotly_chart(fig_box, use_container_width=True)
                    if len(factor_principal) >= 2:
                        st.subheader("📊 Gráficos de Interacción")
                        fig_int = graficos_interaccion_multifactorial(df_analisis, factor_principal, resp, max_pares=4)
                        if fig_int: st.plotly_chart(fig_int, use_container_width=True)
                        fig_heat = graficos_interaccion_heatmap(df_analisis, factor_principal, resp)
                        if fig_heat: st.plotly_chart(fig_heat, use_container_width=True)
                    st.subheader("🔬 Pruebas Múltiples")
                    mse_c, gl_c = extraer_mse_gl(anova)
                    for factor in factor_principal:
                        n_niveles = len(df_analisis[factor].unique())
                        if n_niveles > 2:
                            st.markdown(f"#### 📌 {factor} ({n_niveles} niveles)")
                            with st.expander(f"Tukey HSD para {factor}", expanded=False):
                                tukey = realizar_tukey(df_analisis, factor, resp)
                                if tukey is not None: st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
                            with st.expander(f"LSD para {factor}", expanded=False):
                                lsd_df = realizar_lsd(df_analisis, factor, resp, mse_c, gl_c)
                                if lsd_df is not None: st.dataframe(lsd_df)
                        else:
                            st.markdown(f"#### 📌 {factor} (2 niveles)")
                            with st.expander(f"LSD para {factor}", expanded=False):
                                lsd_df = realizar_lsd(df_analisis, factor, resp, mse_c, gl_c)
                                if lsd_df is not None: st.dataframe(lsd_df)
                    st.subheader("📝 Conclusiones del Análisis")
                    for c in generar_conclusiones_factorial(anova, modelo, df_analisis, factor_principal, resp):
                        if c.startswith("✅"): st.success(c)
                        elif c.startswith("❌"): st.error(c)
                        else: st.info(c)
            elif tipo == 'single_factor':
                f = factor_principal[0]
                df_analisis[f] = df_analisis[f].astype('category')
                modelo, anova = analizar_anova_mejorada(df_analisis, f'{resp} ~ C({f})', "DCA Wizard", "wizard", "wizard", tipo_anova)
                if modelo is not None:
                    mostrar_tabla_anova("wizard")
                    mostrar_summary_statistics(modelo, df_analisis, [f], resp, "wizard")
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    mostrar_graficos_residuos_directo(modelo, df_analisis, f, resp, "wizard", "Gráficos de Residuos")
                    fig = px.box(df_analisis, x=f, y=resp, color=f); fig.update_layout(showlegend=False, height=350); st.plotly_chart(fig, use_container_width=True)
                    medias = df_analisis.groupby(f)[resp].mean().reset_index()
                    fig2 = px.bar(medias, x=f, y=resp, color=f, text_auto=True); fig2.update_layout(showlegend=False, height=300); st.plotly_chart(fig2, use_container_width=True)
                    st.subheader("🔬 Pruebas Múltiples")
                    mse_c, gl_c = extraer_mse_gl(anova)
                    with st.expander("Tukey HSD", expanded=False):
                        tukey = realizar_tukey(df_analisis, f, resp)
                        if tukey is not None: st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
                    with st.expander("LSD", expanded=False):
                        lsd_df = realizar_lsd(df_analisis, f, resp, mse_c, gl_c)
                        if lsd_df is not None: st.dataframe(lsd_df)
                    st.subheader("📝 Conclusiones del Análisis")
                    for c in generar_conclusiones_anova(anova, modelo, df_analisis, f, resp)[0]:
                        if c.startswith("✅") or c.startswith("🏆") or c.startswith("📈"): st.success(c)
                        elif c.startswith("❌"): st.error(c)
                        else: st.info(c)
            elif tipo == 'single_factor_block':
                f = factor_principal[0]; factor_bloque = [f for f in factores_reales if factores_def[f].get('es_bloque', False)]
                df_analisis[f] = df_analisis[f].astype('category')
                if factor_bloque:
                    blq = factor_bloque[0]; df_analisis[blq] = df_analisis[blq].astype('category'); formula = f'{resp} ~ C({f}) + C({blq})'
                else: formula = f'{resp} ~ C({f})'
                modelo, anova = analizar_anova_mejorada(df_analisis, formula, "DBCA Wizard", "wizard", "wizard", tipo_anova)
                if modelo is not None:
                    mostrar_tabla_anova("wizard")
                    mostrar_summary_statistics(modelo, df_analisis, [f] + (factor_bloque if factor_bloque else []), resp, "wizard")
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    mostrar_graficos_residuos_directo(modelo, df_analisis, f, resp, "wizard", "Gráficos de Residuos")
                    fig = px.box(df_analisis, x=f, y=resp, color=f); fig.update_layout(showlegend=False, height=350); st.plotly_chart(fig, use_container_width=True)
                    if factor_bloque:
                        fig_blq = px.box(df_analisis, x=blq, y=resp, color=blq); fig_blq.update_layout(showlegend=False, height=350); st.plotly_chart(fig_blq, use_container_width=True)
                    st.subheader("🔬 Pruebas Múltiples")
                    mse_c, gl_c = extraer_mse_gl(anova)
                    with st.expander("Tukey HSD", expanded=False):
                        tukey = realizar_tukey(df_analisis, f, resp)
                        if tukey is not None: st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
                    with st.expander("LSD", expanded=False):
                        lsd_df = realizar_lsd(df_analisis, f, resp, mse_c, gl_c)
                        if lsd_df is not None: st.dataframe(lsd_df)
                    st.subheader("📝 Conclusiones del Análisis")
                    for c in generar_conclusiones_anova(anova, modelo, df_analisis, f, resp)[0]:
                        if c.startswith("✅") or c.startswith("🏆") or c.startswith("📈"): st.success(c)
                        elif c.startswith("❌"): st.error(c)
                        else: st.info(c)
            elif tipo in ['latin_square', 'greco_latin']:
                f = factor_principal[0]
                for col in ['Fila', 'Columna', f]:
                    if col in df_analisis.columns: df_analisis[col] = df_analisis[col].astype('category')
                if tipo == 'latin_square': formula = f'{resp} ~ C(Fila) + C(Columna) + C({f})'
                else:
                    if 'Greco' in df_analisis.columns:
                        df_analisis['Greco'] = df_analisis['Greco'].astype('category')
                        formula = f'{resp} ~ C(Fila) + C(Columna) + C({f}) + C(Greco)'
                    else: formula = f'{resp} ~ C(Fila) + C(Columna) + C({f})'
                modelo, anova = analizar_anova_mejorada(df_analisis, formula, "CL/GL Wizard", "wizard", "wizard", tipo_anova)
                if modelo is not None:
                    mostrar_tabla_anova("wizard")
                    factores_ss = ['Fila', 'Columna', f] + (['Greco'] if tipo == 'greco_latin' and 'Greco' in df_analisis.columns else [])
                    mostrar_summary_statistics(modelo, df_analisis, factores_ss, resp, "wizard")
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    mostrar_graficos_residuos_directo(modelo, df_analisis, f, resp, "wizard", "Gráficos de Residuos")
                    fig = px.box(df_analisis, x=f, y=resp, color=f); fig.update_layout(showlegend=False, height=350); st.plotly_chart(fig, use_container_width=True)
                    if 'Fila' in df_analisis.columns and 'Columna' in df_analisis.columns:
                        pivot = df_analisis.pivot_table(index='Fila', columns='Columna', values=resp, aggfunc='mean')
                        fig_heat = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='Blues'); st.plotly_chart(fig_heat, use_container_width=True)
                    st.subheader("🔬 Pruebas Múltiples")
                    mse_c, gl_c = extraer_mse_gl(anova)
                    with st.expander(f"Tukey HSD para {f}", expanded=False):
                        tukey = realizar_tukey(df_analisis, f, resp)
                        if tukey is not None: st.dataframe(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
                    with st.expander(f"LSD para {f}", expanded=False):
                        lsd_df = realizar_lsd(df_analisis, f, resp, mse_c, gl_c)
                        if lsd_df is not None: st.dataframe(lsd_df)
                    if tipo == 'greco_latin' and 'Greco' in df_analisis.columns:
                        with st.expander("Tukey HSD para Greco", expanded=False):
                            tukey_g = realizar_tukey(df_analisis, 'Greco', resp)
                            if tukey_g is not None: st.dataframe(pd.DataFrame(data=tukey_g.summary().data[1:], columns=tukey_g.summary().data[0]))
                        with st.expander("LSD para Greco", expanded=False):
                            lsd_g = realizar_lsd(df_analisis, 'Greco', resp, mse_c, gl_c)
                            if lsd_g is not None: st.dataframe(lsd_g)
                    st.subheader("📝 Conclusiones del Análisis")
                    for c in generar_conclusiones_anova(anova, modelo, df_analisis, f, resp)[0]:
                        if c.startswith("✅") or c.startswith("🏆") or c.startswith("📈"): st.success(c)
                        elif c.startswith("❌"): st.error(c)
                        else: st.info(c)
        except Exception as e:
            st.error(f"❌ Error durante el análisis: {str(e)}")
            st.exception(e)
        st.markdown("---")
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("⬅️ Volver al Paso 6", key="wiz_back_7"): st.session_state.wiz_paso = 6; st.rerun()
        with col_btn2:
            if st.button("🔄 Nuevo Diseño", key="wiz_nuevo"):
                st.session_state.wiz_paso = 1; st.session_state.wiz_tipo_diseno = None; st.session_state.wiz_factores = {}
                st.session_state.wiz_respuesta = None; st.session_state.wiz_datos = None; st.session_state.wiz_config = {}
                st.rerun()

# ============================================================
# PESTAÑAS 1 A 7 Y 9 A 13 (RESTAURADAS EXACTAMENTE IGUAL)
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Completamente Aleatorizado")
        df_temp = obtener_datos_generales("dca", 2)
        if df_temp is not None and len(df_temp.columns) >= 2:
            tipo_anova = selector_tipo_anova("dca")
            col_factor = st.selectbox("Selecciona la columna del factor:", df_temp.columns.tolist(), key="factor_dca")
            col_respuesta = st.selectbox("Selecciona la columna de respuesta:", [col for col in df_temp.columns if col != col_factor], key="respuesta_dca")
            if st.button("🔍 Calcular ANOVA", key="dca_btn"):
                df = df_temp.copy()
                df.columns = ['Tratamiento' if col == col_factor else 'Valor' if col == col_respuesta else col for col in df.columns]
                df = df[['Tratamiento', 'Valor']]; df['Tratamiento'] = df['Tratamiento'].astype('category')
                modelo, anova = analizar_anova_mejorada(df, 'Valor ~ C(Tratamiento)', "ANOVA Un Factor", "dca", "dca", tipo_anova)
                if modelo is not None:
                    st.session_state.pruebas_multiples['dca'] = {'df': df, 'modelo': modelo, 'anova': anova}
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    mostrar_graficos_residuos_directo(modelo, df, 'Tratamiento', 'Valor', "dca", "Gráficos de Residuos - DCA")
    with col2:
        if 'dca' in st.session_state.resultados_anova:
            res = st.session_state.resultados_anova['dca']
            df, modelo, anova = res['df'], res['modelo'], res['anova']
            mostrar_summary_statistics(modelo, df, ['Tratamiento'], 'Valor', "dca")
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento', title="Distribución por Tratamiento"); fig.update_layout(showlegend=False, height=350); st.plotly_chart(fig, use_container_width=True)
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor', title="Medias por Tratamiento", color='Tratamiento', text_auto=True); fig2.update_layout(showlegend=False, height=300); fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside'); st.plotly_chart(fig2, use_container_width=True)
            st.subheader("🔬 Pruebas Múltiples")
            with st.form(key="form_pruebas_dca"):
                metodo = st.radio("Selecciona el método:", ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"], horizontal=True, key="metodo_dca_form")
                submitted = st.form_submit_button("Calcular Prueba")
                if submitted:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    if metodo == "Tukey HSD":
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None: st.session_state.pruebas_multiples['dca_resultado'] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, 'Tratamiento', 'Valor', mse_c, gl_c)
                        if lsd_df is not None: st.session_state.pruebas_multiples['dca_resultado'] = ('lsd', lsd_df, mse_c, gl_c)
            if 'dca_resultado' in st.session_state.pruebas_multiples:
                res = st.session_state.pruebas_multiples['dca_resultado']
                if res[0] == 'tukey':
                    st.dataframe(pd.DataFrame(data=res[1].summary().data[1:], columns=res[1].summary().data[0]))
                    for l in interpretar_tukey(res[1]): st.write(l)
                else:
                    st.write(f"**MSE = {res[2]:.4f}, GL Error = {int(res[3])}**"); st.dataframe(res[1])
                    for l in interpretar_lsd(res[1]): st.write(l)
            st.subheader("📝 Conclusiones del Análisis")
            for c in generar_conclusiones_anova(anova, modelo, df, 'Tratamiento', 'Valor')[0]:
                if c.startswith("✅") or c.startswith("🏆") or c.startswith("📈"): st.success(c)
                elif c.startswith("❌"): st.error(c)
                else: st.info(c)
            if st.button("🗑️ Limpiar resultados", key="limpiar_dca"):
                if 'dca' in st.session_state.resultados_anova: del st.session_state.resultados_anova['dca']
                if 'dca_resultado' in st.session_state.pruebas_multiples: del st.session_state.pruebas_multiples['dca_resultado']
                st.rerun()

with tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño en Bloques")
        df_temp = obtener_datos_generales("dbca", 3)
        if df_temp is not None and len(df_temp.columns) >= 3:
            tipo_anova = selector_tipo_anova("dbca")
            col_factor = st.selectbox("Selecciona la columna del factor:", df_temp.columns.tolist(), key="factor_dbca")
            col_bloque = st.selectbox("Selecciona la columna del bloque:", [col for col in df_temp.columns if col != col_factor], key="bloque_dbca")
            col_respuesta = st.selectbox("Selecciona la columna de respuesta:", [col for col in df_temp.columns if col not in [col_factor, col_bloque]], key="respuesta_dbca")
            if st.button("🔍 Calcular", key="dbca_btn"):
                df = df_temp.copy()
                df.columns = ['Tratamiento' if col == col_factor else 'Bloque' if col == col_bloque else 'Valor' if col == col_respuesta else col for col in df.columns]
                df = df[['Tratamiento', 'Bloque', 'Valor']]; df['Tratamiento'] = df['Tratamiento'].astype('category'); df['Bloque'] = df['Bloque'].astype('category')
                modelo, anova = analizar_anova_mejorada(df, 'Valor ~ C(Tratamiento) + C(Bloque)', "ANOVA en Bloques", "dbca", "dbca", tipo_anova)
                if modelo is not None:
                    st.session_state.pruebas_multiples['dbca'] = {'df': df, 'modelo': modelo, 'anova': anova}
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    mostrar_graficos_residuos_directo(modelo, df, 'Tratamiento', 'Valor', "dbca", "Gráficos de Residuos - DBCA")
    with col2:
        if 'dbca' in st.session_state.resultados_anova:
            res = st.session_state.resultados_anova['dbca']
            df, modelo, anova = res['df'], res['modelo'], res['anova']
            mostrar_summary_statistics(modelo, df, ['Tratamiento', 'Bloque'], 'Valor', "dbca")
            fig_box_trat = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento', title="Distribución por Tratamiento"); fig_box_trat.update_layout(showlegend=False, height=350); st.plotly_chart(fig_box_trat, use_container_width=True)
            fig_box_bloque = px.box(df, x='Bloque', y='Valor', color='Bloque', title="Distribución por Bloque"); fig_box_bloque.update_layout(showlegend=False, height=350); st.plotly_chart(fig_box_bloque, use_container_width=True)
            fig_facet = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento', facet_col='Bloque'); fig_facet.update_layout(height=350); st.plotly_chart(fig_facet, use_container_width=True)
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor', title="Medias por Tratamiento", color='Tratamiento', text_auto=True); fig2.update_layout(showlegend=False, height=300); fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside'); st.plotly_chart(fig2, use_container_width=True)
            st.subheader("🔬 Pruebas Múltiples para Tratamientos")
            with st.form(key="form_pruebas_dbca"):
                metodo = st.radio("Selecciona el método:", ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"], horizontal=True, key="metodo_dbca_form")
                submitted = st.form_submit_button("Calcular Prueba")
                if submitted:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    if metodo == "Tukey HSD":
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None: st.session_state.pruebas_multiples['dbca_resultado'] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, 'Tratamiento', 'Valor', mse_c, gl_c)
                        if lsd_df is not None: st.session_state.pruebas_multiples['dbca_resultado'] = ('lsd', lsd_df, mse_c, gl_c)
            if 'dbca_resultado' in st.session_state.pruebas_multiples:
                res = st.session_state.pruebas_multiples['dbca_resultado']
                if res[0] == 'tukey':
                    st.dataframe(pd.DataFrame(data=res[1].summary().data[1:], columns=res[1].summary().data[0]))
                    for l in interpretar_tukey(res[1]): st.write(l)
                else:
                    st.write(f"**MSE = {res[2]:.4f}, GL Error = {int(res[3])}**"); st.dataframe(res[1])
                    for l in interpretar_lsd(res[1]): st.write(l)
            st.subheader("📝 Conclusiones del Análisis")
            for c in generar_conclusiones_anova(anova, modelo, df, 'Tratamiento', 'Valor')[0]:
                if c.startswith("✅") or c.startswith("🏆") or c.startswith("📈"): st.success(c)
                elif c.startswith("❌"): st.error(c)
                else: st.info(c)
            if st.button("🗑️ Limpiar resultados", key="limpiar_dbca"):
                if 'dbca' in st.session_state.resultados_anova: del st.session_state.resultados_anova['dbca']
                if 'dbca_resultado' in st.session_state.pruebas_multiples: del st.session_state.pruebas_multiples['dbca_resultado']
                st.rerun()

with tab3:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Cuadrado Latino")
        df_temp = obtener_datos_generales("cl", 4)
        if df_temp is not None and len(df_temp.columns) >= 4:
            tipo_anova = selector_tipo_anova("cl")
            col_fila = st.selectbox("Columna de fila:", df_temp.columns.tolist(), key="fila_cl")
            cols_restantes = [c for c in df_temp.columns if c != col_fila]
            col_columna = st.selectbox("Columna de columna:", cols_restantes, key="columna_cl")
            cols_restantes = [c for c in cols_restantes if c != col_columna]
            col_tratamiento = st.selectbox("Columna de tratamiento:", cols_restantes, key="tratamiento_cl")
            cols_restantes = [c for c in cols_restantes if c != col_tratamiento]
            col_respuesta = st.selectbox("Columna de respuesta:", cols_restantes, key="respuesta_cl")
            if st.button("🔍 Calcular", key="cl_btn"):
                df = df_temp.copy()
                df.columns = ['Fila' if col == col_fila else 'Columna' if col == col_columna else 'Tratamiento' if col == col_tratamiento else 'Valor' if col == col_respuesta else col for col in df.columns]
                df = df[['Fila', 'Columna', 'Tratamiento', 'Valor']]
                for col in ['Fila', 'Columna', 'Tratamiento']: df[col] = df[col].astype('category')
                modelo, anova = analizar_anova_mejorada(df, 'Valor ~ C(Fila) + C(Columna) + C(Tratamiento)', "Cuadrado Latino", "cl", "cl", tipo_anova)
                if modelo is not None:
                    st.session_state.pruebas_multiples['cl'] = {'df': df, 'modelo': modelo, 'anova': anova}
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    mostrar_graficos_residuos_directo(modelo, df, 'Tratamiento', 'Valor', "cl", "Gráficos de Residuos - Cuadrado Latino")
    with col2:
        if 'cl' in st.session_state.resultados_anova:
            res = st.session_state.resultados_anova['cl']
            df, modelo, anova = res['df'], res['modelo'], res['anova']
            mostrar_summary_statistics(modelo, df, ['Fila', 'Columna', 'Tratamiento'], 'Valor', "cl")
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento', title="Distribución por Tratamiento"); fig.update_layout(showlegend=False, height=350); st.plotly_chart(fig, use_container_width=True)
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor', title="Medias por Tratamiento", color='Tratamiento', text_auto=True); fig2.update_layout(showlegend=False, height=300); fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside'); st.plotly_chart(fig2, use_container_width=True)
            pivot = df.pivot(index='Fila', columns='Columna', values='Valor')
            fig3 = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='Blues', title="Mapa de Calor: Fila × Columna"); fig3.update_layout(height=350); st.plotly_chart(fig3, use_container_width=True)
            st.subheader("🔬 Pruebas Múltiples para Tratamientos")
            with st.form(key="form_pruebas_cl"):
                metodo = st.radio("Selecciona el método:", ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"], horizontal=True, key="metodo_cl_form")
                submitted = st.form_submit_button("Calcular Prueba")
                if submitted:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    if metodo == "Tukey HSD":
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None: st.session_state.pruebas_multiples['cl_resultado'] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, 'Tratamiento', 'Valor', mse_c, gl_c)
                        if lsd_df is not None: st.session_state.pruebas_multiples['cl_resultado'] = ('lsd', lsd_df, mse_c, gl_c)
            if 'cl_resultado' in st.session_state.pruebas_multiples:
                res = st.session_state.pruebas_multiples['cl_resultado']
                if res[0] == 'tukey':
                    st.dataframe(pd.DataFrame(data=res[1].summary().data[1:], columns=res[1].summary().data[0]))
                    for l in interpretar_tukey(res[1]): st.write(l)
                else:
                    st.write(f"**MSE = {res[2]:.4f}, GL Error = {int(res[3])}**"); st.dataframe(res[1])
                    for l in interpretar_lsd(res[1]): st.write(l)
            st.subheader("📝 Conclusiones del Análisis")
            for c in generar_conclusiones_anova(anova, modelo, df, 'Tratamiento', 'Valor')[0]:
                if c.startswith("✅") or c.startswith("🏆") or c.startswith("📈"): st.success(c)
                elif c.startswith("❌"): st.error(c)
                else: st.info(c)
            if st.button("🗑️ Limpiar resultados", key="limpiar_cl"):
                if 'cl' in st.session_state.resultados_anova: del st.session_state.resultados_anova['cl']
                if 'cl_resultado' in st.session_state.pruebas_multiples: del st.session_state.pruebas_multiples['cl_resultado']
                st.rerun()

with tab4:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Cuadrado Greco-Latino")
        df_temp = obtener_datos_generales("cgl", 5)
        if df_temp is not None and len(df_temp.columns) >= 5:
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
            col_respuesta = st.selectbox("Columna de respuesta:", cols_restantes, key="respuesta_cgl")
            if st.button("🔍 Calcular", key="cgl_btn"):
                df = df_temp.copy()
                df.columns = ['Fila' if col == col_fila else 'Columna' if col == col_columna else 'Tratamiento' if col == col_tratamiento else 'Greco' if col == col_greco else 'Valor' if col == col_respuesta else col for col in df.columns]
                df = df[['Fila', 'Columna', 'Tratamiento', 'Greco', 'Valor']]
                for col in ['Fila', 'Columna', 'Tratamiento', 'Greco']: df[col] = df[col].astype('category')
                modelo, anova = analizar_anova_mejorada(df, 'Valor ~ C(Fila) + C(Columna) + C(Tratamiento) + C(Greco)', "Greco-Latino", "cgl", "cgl", tipo_anova)
                if modelo is not None:
                    st.session_state.pruebas_multiples['cgl'] = {'df': df, 'modelo': modelo, 'anova': anova}
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    mostrar_graficos_residuos_directo(modelo, df, 'Tratamiento', 'Valor', "cgl", "Gráficos de Residuos - Greco-Latino")
    with col2:
        if 'cgl' in st.session_state.resultados_anova:
            res = st.session_state.resultados_anova['cgl']
            df, modelo, anova = res['df'], res['modelo'], res['anova']
            mostrar_summary_statistics(modelo, df, ['Fila', 'Columna', 'Tratamiento', 'Greco'], 'Valor', "cgl")
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento'); fig.update_layout(showlegend=False, height=350); st.plotly_chart(fig, use_container_width=True)
            medias = df.groupby('Tratamiento')['Valor'].mean().reset_index()
            fig2 = px.bar(medias, x='Tratamiento', y='Valor', title="Medias por Tratamiento", color='Tratamiento', text_auto=True); fig2.update_layout(showlegend=False, height=300); fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside'); st.plotly_chart(fig2, use_container_width=True)
            pivot = df.pivot(index='Fila', columns='Columna', values='Valor')
            fig3 = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='Greens', title="Mapa de Calor: Fila × Columna"); fig3.update_layout(height=350); st.plotly_chart(fig3, use_container_width=True)
            try:
                pivot_tg = df.pivot_table(index='Tratamiento', columns='Greco', values='Valor', aggfunc='mean')
                fig4 = px.imshow(pivot_tg, text_auto=True, aspect="auto", color_continuous_scale='Blues', title="Mapa de Calor: Tratamiento × Greco"); fig4.update_layout(height=350); st.plotly_chart(fig4, use_container_width=True)
            except: pass
            st.subheader("🔬 Pruebas Múltiples - Tratamiento")
            with st.form(key="form_pruebas_cgl_trat"):
                metodo_trat = st.radio("Método:", ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"], horizontal=True, key="metodo_cgl_trat_form")
                submitted_trat = st.form_submit_button("Calcular Prueba para Tratamiento")
                if submitted_trat:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    if metodo_trat == "Tukey HSD":
                        tukey = realizar_tukey(df, 'Tratamiento', 'Valor')
                        if tukey is not None: st.session_state.pruebas_multiples['cgl_trat_resultado'] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, 'Tratamiento', 'Valor', mse_c, gl_c)
                        if lsd_df is not None: st.session_state.pruebas_multiples['cgl_trat_resultado'] = ('lsd', lsd_df, mse_c, gl_c)
            if 'cgl_trat_resultado' in st.session_state.pruebas_multiples:
                res = st.session_state.pruebas_multiples['cgl_trat_resultado']
                if res[0] == 'tukey':
                    st.dataframe(pd.DataFrame(data=res[1].summary().data[1:], columns=res[1].summary().data[0]))
                    for l in interpretar_tukey(res[1]): st.write(l)
                else:
                    st.write(f"**MSE = {res[2]:.4f}, GL Error = {int(res[3])}**"); st.dataframe(res[1])
                    for l in interpretar_lsd(res[1]): st.write(l)
            st.subheader("🔬 Pruebas Múltiples - Greco")
            with st.form(key="form_pruebas_cgl_greco"):
                metodo_greco = st.radio("Método:", ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"], horizontal=True, key="metodo_cgl_greco_form")
                submitted_greco = st.form_submit_button("Calcular Prueba para Greco")
                if submitted_greco:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    if metodo_greco == "Tukey HSD":
                        tukey = realizar_tukey(df, 'Greco', 'Valor')
                        if tukey is not None: st.session_state.pruebas_multiples['cgl_greco_resultado'] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, 'Greco', 'Valor', mse_c, gl_c)
                        if lsd_df is not None: st.session_state.pruebas_multiples['cgl_greco_resultado'] = ('lsd', lsd_df, mse_c, gl_c)
            if 'cgl_greco_resultado' in st.session_state.pruebas_multiples:
                res = st.session_state.pruebas_multiples['cgl_greco_resultado']
                if res[0] == 'tukey':
                    st.dataframe(pd.DataFrame(data=res[1].summary().data[1:], columns=res[1].summary().data[0]))
                    for l in interpretar_tukey(res[1]): st.write(l)
                else:
                    st.write(f"**MSE = {res[2]:.4f}, GL Error = {int(res[3])}**"); st.dataframe(res[1])
                    for l in interpretar_lsd(res[1]): st.write(l)
            st.subheader("📝 Conclusiones del Análisis")
            for c in generar_conclusiones_anova(anova, modelo, df, 'Tratamiento', 'Valor')[0]:
                if c.startswith("✅") or c.startswith("🏆") or c.startswith("📈"): st.success(c)
                elif c.startswith("❌"): st.error(c)
                else: st.info(c)
            if st.button("🗑️ Limpiar resultados", key="limpiar_cgl"):
                for key in ['cgl', 'cgl_trat_resultado', 'cgl_greco_resultado']:
                    if key in st.session_state.resultados_anova: del st.session_state.resultados_anova[key]
                    if key in st.session_state.pruebas_multiples: del st.session_state.pruebas_multiples[key]
                st.rerun()

with tab5:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Factorial 2²")
        df_temp = obtener_datos_generales("f22", 3)
        if df_temp is not None and len(df_temp.columns) >= 3:
            tipo_anova = selector_tipo_anova("f22")
            col_a = st.selectbox("Factor A:", df_temp.columns.tolist(), key="a_f22")
            cols_restantes = [c for c in df_temp.columns if c != col_a]
            col_b = st.selectbox("Factor B:", cols_restantes, key="b_f22")
            cols_restantes = [c for c in cols_restantes if c != col_b]
            col_respuesta = st.selectbox("Respuesta (Y):", cols_restantes, key="respuesta_f22")
            if st.button("🔍 Calcular", key="f22_btn"):
                df = df_temp.copy()
                df.columns = ['A' if col == col_a else 'B' if col == col_b else 'Y' if col == col_respuesta else col for col in df.columns]
                df = df[['A', 'B', 'Y']]; df['A'] = df['A'].astype('category'); df['B'] = df['B'].astype('category')
                modelo, anova = analizar_anova_mejorada(df, 'Y ~ A * B', "Factorial 2²", "f22", "f22", tipo_anova)
                if modelo is not None:
                    st.session_state.pruebas_multiples['f22'] = {'df': df, 'modelo': modelo, 'anova': anova}
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    mostrar_graficos_residuos_directo(modelo, df, 'A', 'Y', "f22", "Gráficos de Residuos - Factorial 2²")
    with col2:
        if 'f22' in st.session_state.resultados_anova:
            res = st.session_state.resultados_anova['f22']
            df, modelo, anova = res['df'], res['modelo'], res['anova']
            mostrar_summary_statistics(modelo, df, ['A', 'B'], 'Y', "f22")
            fig_box = graficar_boxplot_por_factor(df, ['A', 'B'], 'Y')
            if fig_box: st.plotly_chart(fig_box, use_container_width=True)
            fig_inter = graficar_interaccion_mejorada(df, 'A', 'B', 'Y')
            if fig_inter: st.plotly_chart(fig_inter, use_container_width=True)
            df['Combinacion'] = df['A'].astype(str) + ',' + df['B'].astype(str)
            fig1 = px.box(df, x='Combinacion', y='Y', color='Combinacion', title="Distribución por Combinación"); fig1.update_layout(showlegend=False, height=400); st.plotly_chart(fig1, use_container_width=True)
            medias = df.groupby(['A', 'B'])['Y'].mean().reset_index()
            fig3 = px.bar(medias, x='A', y='Y', color='B', title="Medias por Combinación", barmode='group', text_auto=True); fig3.update_layout(height=350); fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside'); st.plotly_chart(fig3, use_container_width=True)
            st.subheader("🔬 Pruebas Múltiples")
            with st.form(key="form_pruebas_f22"):
                metodo = st.radio("Método:", ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"], horizontal=True, key="metodo_f22_form")
                factores_prueba = st.multiselect("Factores para probar:", ['A', 'B'], default=['A', 'B'], key="factores_f22_form")
                submitted = st.form_submit_button("Calcular Pruebas")
                if submitted:
                    mse_c, gl_c = extraer_mse_gl(anova)
                    resultados_multiples = {}
                    for factor in factores_prueba:
                        n_niveles = len(df[factor].unique())
                        if metodo == "Tukey HSD":
                            if n_niveles > 2:
                                tukey = realizar_tukey(df, factor, 'Y')
                                if tukey is not None: resultados_multiples[factor] = ('tukey', tukey)
                            else: st.info(f"ℹ️ **{factor}:** Tukey no aplica. Usa LSD.")
                        else:
                            lsd_df = realizar_lsd(df, factor, 'Y', mse_c, gl_c)
                            if lsd_df is not None: resultados_multiples[factor] = ('lsd', lsd_df, mse_c, gl_c)
                    st.session_state.pruebas_multiples['f22_resultado'] = resultados_multiples
            if 'f22_resultado' in st.session_state.pruebas_multiples:
                for factor, r in st.session_state.pruebas_multiples['f22_resultado'].items():
                    st.markdown(f"#### 📌 Resultados para Factor {factor}")
                    if r[0] == 'tukey':
                        st.dataframe(pd.DataFrame(data=r[1].summary().data[1:], columns=r[1].summary().data[0]))
                        for l in interpretar_tukey(r[1]): st.write(l)
                    else:
                        st.write(f"**MSE = {r[2]:.4f}, GL Error = {int(r[3])}**"); st.dataframe(r[1])
                        for l in interpretar_lsd(r[1]): st.write(l)
            st.subheader("📝 Conclusiones del Análisis")
            for c in generar_conclusiones_factorial(anova, modelo, df, ['A', 'B'], 'Y'):
                if c.startswith("✅") or c.startswith("📈"): st.success(c)
                elif c.startswith("❌"): st.error(c)
                else: st.info(c)
            if st.button("🗑️ Limpiar resultados", key="limpiar_f22"):
                if 'f22' in st.session_state.resultados_anova: del st.session_state.resultados_anova['f22']
                if 'f22_resultado' in st.session_state.pruebas_multiples: del st.session_state.pruebas_multiples['f22_resultado']
                st.rerun()

with tab6:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Factorial 2³")
        df_temp = obtener_datos_generales("f23", 4)
        if df_temp is not None and len(df_temp.columns) >= 4:
            tipo_anova = selector_tipo_anova("f23")
            col_a = st.selectbox("Factor A:", df_temp.columns.tolist(), key="a_f23")
            cols_restantes = [c for c in df_temp.columns if c != col_a]
            col_b = st.selectbox("Factor B:", cols_restantes, key="b_f23")
            cols_restantes = [c for c in cols_restantes if c != col_b]
            col_c = st.selectbox("Factor C:", cols_restantes, key="c_f23")
            cols_restantes = [c for c in cols_restantes if c != col_c]
            col_respuesta = st.selectbox("Respuesta (Y):", cols_restantes, key="respuesta_f23")
            if st.button("🔍 Calcular", key="f23_btn"):
                df = df_temp.copy()
                df.columns = ['A' if col == col_a else 'B' if col == col_b else 'C' if col == col_c else 'Y' if col == col_respuesta else col for col in df.columns]
                df = df[['A', 'B', 'C', 'Y']]
                for col in ['A', 'B', 'C']: df[col] = df[col].astype('category')
                modelo, anova = analizar_anova_mejorada(df, 'Y ~ A * B * C', "Factorial 2³", "f23", "f23", tipo_anova)
                if modelo is not None:
                    st.session_state.pruebas_multiples['f23'] = {'df': df, 'modelo': modelo, 'anova': anova}
                    fig_sc = graficar_sumas_cuadrados(anova)
                    if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                    st.subheader("📊 Gráficos de Interacción")
                    factores_list = ['A', 'B', 'C']
                    fig_interaccion = graficos_interaccion_multifactorial(df, factores_list, 'Y', max_pares=3)
                    if fig_interaccion is not None: st.plotly_chart(fig_interaccion, use_container_width=True)
                    st.subheader("📊 Mapa de Calor de Interacciones")
                    fig_heatmap = graficos_interaccion_heatmap(df, factores_list, 'Y')
                    if fig_heatmap is not None: st.plotly_chart(fig_heatmap, use_container_width=True)
                    mostrar_graficos_residuos_directo(modelo, df, 'A', 'Y', "f23", "Gráficos de Residuos - Factorial 2³")
    with col2:
        if 'f23' in st.session_state.resultados_anova:
            res = st.session_state.resultados_anova['f23']
            df, modelo, anova = res['df'], res['modelo'], res['anova']
            mostrar_summary_statistics(modelo, df, ['A', 'B', 'C'], 'Y', "f23")
            fig_box = graficar_boxplot_por_factor(df, ['A', 'B', 'C'], 'Y')
            if fig_box: st.plotly_chart(fig_box, use_container_width=True)
            fig_inter = graficar_interaccion_mejorada(df, 'A', 'B', 'Y')
            if fig_inter: st.plotly_chart(fig_inter, use_container_width=True)
            df['Combinacion'] = df['A'].astype(str) + ',' + df['B'].astype(str) + ',' + df['C'].astype(str)
            fig1 = px.box(df, x='Combinacion', y='Y', color='Combinacion', title="Distribución por Combinación"); fig1.update_layout(showlegend=False, height=350); st.plotly_chart(fig1, use_container_width=True)
            medias = df.groupby(['A', 'B', 'C'])['Y'].mean().reset_index()
            fig3 = px.bar(medias, x='A', y='Y', color='B', facet_col='C', title="Medias por Combinación", barmode='group', text_auto=True); fig3.update_layout(height=350); fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside'); st.plotly_chart(fig3, use_container_width=True)
            st.subheader("📝 Conclusiones del Análisis")
            for c in generar_conclusiones_factorial(anova, modelo, df, ['A', 'B', 'C'], 'Y'):
                if c.startswith("✅"): st.success(c)
                elif c.startswith("❌"): st.error(c)
                else: st.info(c)
            if st.button("🗑️ Limpiar resultados", key="limpiar_f23"):
                for key in ['f23', 'f23_resultado_tab']:
                    if key in st.session_state.resultados_anova: del st.session_state.resultados_anova[key]
                    if key in st.session_state.pruebas_multiples: del st.session_state.pruebas_multiples[key]
                st.rerun()

with tab6_resultados:
    st.header("🔬 Pruebas Múltiples - Factorial 2³")
    if 'f23' in st.session_state.resultados_anova:
        res = st.session_state.resultados_anova['f23']
        df, anova, modelo = res['df'], res['anova'], res['modelo']
        mse_c, gl_c = extraer_mse_gl(anova)
        st.info(f"📊 **MSE del modelo completo:** {mse_c:.4f} | **GL Error:** {int(gl_c)}")
        with st.expander("📋 Summary Statistics — Least Squares Means", expanded=False):
            mostrar_summary_statistics(modelo, df, ['A', 'B', 'C'], 'Y', "f23_resultados")
        st.markdown("---")
        st.subheader("⚙️ Configuración")
        metodo = st.radio("Método:", ["Tukey HSD", "LSD (Diferencia Mínima Significativa)"], horizontal=True, key="met_f23_final")
        col1, col2, col3 = st.columns(3)
        with col1: usar_a = st.checkbox("Factor A", value=True, key="chk_a_f23_final")
        with col2: usar_b = st.checkbox("Factor B", value=True, key="chk_b_f23_final")
        with col3: usar_c = st.checkbox("Factor C", value=True, key="chk_c_f23_final")
        factores_prueba = []
        if usar_a: factores_prueba.append('A')
        if usar_b: factores_prueba.append('B')
        if usar_c: factores_prueba.append('C')
        if st.button("Calcular Pruebas Múltiples", key="btn_calc_f23_final"):
            if not factores_prueba: st.warning("⚠️ Selecciona al menos un factor.")
            else:
                resultados_multiples = {}
                for factor in factores_prueba:
                    if metodo == "Tukey HSD":
                        tukey = realizar_tukey(df, factor, 'Y')
                        if tukey is not None: resultados_multiples[factor] = ('tukey', tukey)
                    else:
                        lsd_df = realizar_lsd(df, factor, 'Y', mse_c, gl_c)
                        if lsd_df is not None: resultados_multiples[factor] = ('lsd', lsd_df, mse_c, gl_c)
                st.session_state.pruebas_multiples['f23_resultado_tab'] = resultados_multiples
        if 'f23_resultado_tab' in st.session_state.pruebas_multiples:
            for factor, r in st.session_state.pruebas_multiples['f23_resultado_tab'].items():
                st.markdown(f"#### 📌 Resultados para Factor {factor}")
                if r[0] == 'tukey':
                    st.dataframe(pd.DataFrame(data=r[1].summary().data[1:], columns=r[1].summary().data[0]))
                    for l in interpretar_tukey(r[1]): st.write(l)
                else:
                    st.write(f"**MSE = {r[2]:.4f}, GL Error = {int(r[3])}**"); st.dataframe(r[1])
                    for l in interpretar_lsd(r[1]): st.write(l)
    else:
        st.warning("⚠️ Primero calcula el ANOVA en la pestaña 'Factorial Tres Factores'.")

# ============================================================
# TAB 7: DISEÑO 2^k (CON PARETO, DANIEL Y MODELO REDUCIDO)
# ============================================================
with tab7:
    st.subheader("📌 Diseño 2^k")
    k = st.number_input("Factores (k):", min_value=2, max_value=8, value=3, step=1, key="k_2k")
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
            mapeo = {factores[i]: f'F{i+1}' for i in range(k)}
            mapeo[col_r] = 'Y'
            df.rename(columns=mapeo, inplace=True)
            for i in range(k): df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
            factores_list = [f'F{i+1}' for i in range(k)]
            modelo, anova = analizar_anova_mejorada(df, 'Y ~ ' + ' * '.join(factores_list), f"2^{k}", "d2k", "d2k", tipo_anova)
            
            if modelo is not None:
                st.session_state.pruebas_multiples['d2k'] = {'df': df, 'modelo': modelo, 'anova': anova}
                mostrar_tabla_anova("d2k")
                mostrar_summary_statistics(modelo, df, factores_list, 'Y', "d2k")
                fig_sc = graficar_sumas_cuadrados(anova)
                if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                
                # ⭐ NUEVO: Pareto, Daniel y Excluir Efectos
                mostrar_pareto_y_daniel(modelo, anova, 'Y', len(df))
                ui_excluir_efectos(df, 'Y', anova, tipo_anova, "d2k")
                
                # ⭐ MOSTRAR MEJOR ANOVA
                if 'd2k_red' in st.session_state.resultados_anova:
                    st.markdown("### 🌟 Mejor ANOVA (Modelo Reducido)")
                    mostrar_tabla_anova("d2k_red")
                    
                mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "d2k", "Residuos")

# ============================================================
# ⭐ TAB 8: DISEÑO FRACCIONADO (CON TABLAS ANOVA Y MEJOR ANOVA)
# ============================================================
def generar_diseno_fraccionado(k, p, randomize=True):
    basicos = k - p
    if basicos < 2: return None
    combinaciones = list(product(*[[-1, 1]] * basicos))
    df = pd.DataFrame(combinaciones, columns=[chr(65 + i) for i in range(basicos)])
    generadores = {(3, 1): ['AB'], (4, 1): ['ABC'], (5, 1): ['ABCD'], (5, 2): ['AB', 'AC'], (6, 1): ['ABCDE'], (6, 2): ['ABC', 'ABD'], (7, 1): ['ABCDEF'], (7, 2): ['ABCD', 'ABCE'], (7, 3): ['ABC', 'BCD', 'ACD'], (7, 4): ['AB', 'AC', 'BC', 'ABC'], (8, 1): ['ABCDEFG'], (8, 2): ['ABCD', 'ABCE'], (8, 3): ['ABC', 'BCD', 'ACD'], (8, 4): ['AB', 'AC', 'BC', 'ABC']}
    gen_list = generadores.get((k, p), ['AB', 'AC', 'BC'][:p])
    for i, gen in enumerate(gen_list):
        val = np.ones(len(df))
        for char in gen:
            if char in df.columns: val *= df[char]
        df[chr(65 + basicos + i)] = val.astype(int)
    if randomize: df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

with tab8:
    st.header("📌 Diseño Fraccionado 2^(k-p)")
    
    st.subheader("⚙️ 1. Configuración de Datos")
    origen = st.radio("Selecciona el origen de los datos:", ["Generar Diseño Vacío", "Cargar desde CSV", "Cargar desde URL"], horizontal=True, key="fr_origen")
    
    if origen == "Generar Diseño Vacío":
        col1, col2, col3, col4 = st.columns(4)
        with col1: k = st.number_input("Factores (k):", min_value=3, max_value=8, value=7, step=1, key="fr_k")
        with col2: p = st.number_input("Fracción (p):", min_value=1, max_value=max(1, k - 2), value=4, step=1, key="fr_p")
        with col3: n_replicas = st.number_input("Réplicas:", min_value=1, max_value=5, value=1, step=1, key="fr_rep")
        with col4: randomizar = st.checkbox("Aleatorizar", value=True, key="fr_rand")

        resolucion = "III" if p >= 4 else "IV" if p >= 2 else "V"
        n_base = 2 ** (k - p)
        st.info(f"**Diseño 2^({k}-{p})** | Resolución: **{resolucion}** | Base de corridas: **{n_base}** | Total: **{n_base * n_replicas}**")

        with st.expander("📝 Nombres de los Factores (opcional)", expanded=False):
            nombres_factores = []
            cols_nombres = st.columns(min(k, 5))
            for i in range(k):
                with cols_nombres[i % len(cols_nombres)]:
                    nombres_factores.append(st.text_input(f"Factor {chr(65+i)}:", value=f"Factor_{chr(65+i)}", key=f"fr_nombre_{i}"))

        if st.button("🎲 Generar Diseño", key="fr_btn_generar", type="primary"):
            df_gen = generar_diseno_fraccionado(k, p, randomizar)
            if df_gen is not None:
                if n_replicas > 1: df_gen = pd.concat([df_gen] * n_replicas, ignore_index=True).reset_index(drop=True)
                df_gen = df_gen.rename(columns={chr(65 + i): nombres_factores[i] for i in range(k)})
                df_gen['Respuesta'] = np.nan
                st.session_state.fr_df = df_gen
                st.session_state.fr_analizado = False
                st.session_state.fr_colapsado = None
                st.rerun()
                
    elif origen == "Cargar desde CSV":
        archivo = st.file_uploader("Sube tu archivo CSV (debe incluir la columna 'Respuesta'):", type=['csv', 'txt'], key="fr_upload")
        if archivo is not None and st.button("📊 Cargar CSV", key="fr_btn_csv"):
            df_subido = cargar_datos_desde_csv(archivo)
            if df_subido is not None:
                st.session_state.fr_df = df_subido
                st.session_state.fr_analizado = False
                st.session_state.fr_colapsado = None
                st.rerun()
                
    elif origen == "Cargar desde URL":
        url = st.text_input("URL del CSV:", key="fr_url")
        if url and st.button("📊 Cargar URL", key="fr_btn_url"):
            df_subido = cargar_datos_desde_url(url)
            if df_subido is not None:
                st.session_state.fr_df = df_subido
                st.session_state.fr_analizado = False
                st.session_state.fr_colapsado = None
                st.rerun()

    if st.button("🗑️ Limpiar Todo", key="fr_btn_limpiar"):
        st.session_state.fr_df = None
        st.session_state.fr_analizado = False
        st.session_state.fr_colapsado = None
        if 'fr' in st.session_state.resultados_anova: del st.session_state.resultados_anova['fr']
        if 'fr_red' in st.session_state.resultados_anova: del st.session_state.resultados_anova['fr_red']
        st.rerun()

    if st.session_state.get('fr_df') is not None:
        st.markdown("---")
        st.subheader("📋 Tabla del Diseño (edita la columna 'Respuesta')")
        df_editado = st.data_editor(st.session_state.fr_df, num_rows="fixed", key="fr_editor")
        st.session_state.fr_df = df_editado

        st.markdown("---")
        st.subheader("📊 2. Análisis ANOVA y Gráficos")
        tipo_anova = selector_tipo_anova("fr")
        factores_cols = [c for c in df_editado.columns if c != 'Respuesta' and c != 'Orden']

        if st.button("🔍 Calcular ANOVA Fraccionado", key="fr_btn_calc", type="primary"):
            if df_editado['Respuesta'].isnull().all():
                st.error("❌ La columna 'Respuesta' está vacía.")
            else:
                st.session_state.fr_analizado = True
                st.session_state.fr_colapsado = None
                df_analisis = df_editado.copy().reset_index(drop=True)
                for f in factores_cols: df_analisis[f] = df_analisis[f].astype('category')
                formula = 'Respuesta ~ ' + ' + '.join(factores_cols)
                analizar_anova_mejorada(df_analisis, formula, "ANOVA Inicial (Efectos Principales)", "fr", "fr", tipo_anova)
                st.rerun()

        # Mostrar resultados persistentes
        if st.session_state.fr_analizado and 'fr' in st.session_state.resultados_anova:
            res = st.session_state.resultados_anova['fr']
            modelo, anova, df_analisis = res['modelo'], res['anova'], res['df']

            # ⭐ 1. MOSTRAR TABLA ANOVA INICIAL
            st.markdown("### 🔹 ANOVA Inicial")
            mostrar_tabla_anova("fr")
            
            mostrar_summary_statistics(modelo, df_analisis, factores_cols, 'Respuesta', "fr")
            mostrar_pareto_y_daniel(modelo, anova, 'Respuesta', len(df_analisis))
            graficar_efectos_e_interacciones(df_analisis, factores_cols, 'Respuesta')
            
            st.markdown("---")
            st.subheader("🛠️ 3. Refinamiento (Selección del Mejor Modelo)")
            todos_efectos = [idx for idx in anova.index if idx != 'Residual']
            efectos_seleccionados = st.multiselect("Efectos a MANTENER en el modelo final:", todos_efectos, default=todos_efectos, key="fr_excluir")
            
            if st.button("🔄 Recalcular Modelo Reducido", key="fr_recalc"):
                if len(efectos_seleccionados) == 0:
                    st.warning("⚠️ Debes mantener al menos un efecto.")
                else:
                    formula_red = 'Respuesta ~ ' + ' + '.join(efectos_seleccionados)
                    analizar_anova_mejorada(df_analisis, formula_red, "Modelo Reducido (Mejor ANOVA)", "fr_red", "reducido", tipo_anova)
                    st.rerun()
                    
            # ⭐ 2. MOSTRAR MEJOR ANOVA Y SUS RESULTADOS
            if 'fr_red' in st.session_state.resultados_anova:
                res_red = st.session_state.resultados_anova['fr_red']
                st.markdown("### 🌟 Mejor ANOVA (Modelo Reducido)")
                mostrar_tabla_anova("fr_red")
                mostrar_summary_statistics(res_red['modelo'], df_analisis, efectos_seleccionados, 'Respuesta', "fr_red")
                mostrar_graficos_residuos_directo(res_red['modelo'], df_analisis, efectos_seleccionados[0], 'Respuesta', "fr_red", "Residuos del Mejor Modelo")
                st.subheader("📝 Conclusiones del Modelo Final")
                for c in generar_conclusiones_factorial(res_red['anova'], res_red['modelo'], df_analisis, efectos_seleccionados, 'Respuesta'):
                    if c.startswith("✅"): st.success(c)
                    elif c.startswith("❌"): st.error(c)
                    else: st.info(c)

            st.markdown("---")
            st.subheader("📉 4. Colapsar Diseño")
            col_fijar = st.selectbox("Factor a fijar:", factores_cols, key="fr_colapsar_factor")
            niveles_fijar = sorted(df_analisis[col_fijar].dropna().unique(), key=lambda x: str(x))
            col_nivel = st.selectbox("Nivel a fijar:", niveles_fijar, key="fr_colapsar_nivel")
            
            if st.button("Colapsar y Re-analizar", key="fr_btn_colapsar"):
                df_col = df_analisis[df_analisis[col_fijar] == col_nivel].copy().reset_index(drop=True)
                f_restantes = [f for f in factores_cols if f != col_fijar]
                if len(f_restantes) > 0 and not df_col['Respuesta'].isnull().all():
                    for f in f_restantes: df_col[f] = df_col[f].astype('category')
                    formula_col = 'Respuesta ~ ' + ' + '.join(f_restantes)
                    modelo_c, anova_c = analizar_anova_mejorada(df_col, formula_col, "ANOVA Colapsado", "fr_col", "colapsado", tipo_anova)
                    if modelo_c is not None:
                        st.session_state.fr_colapsado = (modelo_c, anova_c, df_col, f_restantes)
                        st.rerun()

            if st.session_state.fr_colapsado is not None:
                modelo_c, anova_c, df_col, f_restantes = st.session_state.fr_colapsado
                st.success(f"✅ Análisis colapsado (Fijando {col_fijar} en {col_nivel}):")
                mostrar_tabla_anova("fr_col")
                mostrar_summary_statistics(modelo_c, df_col, f_restantes, 'Respuesta', "fr_col")
                mostrar_pareto_y_daniel(modelo_c, anova_c, 'Respuesta', len(df_col))
                graficar_efectos_e_interacciones(df_col, f_restantes, 'Respuesta')
                if st.button("Cerrar Sub-análisis", key="fr_cerrar_colapso"):
                    st.session_state.fr_colapsado = None
                    st.rerun()
    else:
        st.info("👆 Selecciona un origen de datos y presiona el botón para comenzar.")

# ============================================================
# TAB 9 a 13
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
            mapeo = {factores[i]: f'F{i+1}' for i in range(n_f)}
            mapeo[col_r] = 'Y'
            df.rename(columns=mapeo, inplace=True)
            vars_list = [f'F{i+1}' for i in range(n_f)]
            formula = 'Y ~ ' + ' + '.join(vars_list) + ' + ' + ' + '.join([f'I({v}**2)' for v in vars_list])
            if n_f >= 2: formula += ' + ' + ' + '.join([f'F{i+1}:F{j+1}' for i in range(n_f) for j in range(i+1, n_f)])
            modelo, anova = analizar_anova_mejorada(df, formula, "MSR", "msr", "msr", tipo_anova)
            if modelo is not None:
                mostrar_summary_statistics(modelo, df, vars_list, 'Y', "msr")
                mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "msr", "Residuos")

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
            mapeo = {factores[i]: f'F{i+1}' for i in range(min(n_f, 8))}
            restantes = [c for c in cols if c not in factores and c != col_r]
            for i in range(8, n_f):
                if restantes: mapeo[restantes.pop(0)] = f'F{i+1}'
            mapeo[col_r] = 'Y'
            df.rename(columns=mapeo, inplace=True)
            for i in range(n_f): df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
            factores_list = [f'F{i+1}' for i in range(n_f)]
            modelo, anova = analizar_anova_mejorada(df, 'Y ~ ' + ' + '.join(factores_list), "Taguchi", "tag", "tag", tipo_anova)
            if modelo is not None:
                mostrar_summary_statistics(modelo, df, factores_list, 'Y', "tag")
                mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "tag", "Residuos")

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
                st.markdown("---")
                st.markdown("**Summary Statistics de la Respuesta 1 (Y1):**")
                mostrar_summary_statistics(m1, df, ['F1', 'F2'], 'Y1', "omr_y1", incluir_descriptivas=False)
            except Exception as e: st.error(f"❌ {e}")

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
                mostrar_summary_statistics(modelo, df, ['C1', 'C2', 'C3'], 'Y', "ad")
                mostrar_graficos_residuos_directo(modelo, df, 'C1', 'Y', "ad", "Residuos")

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
                mostrar_summary_statistics(modelo, df, ['F1', 'F2', 'F3'], 'Y', "sp")
                fig_sc = graficar_sumas_cuadrados(anova)
                if fig_sc: st.plotly_chart(fig_sc, use_container_width=True)
                mostrar_graficos_residuos_directo(modelo, df, 'F1', 'Y', "sp", "Residuos")
