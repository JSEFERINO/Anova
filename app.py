import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="📊 Diseños Experimentales y ANOVA",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Calculadora de Diseños Experimentales y ANOVA")
st.markdown("---")

# Función para procesar datos
def procesar_datos(texto, n_columnas):
    try:
        lineas = [line.strip() for line in texto.split('\n') if line.strip()]
        datos = []
        for linea in lineas:
            valores = [v.strip() for v in linea.split(',') if v.strip()]
            if len(valores) == n_columnas:
                try:
                    valores_num = [float(v) for v in valores]
                    datos.append(valores_num)
                except:
                    datos.append(valores)
        if not datos:
            return None
        df = pd.DataFrame(datos)
        df.columns = [f'V{i+1}' for i in range(n_columnas)]
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
        return df
    except:
        return None

# Crear pestañas
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "📊 ANOVA Un Factor",
    "📊 Diseño en Bloques",
    "📐 Cuadrado Latino",
    "📐 Greco-Latino",
    "🎯 Factorial 2²",
    "🎯 Factorial 2³",
    "📊 Diseño 2^k",
    "📊 Fraccionado",
    "📈 MSR",
    "⚙️ Taguchi",
    "🎯 Optimización",
    "➕ Aditivos",
    "📦 Split-Plot"
])

# PESTAÑA 1: ANOVA UN FACTOR
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Completamente Aleatorizado")
        st.caption("Formato: Tratamiento,Valor")
        datos = st.text_area("Datos:", value="A,23\nA,25\nA,24\nA,22\nA,26\nB,30\nB,32\nB,28\nB,31\nB,29\nC,35\nC,33\nC,37\nC,34\nC,36", height=200, key="dca_text")
        if st.button("🔍 Calcular", key="dca_btn"):
            try:
                df = procesar_datos(datos, 2)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    df.columns = ['Tratamiento', 'Valor']
                    df['Tratamiento'] = df['Tratamiento'].astype('category')
                    modelo = ols('Valor ~ C(Tratamiento)', data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ ANOVA calculado")
                    st.dataframe(anova_table)
                    st.session_state['dca'] = {'df': df}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'dca' in st.session_state:
            df = st.session_state['dca']['df']
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 2: DISEÑO EN BLOQUES
with tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño en Bloques")
        st.caption("Formato: Tratamiento,Bloque,Valor")
        datos = st.text_area("Datos:", value="A,B1,23\nA,B2,25\nA,B3,24\nA,B4,22\nB,B1,30\nB,B2,32\nB,B3,28\nB,B4,31\nC,B1,35\nC,B2,33\nC,B3,37\nC,B4,34", height=200, key="dbca_text")
        if st.button("🔍 Calcular", key="dbca_btn"):
            try:
                df = procesar_datos(datos, 3)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    df.columns = ['Tratamiento', 'Bloque', 'Valor']
                    df['Tratamiento'] = df['Tratamiento'].astype('category')
                    df['Bloque'] = df['Bloque'].astype('category')
                    modelo = ols('Valor ~ C(Tratamiento) + C(Bloque)', data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ ANOVA calculado")
                    st.dataframe(anova_table)
                    st.session_state['dbca'] = {'df': df}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'dbca' in st.session_state:
            df = st.session_state['dbca']['df']
            fig = px.box(df, x='Tratamiento', y='Valor', color='Tratamiento', facet_col='Bloque')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 3: CUADRADO LATINO
with tab3:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Cuadrado Latino (3×3)")
        st.caption("Formato: Fila,Columna,Tratamiento,Valor")
        datos = st.text_area("Datos:", value="1,1,A,10\n1,2,B,12\n1,3,C,11\n2,1,B,14\n2,2,C,15\n2,3,A,13\n3,1,C,12\n3,2,A,11\n3,3,B,13", height=200, key="cl_text")
        if st.button("🔍 Calcular", key="cl_btn"):
            try:
                df = procesar_datos(datos, 4)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    df.columns = ['Fila', 'Columna', 'Tratamiento', 'Valor']
                    for col in ['Fila', 'Columna', 'Tratamiento']:
                        df[col] = df[col].astype('category')
                    modelo = ols('Valor ~ C(Fila) + C(Columna) + C(Tratamiento)', data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ ANOVA calculado")
                    st.dataframe(anova_table)
                    st.session_state['cl'] = {'df': df}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'cl' in st.session_state:
            df = st.session_state['cl']['df']
            pivot = df.pivot(index='Fila', columns='Columna', values='Valor')
            fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='Blues')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 4: CUADRADO GRECO-LATINO
with tab4:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Cuadrado Greco-Latino (3×3)")
        st.caption("Formato: Fila,Columna,Tratamiento,Greco,Valor")
        datos = st.text_area("Datos:", value="1,1,A,α,10\n1,2,B,β,12\n1,3,C,γ,11\n2,1,B,γ,14\n2,2,C,α,15\n2,3,A,β,13\n3,1,C,β,12\n3,2,A,γ,11\n3,3,B,α,13", height=200, key="cgl_text")
        if st.button("🔍 Calcular", key="cgl_btn"):
            try:
                df = procesar_datos(datos, 5)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    df.columns = ['Fila', 'Columna', 'Tratamiento', 'Greco', 'Valor']
                    for col in ['Fila', 'Columna', 'Tratamiento', 'Greco']:
                        df[col] = df[col].astype('category')
                    modelo = ols('Valor ~ C(Fila) + C(Columna) + C(Tratamiento) + C(Greco)', data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ ANOVA calculado")
                    st.dataframe(anova_table)
                    st.session_state['cgl'] = {'df': df}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'cgl' in st.session_state:
            df = st.session_state['cgl']['df']
            pivot = df.pivot(index='Fila', columns='Columna', values='Valor')
            fig = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale='Greens')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 5: FACTORIAL 2²
with tab5:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Factorial 2²")
        st.caption("Formato: A,B,Y")
        datos = st.text_area("Datos:", value="-1,-1,10\n-1,-1,12\n-1,1,14\n-1,1,16\n1,-1,18\n1,-1,20\n1,1,22\n1,1,24", height=200, key="f22_text")
        if st.button("🔍 Calcular", key="f22_btn"):
            try:
                df = procesar_datos(datos, 3)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    df.columns = ['A', 'B', 'Y']
                    df['A'] = df['A'].astype('category')
                    df['B'] = df['B'].astype('category')
                    modelo = ols('Y ~ A * B', data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ ANOVA calculado")
                    st.dataframe(anova_table)
                    st.session_state['f22'] = {'df': df}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'f22' in st.session_state:
            df = st.session_state['f22']['df']
            medias = df.groupby(['A', 'B'])['Y'].mean().reset_index()
            fig = px.line(medias, x='A', y='Y', color='B', markers=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 6: FACTORIAL 2³
with tab6:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Factorial 2³")
        st.caption("Formato: A,B,C,Y")
        datos = st.text_area("Datos:", value="-1,-1,-1,10\n-1,-1,-1,12\n-1,-1,1,14\n-1,-1,1,16\n-1,1,-1,18\n-1,1,-1,20\n-1,1,1,22\n-1,1,1,24\n1,-1,-1,26\n1,-1,-1,28\n1,-1,1,30\n1,-1,1,32\n1,1,-1,34\n1,1,-1,36\n1,1,1,38\n1,1,1,40", height=200, key="f23_text")
        if st.button("🔍 Calcular", key="f23_btn"):
            try:
                df = procesar_datos(datos, 4)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    df.columns = ['A', 'B', 'C', 'Y']
                    for col in ['A', 'B', 'C']:
                        df[col] = df[col].astype('category')
                    modelo = ols('Y ~ A * B * C', data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ ANOVA calculado")
                    st.dataframe(anova_table)
                    st.session_state['f23'] = {'df': df}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'f23' in st.session_state:
            df = st.session_state['f23']['df']
            fig = make_subplots(rows=1, cols=3, subplot_titles=['A', 'B', 'C'])
            for i, factor in enumerate(['A', 'B', 'C']):
                medias = df.groupby(factor)['Y'].mean().reset_index()
                fig.add_trace(go.Bar(x=medias[factor], y=medias['Y']), row=1, col=i+1)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 7: DISEÑO 2^k
with tab7:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Factorial 2^k")
        k = st.number_input("Número de factores (k):", min_value=2, max_value=4, value=3, step=1, key="k_2k")
        st.caption(f"Formato: F1,F2,...,F{k},Y")
        datos = st.text_area("Datos:", value="-1,-1,-1,10\n-1,-1,1,12\n-1,1,-1,14\n-1,1,1,16\n1,-1,-1,18\n1,-1,1,20\n1,1,-1,22\n1,1,1,24", height=200, key="2k_text")
        if st.button("🔍 Calcular", key="2k_btn"):
            try:
                df = procesar_datos(datos, k+1)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    nombres = [f'F{i+1}' for i in range(k)] + ['Y']
                    df.columns = nombres
                    for i in range(k):
                        df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
                    factores = [f'F{i+1}' for i in range(k)]
                    formula = 'Y ~ ' + ' * '.join(factores)
                    modelo = ols(formula, data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ ANOVA calculado")
                    st.dataframe(anova_table)
                    st.session_state['d2k'] = {'df': df, 'k': k}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'd2k' in st.session_state:
            df = st.session_state['d2k']['df']
            k = st.session_state['d2k']['k']
            fig = make_subplots(rows=(k+1)//2, cols=2, subplot_titles=[f'F{i+1}' for i in range(k)])
            for i in range(k):
                row = i//2 + 1
                col = i%2 + 1
                medias = df.groupby(f'F{i+1}')['Y'].mean().reset_index()
                fig.add_trace(go.Bar(x=medias[f'F{i+1}'], y=medias['Y']), row=row, col=col)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 8: FRACCIONADO
with tab8:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Fraccionado 2^(k-p)")
        k = st.number_input("k:", min_value=3, max_value=5, value=4, step=1, key="k_fracc")
        p = st.number_input("p:", min_value=1, max_value=2, value=1, step=1, key="p_fracc")
        st.caption(f"Formato: F1,F2,...,F{k},Y")
        datos = st.text_area("Datos:", value="-1,-1,-1,-1,10\n-1,-1,-1,1,12\n-1,-1,1,-1,14\n-1,-1,1,1,16\n-1,1,-1,-1,18\n-1,1,-1,1,20\n-1,1,1,-1,22\n-1,1,1,1,24", height=200, key="fracc_text")
        if st.button("🔍 Calcular", key="fracc_btn"):
            try:
                df = procesar_datos(datos, k+1)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    nombres = [f'F{i+1}' for i in range(k)] + ['Y']
                    df.columns = nombres
                    for i in range(k):
                        df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
                    factores = [f'F{i+1}' for i in range(k)]
                    formula = 'Y ~ ' + ' + '.join(factores)
                    modelo = ols(formula, data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ ANOVA calculado")
                    st.dataframe(anova_table)
                    st.session_state['fracc'] = {'df': df, 'k': k}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'fracc' in st.session_state:
            df = st.session_state['fracc']['df']
            k = st.session_state['fracc']['k']
            fig = make_subplots(rows=(k+1)//2, cols=2, subplot_titles=[f'F{i+1}' for i in range(k)])
            for i in range(k):
                row = i//2 + 1
                col = i%2 + 1
                medias = df.groupby(f'F{i+1}')['Y'].mean().reset_index()
                fig.add_trace(go.Bar(x=medias[f'F{i+1}'], y=medias['Y']), row=row, col=col)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 9: MSR
with tab9:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Metodología de Superficie de Respuesta")
        n_fact = st.number_input("Número de factores:", min_value=2, max_value=4, value=2, step=1, key="n_fact_msr")
        st.caption(f"Formato: F1,F2,...,F{n_fact},Y")
        datos = st.text_area("Datos:", value="-1,-1,10\n-1,1,12\n1,-1,14\n1,1,16\n0,0,11\n0,0,13\n-1.414,0,9\n1.414,0,15\n0,-1.414,11\n0,1.414,17", height=200, key="msr_text")
        if st.button("🔍 Calcular MSR", key="msr_btn"):
            try:
                nf = int(n_fact)
                df = procesar_datos(datos, nf+1)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    nombres = [f'F{i+1}' for i in range(nf)] + ['Y']
                    df.columns = nombres
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
                    modelo = ols(formula, data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ Modelo MSR ajustado")
                    st.dataframe(anova_table)
                    st.write(f"**R² = {modelo.rsquared:.6f}**")
                    st.session_state['msr'] = {'df': df, 'modelo': modelo}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'msr' in st.session_state:
            df = st.session_state['msr']['df']
            if len(df.columns) == 3:
                fig = px.scatter_3d(df, x='F1', y='F2', z='Y', title="Superficie de Respuesta")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 10: TAGUCHI
with tab10:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseño Robusto Taguchi")
        arreglo = st.selectbox("Arreglo Ortogonal:", ["L4 (2³)", "L8 (2⁷)", "L9 (3⁴)"], index=1, key="taguchi_arreglo")
        n_fact = {"L4 (2³)": 3, "L8 (2⁷)": 7, "L9 (3⁴)": 4}[arreglo]
        st.caption(f"Formato: F1,F2,...,F{n_fact},Y")
        datos = st.text_area("Datos:", value="-1,-1,-1,10\n-1,-1,1,12\n-1,1,-1,14\n-1,1,1,16\n1,-1,-1,18\n1,-1,1,20\n1,1,-1,22\n1,1,1,24", height=200, key="taguchi_text")
        if st.button("🔍 Calcular Taguchi", key="taguchi_btn"):
            try:
                df = procesar_datos(datos, n_fact+1)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    nombres = [f'F{i+1}' for i in range(n_fact)] + ['Y']
                    df.columns = nombres
                    for i in range(n_fact):
                        df[f'F{i+1}'] = df[f'F{i+1}'].astype('category')
                    factores = [f'F{i+1}' for i in range(n_fact)]
                    formula = 'Y ~ ' + ' + '.join(factores)
                    modelo = ols(formula, data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ Análisis Taguchi completado")
                    st.dataframe(anova_table)
                    st.session_state['taguchi'] = {'df': df}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'taguchi' in st.session_state:
            df = st.session_state['taguchi']['df']
            st.write("✅ Análisis completado")

# PESTAÑA 11: OPTIMIZACIÓN
with tab11:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Optimización Múltiple")
        st.caption("Formato: F1,F2,Y1,Y2")
        datos = st.text_area("Datos:", value="-1,-1,10,20\n-1,1,12,22\n1,-1,14,18\n1,1,16,24\n0,0,11,21", height=150, key="omr_text")
        if st.button("🔍 Optimizar", key="omr_btn"):
            try:
                df = procesar_datos(datos, 4)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    df.columns = ['F1', 'F2', 'Y1', 'Y2']
                    modelo1 = ols('Y1 ~ F1 + F2 + I(F1**2) + I(F2**2) + F1:F2', data=df).fit()
                    modelo2 = ols('Y2 ~ F1 + F2 + I(F1**2) + I(F2**2) + F1:F2', data=df).fit()
                    grid = pd.DataFrame([(f1, f2) for f1 in np.linspace(-1.5, 1.5, 20) for f2 in np.linspace(-1.5, 1.5, 20)], columns=['F1', 'F2'])
                    grid['Y1'] = modelo1.predict(grid)
                    grid['Y2'] = modelo2.predict(grid)
                    grid['D1'] = ((grid['Y1'] - 10) / (20 - 10)).clip(0, 1)
                    grid['D2'] = (1 - (grid['Y2'] - 18) / (30 - 18)).clip(0, 1)
                    grid['D'] = (grid['D1'] * grid['D2']) ** 0.5
                    optimo = grid.loc[grid['D'].idxmax()]
                    st.success("✅ Optimización completada")
                    st.write(f"**Óptimo:** F1={optimo['F1']:.3f}, F2={optimo['F2']:.3f}")
                    st.write(f"**Deseabilidad = {optimo['D']:.4f}**")
                    st.session_state['omr'] = {'grid': grid, 'optimo': optimo}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'omr' in st.session_state:
            grid = st.session_state['omr']['grid']
            optimo = st.session_state['omr']['optimo']
            pivot = grid.pivot(index='F1', columns='F2', values='D')
            fig = px.imshow(pivot, aspect="auto", color_continuous_scale='Blues')
            fig.add_scatter(x=[optimo['F2']], y=[optimo['F1']], mode='markers', marker=dict(color='red', size=15, symbol='x'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 12: ADITIVOS
with tab12:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Diseños Añadidos")
        st.caption("Formato: C1,C2,C3,Y")
        datos = st.text_area("Datos:", value="0.2,0.3,0.5,10\n0.3,0.2,0.5,12\n0.4,0.3,0.3,14\n0.5,0.2,0.3,16\n0.3,0.4,0.3,18\n0.2,0.5,0.3,20\n0.5,0.3,0.2,22\n0.3,0.4,0.3,24", height=200, key="aditivos_text")
        if st.button("🔍 Calcular", key="aditivos_btn"):
            try:
                df = procesar_datos(datos, 4)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    df.columns = ['C1', 'C2', 'C3', 'Y']
                    modelo = ols('Y ~ C1 + C2 + C3', data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ Modelo aditivo ajustado")
                    st.dataframe(anova_table)
                    st.write(f"**R² = {modelo.rsquared:.6f}**")
                    st.session_state['aditivos'] = {'df': df}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'aditivos' in st.session_state:
            df = st.session_state['aditivos']['df']
            fig = make_subplots(rows=1, cols=3, subplot_titles=['C1', 'C2', 'C3'])
            for i in range(3):
                comp = f'C{i+1}'
                medias = df.groupby(comp)['Y'].mean().reset_index()
                fig.add_trace(go.Scatter(x=medias[comp], y=medias['Y'], mode='lines+markers'), row=1, col=i+1)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# PESTAÑA 13: SPLIT-PLOT
with tab13:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📌 Parcelas Divididas")
        st.caption("Formato: F1,F2,F3,Y")
        datos = st.text_area("Datos:", value="-1,-1,-1,10\n-1,-1,1,12\n-1,1,-1,14\n-1,1,1,16\n1,-1,-1,18\n1,-1,1,20\n1,1,-1,22\n1,1,1,24", height=200, key="split_text")
        if st.button("🔍 Calcular", key="split_btn"):
            try:
                df = procesar_datos(datos, 4)
                if df is None:
                    st.error("❌ Error: Datos no válidos")
                else:
                    df.columns = ['F1', 'F2', 'F3', 'Y']
                    for col in ['F1', 'F2', 'F3']:
                        df[col] = df[col].astype('category')
                    modelo = ols('Y ~ F1 * F2 * F3', data=df).fit()
                    anova_table = sm.stats.anova_lm(modelo, typ=2)
                    st.success("✅ ANOVA calculado")
                    st.dataframe(anova_table)
                    st.session_state['split'] = {'df': df}
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    with col2:
        if 'split' in st.session_state:
            df = st.session_state['split']['df']
            fig = px.box(df, x='F1', y='Y', color='F2', facet_col='F3')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

print("✅ app.py creado exitosamente!")
