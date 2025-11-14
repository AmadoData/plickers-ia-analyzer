import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os 
import json 
from google import genai
from google.genai.errors import APIError
from fpdf import FPDF
import base64
 
warnings.filterwarnings("ignore", category=UserWarning)
 
# ==============================================================================
# CONFIGURACIÓN INICIAL Y CLAVE API
# ==============================================================================
THRESHOLD_ACIERTO = 0.60 

# Inicializa el cliente de Gemini y la API key
# La inicialización se intenta, pero NO se detiene la app si falla la clave.
client = None
if 'GEMINI_API_KEY' in os.environ and os.environ['GEMINI_API_KEY']:
    try:
        # Intenta inicializar el cliente solo si la clave existe
        client = genai.Client()
    except Exception as e:
        # En caso de un error de inicialización, client permanece como None
        st.warning(f"⚠️ Advertencia: Error al inicializar el cliente de Gemini. Las funciones de IA no estarán disponibles. {e}")
else:
    st.warning("⚠️ Advertencia: La variable de entorno 'GEMINI_API_KEY' no está configurada. Las funciones de IA no estarán disponibles.")


# ==============================================================================
# FUNCIONES CENTRALES (Adaptadas para Streamlit)
# ==============================================================================
 
@st.cache_data
def load_and_clean_data(csv_file_content):
    """Carga los datos, detecta el encabezado, extrae la clave de respuestas y estandariza las columnas."""
    
    csv_data = csv_file_content.getvalue().decode('utf-8', errors='replace')
    lines = csv_data.splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        if 'Card Number' in line or 'First name' in line:
            header_idx = i
            break
 
    if header_idx == -1:
        raise ValueError("No se pudo identificar la fila de encabezado (buscando 'Card Number' o 'First name').")
    
    skip_rows_count = header_idx
 
    # Lectura robusta (Coma o Punto y coma)
    try:
        df = pd.read_csv(
            io.StringIO(csv_data), 
            skiprows=skip_rows_count, 
            header=0, 
            delimiter=',', 
            quotechar='"', 
            engine='python'
        )
    except Exception:
        # Fallback para archivos con separador ';'
        df = pd.read_csv(
            io.StringIO(csv_data), 
            skiprows=skip_rows_count, 
            header=0, 
            delimiter=';', 
            quotechar='"', 
            engine='python'
        )
 
    # Limpieza y estandarización de columnas
    df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
    col_keywords = {'Score': 'Score', 'Correct': 'Correct', 'Answered': 'Answered', 
                    'Card': 'Card Number', 'First': 'First name', 'Last': 'Last Name'}
    new_column_names = {}
    for current_col in df.columns:
        found_match = False
        for keyword, standardized_name in col_keywords.items():
            # Usar 'in' para que detecte 'First name' incluso si la columna se llama 'Student First name'
            if keyword in current_col: 
                new_column_names[current_col] = standardized_name
                found_match = True
                break
        if not found_match:
            new_column_names[current_col] = current_col
    df.rename(columns=new_column_names, inplace=True)
    
    required_cols = ['Answered', 'Score', 'First name', 'Last Name']
    if not all(col in df.columns for col in required_cols):
         missing = [col for col in required_cols if col not in df.columns]
         raise ValueError(f"Columnas esenciales no encontradas. Faltantes: {missing}. Columnas disponibles: {df.columns.tolist()}")
 
    df = df.dropna(how='all')
    
    standard_cols = ['Card Number', 'First name', 'Last Name', 'Score', 'Correct', 'Answered']
    question_cols = [col for col in df.columns if col not in standard_cols]
    
    # Detección de la fila de respuestas correctas
    answer_key_row = None
    answer_key_index = -1
    
    num_questions = len(question_cols)
    # Umbral dinámico: Mínimo 3 respuestas válidas o la mitad del total
    threshold = max(3, int(num_questions / 2)) 
    
    for i in range(len(df)):
        row = df.iloc[i][question_cols].dropna().astype(str).str.upper()
        if len(row[(row.isin(['A', 'B', 'C', 'D']))]) >= threshold:
           answer_key_row = df.iloc[i]
           answer_key_index = df.index[i]
           break
 
    if answer_key_row is None:
        raise ValueError("No se pudo detectar la fila de respuestas correctas (Answer Key).")
 
    answer_key = answer_key_row[question_cols].to_dict()
    df = df.drop(answer_key_index).reset_index(drop=True) 
    
    df['Score_num'] = df['Score'].astype(str).str.replace('%', '').replace('-', '0').astype(float) / 100
    df = df[df['Score_num'] > 0].reset_index(drop=True)
    
    for col in question_cols:
        df[col] = df[col].astype(str).str.upper().str.strip().str[0]
 
    return df, answer_key, question_cols
 
@st.cache_data
def generate_topics_with_gemini(question_cols):
    """Genera tópicos pedagógicos para cada pregunta usando la API de Gemini."""
    
    if client is None:
        # Fallback si el cliente no se inicializó
        return {q: f'Tópico Genérico {i+1}' for i, q in enumerate(question_cols)}
    
    questions_list_str = "\n".join([f"\"{q}\"" for q in question_cols])
    
    prompt = f"""
    Eres un experto en currículum y análisis de contenido educativo. Te proporcionaré una lista de preguntas de examen. Tu tarea es asignar un nombre de 'Tópico Pedagógico' conciso (máximo 4 palabras) y relevante a cada pregunta.
 
    Proporciona tu respuesta **estrictamente en formato JSON**, donde la **clave** sea el texto COMPLETO de la pregunta (incluyendo comillas si las tiene) y el **valor** sea el 'Tópico Pedagógico' que le asignaste. No incluyas ninguna otra explicación o texto fuera del objeto JSON.
 
    PREGUNTAS:
    {questions_list_str}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        
        topic_map = json.loads(json_text)
        
        final_topic_map = topic_map.copy()
        for q in question_cols:
            # Asegura que todas las preguntas tengan un tópico
            if q not in final_topic_map:
                final_topic_map[q] = 'Tópico No Mapeado'
                     
        return final_topic_map
        
    except Exception as e:
        st.warning(f"❌ ERROR al generar tópicos con Gemini: {e}. Usando tópicos genéricos.")
        # Fallback si la API llama pero falla
        return {q: f'Tópico Genérico {i+1}' for i, q in enumerate(question_cols)}
 
@st.cache_data
def analyze_and_calculate(df_data, answer_key, question_cols):
    """Realiza todos los cálculos de acierto por pregunta, tópico y alumno."""
    
    st.info("🧠 Generando tópicos dinámicos y realizando cálculos estadísticos...")
    question_topics = generate_topics_with_gemini(question_cols)
 
    acierto_por_pregunta = {}
    for q_col in question_cols:
        aciertos = (df_data[q_col] == answer_key[q_col]).sum()
        total_respuestas = len(df_data) 
        acierto_por_pregunta[q_col] = aciertos / total_respuestas if total_respuestas > 0 else 0
 
    df_acierto_pregunta = pd.DataFrame(
        list(acierto_por_pregunta.items()), 
        columns=['Pregunta', '% Acierto']
    )
    df_acierto_pregunta['Tópico'] = df_acierto_pregunta['Pregunta'].map(question_topics)
 
    acierto_por_topico = df_acierto_pregunta.groupby('Tópico')['% Acierto'].mean().reset_index()
    acierto_por_topico['% Acierto'] = acierto_por_topico['% Acierto'].round(2)
    acierto_por_topico = acierto_por_topico.sort_values(by='% Acierto')
 
    rendimiento_alumnos = df_data[['First name', 'Last Name', 'Score_num']].copy()
    rendimiento_alumnos['Nombre Completo'] = rendimiento_alumnos['First name'] + ' ' + rendimiento_alumnos['Last Name']
    rendimiento_alumnos = rendimiento_alumnos.drop(columns=['First name', 'Last Name']).sort_values(by='Score_num')
 
    topicos_criticos = acierto_por_topico[acierto_por_topico['% Acierto'] < THRESHOLD_ACIERTO]
    
    return acierto_por_topico, rendimiento_alumnos, topicos_criticos, df_acierto_pregunta
 
@st.cache_data
def generar_recomendaciones_gemini(topicos_criticos_df):
    """Genera recomendaciones pedagógicas específicas usando la API de Gemini."""
    if topicos_criticos_df.empty:
        return "No se encontraron tópicos con rendimiento inferior al 60%. Excelente trabajo."
    
    if client is None:
        return "❌ ERROR: Cliente Gemini no inicializado. Se usa recomendación genérica."
 
    topicos_data = topicos_criticos_df.copy()
    topicos_data['% Acierto'] = (topicos_data['% Acierto'] * 100).astype(int).astype(str) + '%'
    topicos_criticos_str = topicos_data.to_string(index=False, header=True)
    
    prompt = f"""
    Eres un analista pedagógico experto. Basándote en la siguiente tabla de Tópicos Críticos
    de un examen, donde el acierto colectivo fue inferior al 60%, genera recomendaciones
    pedagógicas específicas para el docente.
 
    La respuesta debe ser una lista numerada, una recomendación por cada tópico crítico.
    Cada recomendación debe ser concisa, práctica, y enfocada a la acción (ej. 'Realizar un debate
    sobre...').
    No incluyas introducciones ni conclusiones, solo la lista de recomendaciones.
 
    TÓPICOS CRÍTICOS:
    {topicos_criticos_str}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        # Formato como lista markdown
        return response.text.strip().replace('\n\n', '\n')
        
    except Exception as e:
        return f"❌ ERROR DE API: Falló la conexión con Gemini ({e}). Se usará una recomendación genérica."
 
def generar_recomendaciones_alumnos(rendimiento_df):
    """Genera recomendaciones personalizadas para alumnos con bajo rendimiento (< 70%)."""
    alumnos_bajo_rendimiento = rendimiento_df[rendimiento_df['Score_num'] < 0.70]
    alumnos_list = alumnos_bajo_rendimiento['Nombre Completo'].tolist()
    
    if not alumnos_list:
        return "No se identificaron alumnos con rendimiento inferior al 70% (Bajo Rendimiento)."
    
    recs = []
    for nombre in alumnos_list:
        score_percent = int(rendimiento_df[rendimiento_df['Nombre Completo'] == nombre]['Score_num'].iloc[0] * 100)
        recs.append(f"**{nombre} ({score_percent}%)**: Refuerzo intensivo. Revisar fichas de estudio, crear un glosario personal y practicar ejercicios de los tópicos críticos.")
        
    return "\n".join(recs)
 
# ==============================================================================
# VISUALIZACIÓN EN STREAMLIT
# ==============================================================================
 
def generate_report_pdf(acierto_por_topico, rendimiento_alumnos, topicos_criticos, docente_recs, alumnos_recs, df_acierto_pregunta):
    """Genera el PDF usando FPDF (función de respaldo para la descarga)."""
 
    class PDF(FPDF):
        # ... [El código de la clase PDF es correcto] ...
        def header(self):
            self.set_y(10)
            self.set_font('Arial', 'B', 16)
            self.cell(40) 
            self.cell(0, 10, 'REPORTE PEDAGÓGICO AVANZADO', 0, 1, 'C') 
            self.set_font('Arial', '', 11)
            self.cell(40) 
            self.cell(0, 5, 'Análisis Dinámico de Resultados de Evaluación', 0, 1, 'C')
            self.set_line_width(0.5)
            self.line(10, 28, 205, 28)
            self.ln(7)
 
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 9)
            # Verifica si se usó Gemini para adaptar el pie de página
            ai_status = "Generado por Analista Pedagógico (Gemini)" if client else "Análisis Estadístico Básico"
            self.cell(0, 10, f'{ai_status} | Página {self.page_no()}', 0, 0, 'R')
 
        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(230, 230, 230) 
            self.cell(0, 8, title, 0, 1, 'L', fill=True)
            self.ln(2)
 
        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            self.multi_cell(0, 5, body)
            self.ln(4)
 
        def print_dataframe(self, df, title, col_widths=None):
            self.chapter_title(title)
            self.set_font('Arial', 'B', 9)
            if col_widths is None:
                col_widths = [self.w / (len(df.columns) + 1)] * len(df.columns)
 
            for i, header in enumerate(df.columns):
                self.cell(col_widths[i], 7, header, 1, 0, 'C', fill=True) 
            self.ln()
 
            self.set_font('Arial', '', 9)
            for _, row in df.iterrows():
                for i, item in enumerate(row.values):
                    self.cell(col_widths[i], 7, str(item), 1, 0, 'C')
                self.ln()
            self.ln(5)
 
    pdf = PDF('P', 'mm', 'Letter')
    pdf.add_page()
    
    # Generar gráficos temporales para el PDF
    # Gráfico 1: Acierto por Tópico
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['red' if acierto < THRESHOLD_ACIERTO else 'green' for acierto in acierto_por_topico['% Acierto']]
    ax1.bar(acierto_por_topico['Tópico'], acierto_por_topico['% Acierto'], color=colors)
    ax1.set_title('Rendimiento Colectivo por Tópico')
    ax1.set_ylabel('% de Acierto Colectivo')
    ax1.set_xlabel('Tópico Inferido')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.axhline(THRESHOLD_ACIERTO, color='gray', linestyle='--', linewidth=1, label=f'Umbral Crítico ({int(THRESHOLD_ACIERTO*100)}%)')
    ax1.legend()
    plt.tight_layout()
    plot_path_1 = 'temp_plot_topicos.png'
    fig1.savefig(plot_path_1, bbox_inches='tight')
    plt.close(fig1) # CERRAR FIGURA
 
    # Gráfico 2: Distribución de Rendimiento
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    scores_percent = rendimiento_alumnos['Score_num'] * 100
    ax2.hist(scores_percent, bins=range(0, 101, 10), edgecolor='black', color='skyblue')
    ax2.axvline(70, color='red', linestyle='--', linewidth=1, label='Bajo Rendimiento (<70%)')
    ax2.set_title('Distribución de Rendimiento por Alumno')
    ax2.set_xlabel('Rendimiento Final (%)')
    ax2.set_ylabel('Número de Alumnos')
    ax2.set_xticks(range(0, 101, 10))
    ax2.legend()
    plt.tight_layout()
    plot_path_2 = 'temp_plot_distribucion.png'
    fig2.savefig(plot_path_2, bbox_inches='tight')
    plt.close(fig2) # CERRAR FIGURA
    
    # Contenido del PDF
    pdf.chapter_title('1. Resumen de Tópicos Críticos (Acierto Colectivo < 60%)')
    if topicos_criticos.empty:
        pdf.chapter_body("¡Excelente! No se identificaron tópicos con rendimiento crítico.")
    else:
        topicos_criticos_display = topicos_criticos.copy()
        topicos_criticos_display['% Acierto'] = (topicos_criticos_display['% Acierto'] * 100).astype(int).astype(str) + '%'
        pdf.print_dataframe(topicos_criticos_display, 'Tópicos Críticos (Bajo Rendimiento)', col_widths=[120, 40])
 
    pdf.add_page()
    pdf.chapter_title('2. Visualización de Rendimiento')
    
    pdf.chapter_body('Gráfico 2.1: Porcentaje de Acierto Colectivo por Tópico')
    y_g1 = pdf.get_y()
    pdf.image(plot_path_1, x=(215.9/2) - (170/2), y=y_g1, w=170) 
    pdf.set_y(y_g1 + 95) 
 
    pdf.chapter_body('Gráfico 2.2: Distribución de Rendimiento Final por Alumno')
    y_g2 = pdf.get_y()
    pdf.image(plot_path_2, x=(215.9/2) - (150/2), y=y_g2, w=150)
    pdf.set_y(y_g2 + 80)
    
    pdf.add_page() 
    pdf.chapter_title('3. Recomendaciones Docentes (Generado por Gemini)')
    pdf.chapter_body(docente_recs)
    
    pdf.chapter_title('4. Rendimiento Individual y Refuerzo Personalizado')
    rendimiento_alumnos_display = rendimiento_alumnos.copy()
    rendimiento_alumnos_display['Rendimiento Final'] = (rendimiento_alumnos_display['Score_num'] * 100).astype(int).astype(str) + '%'
    rendimiento_alumnos_display = rendimiento_alumnos_display.drop(columns=['Score_num'])
    rendimiento_alumnos_display = rendimiento_alumnos_display.rename(columns={'Nombre Completo': 'Alumno'})
    pdf.print_dataframe(rendimiento_alumnos_display, 'Tabla de Rendimiento de Alumnos', col_widths=[100, 50])
    
    pdf.chapter_title('4.1. Recomendaciones para Alumnos con Rendimiento < 70%')
    pdf.chapter_body(alumnos_recs.replace("**", "")) # Quitar negritas para FPDF
 
    # Limpieza de archivos temporales
    os.remove(plot_path_1)
    os.remove(plot_path_2)
 
    # Usar 'latin-1' si FPDF no maneja bien UTF-8 con la fuente predeterminada.
    return pdf.output(dest='S').encode('latin-1') 

def main():
    st.set_page_config(
        page_title="Analista Pedagógico Avanzado (Gemini)",
        layout="wide",
        initial_sidebar_state="expanded"
    )
 
    st.title("👨🏫 Analista Pedagógico Avanzado con Gemini")
    st.subheader("Generación de Reportes Dinámicos de Evaluación")
 
    # --- Carga de Archivo ---
    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV de resultados de examen (Plickers, etc.)",
        type=['csv'],
        key="file_uploader"
    )
 
    if uploaded_file is None:
        st.info("⬆️ Esperando la carga del archivo CSV para iniciar el análisis.")
        return
 
    # --- Procesamiento ---
    try:
        df_data, answer_key, question_cols = load_and_clean_data(uploaded_file)
    except ValueError as e:
        st.error(f"❌ Error de Carga/Formato de Datos: {e}")
        return
    except Exception as e:
        st.error(f"❌ Error crítico al procesar el archivo: {e}")
        return
        
    # Análisis y Cálculo (Incluye llamada a Gemini para tópicos)
    acierto_por_topico, rendimiento_alumnos, topicos_criticos, df_acierto_pregunta = analyze_and_calculate(df_data, answer_key, question_cols)
    
    st.success(f"✅ Análisis completado para {len(rendimiento_alumnos)} alumnos y {len(question_cols)} preguntas.")
 
    # --- Generación de Recomendaciones ---
    docente_recs = generar_recomendaciones_gemini(topicos_criticos)
    alumnos_recs = generar_recomendaciones_alumnos(rendimiento_alumnos)
 
    st.markdown("---")
    
    # ==========================================================================
    # PESTAÑAS DE VISUALIZACIÓN
    # ==========================================================================
    tab_docente, tab_visual, tab_alumnos, tab_detalle = st.tabs([
        "✅ Resumen y Recomendaciones Docentes", 
        "📊 Visualizaciones Clave", 
        "🧑🎓 Rendimiento Individual", 
        "📋 Detalle por Pregunta"
    ])
 
    with tab_docente:
        st.header("1. Tópicos Críticos (Acierto Colectivo < 60%)")
        if topicos_criticos.empty:
            st.success("🎉 ¡Excelente! No se identificaron tópicos con rendimiento crítico. No es necesario refuerzo colectivo.")
        else:
            st.warning(f"⚠️ Se identificaron **{len(topicos_criticos)} tópicos** con acierto por debajo del {int(THRESHOLD_ACIERTO*100)}% que requieren refuerzo.")
            topicos_criticos_display = topicos_criticos.copy()
            topicos_criticos_display['% Acierto'] = (topicos_criticos_display['% Acierto'] * 100).astype(int).astype(str) + '%'
            st.dataframe(topicos_criticos_display.set_index('Tópico'), use_container_width=True)
 
        st.header("2. Recomendaciones Docentes (Generado por Gemini)")
        st.markdown(docente_recs)
 
    with tab_visual:
        st.header("Gráfico 2.1: Rendimiento Colectivo por Tópico")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        colors = ['red' if acierto < THRESHOLD_ACIERTO else 'green' for acierto in acierto_por_topico['% Acierto']]
        ax1.bar(acierto_por_topico['Tópico'], acierto_por_topico['% Acierto'], color=colors)
        ax1.set_title('Rendimiento Colectivo por Tópico')
        ax1.set_ylabel('% de Acierto Colectivo')
        ax1.set_xlabel('Tópico Inferido')
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        ax1.axhline(THRESHOLD_ACIERTO, color='gray', linestyle='--', linewidth=1, label=f'Umbral Crítico ({int(THRESHOLD_ACIERTO*100)}%)')
        ax1.legend()
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1) # Asegurar cierre
 
        st.header("Gráfico 2.2: Distribución de Rendimiento por Alumno")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        scores_percent = rendimiento_alumnos['Score_num'] * 100
        ax2.hist(scores_percent, bins=range(0, 101, 10), edgecolor='black', color='skyblue')
        ax2.axvline(70, color='red', linestyle='--', linewidth=1, label='Bajo Rendimiento (<70%)')
        ax2.set_title('Distribución de Rendimiento por Alumno')
        ax2.set_xlabel('Rendimiento Final (%)')
        ax2.set_ylabel('Número de Alumnos')
        ax2.set_xticks(range(0, 101, 10))
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2) # Asegurar cierre
 
    with tab_alumnos:
        st.header("Rendimiento Individual por Alumno")
        rendimiento_alumnos_display = rendimiento_alumnos.copy()
        rendimiento_alumnos_display['Rendimiento Final'] = (rendimiento_alumnos_display['Score_num'] * 100).astype(int).astype(str) + '%'
        rendimiento_alumnos_display = rendimiento_alumnos_display.drop(columns=['Score_num'])
        rendimiento_alumnos_display = rendimiento_alumnos_display.rename(columns={'Nombre Completo': 'Alumno'})
        
        st.dataframe(rendimiento_alumnos_display.set_index('Alumno'), use_container_width=True)
 
        st.header("Refuerzo Personalizado (Rendimiento < 70%)")
        st.markdown(alumnos_recs)
 
    with tab_detalle:
        st.header("Detalle de Acierto por Pregunta")
        df_display = df_acierto_pregunta.copy()
        df_display['% Acierto'] = (df_display['% Acierto'] * 100).round(0).astype(int).astype(str) + '%'
        st.dataframe(
            df_display.sort_values(by='% Acierto', ascending=False).set_index('Tópico'), 
            use_container_width=True
        )
 
    st.markdown("---")
    # --- Descarga del PDF ---
    pdf_output = generate_report_pdf(acierto_por_topico, rendimiento_alumnos, topicos_criticos, docente_recs, alumnos_recs, df_acierto_pregunta)
    
    st.download_button(
        label="⬇️ Descargar Reporte Completo en PDF",
        data=pdf_output,
        file_name="Reporte_Pedagogico_Gemini.pdf",
        mime="application/pdf"
    )
 
if __name__ == "__main__":
    main()





