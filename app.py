import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os 
from fpdf import FPDF
import base64
from dataclasses import dataclass 
from collections import defaultdict 

warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# CONFIGURACIÓN INICIAL
# ==============================================================================
THRESHOLD_ACIERTO = 0.60 

# La API de Gemini ha sido eliminada.
gemini_status = "ℹ️ Análisis en modo **Estático (sin IA externa)**. Tópicos y recomendaciones son genéricas."


# ==============================================================================
# CLASES DE POO (HASHABLE)
# ==============================================================================

@dataclass(frozen=True) 
class Alumno:
    """Representa a un único estudiante y su rendimiento en la evaluación."""
    first_name: str
    last_name: str
    score_num: float
    question_cols: list 

    def __post_init__(self):
        # Uso de object.__setattr__ para permitir la asignación en una dataclass frozen
        object.__setattr__(self, 'nombre_completo', f"{self.first_name} {self.last_name}")
        object.__setattr__(self, 'score_percent', int(self.score_num * 100))

    def get_recomendacion_personal(self):
        """Genera una recomendación de refuerzo personalizada basada en el score."""
        
        if self.score_percent >= 70:
            return f"{self.nombre_completo} ({self.score_percent}%) tuvo un rendimiento **satisfactorio** y no requiere refuerzo intensivo."
        elif self.score_percent >= 40:
             return f"**{self.nombre_completo} ({self.score_percent}%)**: Requiere refuerzo. Enfocarse en la **revisión de la unidad** que tuvo menor rendimiento colectivo. Práctica dirigida en áreas específicas."
        else:
            return f"**{self.nombre_completo} ({self.score_percent}%)**: Refuerzo intensivo. Revisar fichas de estudio, crear un glosario personal y practicar ejercicios de los tópicos críticos."

@dataclass(frozen=True) 
class EvaluacionAnalizer:
    """Encapsula toda la lógica de análisis estadístico."""
    df_data: pd.DataFrame 
    answer_key: dict
    question_cols: list
    
    def _crear_objetos_alumnos(self):
        """Crea la lista de objetos Alumno y el DataFrame de rendimiento."""
        alumnos_list = []
        for index, row in self.df_data.iterrows():
            alumno = Alumno(
                first_name=row['First name'],
                last_name=row['Last Name'],
                score_num=row['Score_num'],
                question_cols=self.question_cols
            )
            alumnos_list.append(alumno)
        
        rendimiento_data = {
            'Nombre Completo': [a.nombre_completo for a in alumnos_list],
            'Score_num': [a.score_num for a in alumnos_list]
        }
        rendimiento_alumnos_df = pd.DataFrame(rendimiento_data).sort_values(by='Score_num')
        
        return alumnos_list, rendimiento_alumnos_df

    def _generar_recomendaciones_alumnos_interno(self, alumnos_list):
        """Genera recomendaciones individuales usando la lista de objetos Alumno."""
        recs = []
        for alumno in alumnos_list:
            if alumno.score_num < 0.70:
                recs.append(alumno.get_recomendacion_personal())
                
        if not recs:
            return "No se identificaron alumnos con rendimiento inferior al 70% (Bajo Rendimiento)."
        
        return "\n* " + "\n* ".join(recs)
    
    @st.cache_data(show_spinner=True) 
    def analizar(_self, THRESHOLD_ACIERTO):
        """Ejecuta la lógica completa de análisis de la evaluación y retorna un diccionario de resultados."""
        
        alumnos_list, rendimiento_alumnos_df = _self._crear_objetos_alumnos()
        
        st.info("📊 Realizando análisis estadístico y generando recomendaciones genéricas...")

        # 1. Tópicos (FUNCIÓN ESTÁTICA SIN IA)
        question_topics = generate_generic_topics(_self.question_cols)

        # 2. Acierto por Pregunta y Tópico
        acierto_por_pregunta = {}
        for q_col in _self.question_cols:
            aciertos = (_self.df_data[q_col] == _self.answer_key[q_col]).sum()
            total_respuestas = len(_self.df_data) 
            acierto_por_pregunta[q_col] = aciertos / total_respuestas if total_respuestas > 0 else 0

        df_acierto_pregunta = pd.DataFrame(
            list(acierto_por_pregunta.items()), 
            columns=['Pregunta', '% Acierto']
        )
        df_acierto_pregunta['Tópico'] = df_acierto_pregunta['Pregunta'].map(question_topics)

        # 3. Acierto por Tópico y Críticos
        acierto_por_topico = df_acierto_pregunta.groupby('Tópico')['% Acierto'].mean().reset_index()
        acierto_por_topico['% Acierto'] = acierto_por_topico['% Acierto'].round(2)
        acierto_por_topico = acierto_por_topico.sort_values(by='% Acierto')
        
        topicos_criticos = acierto_por_topico[acierto_por_topico['% Acierto'] < THRESHOLD_ACIERTO]
        
        # 4. Generar Recomendaciones Docentes (FUNCIÓN ESTÁTICA SIN IA)
        docente_recs = generar_generic_recomendaciones_docentes(topicos_criticos)
        alumnos_recs = _self._generar_recomendaciones_alumnos_interno(alumnos_list)
        
        return {
            'alumnos_list': alumnos_list,
            'acierto_por_topico': acierto_por_topico,
            'rendimiento_alumnos_df': rendimiento_alumnos_df,
            'topicos_criticos': topicos_criticos,
            'df_acierto_pregunta': df_acierto_pregunta,
            'docente_recs': docente_recs,
            'alumnos_recs': alumnos_recs,
        }

# ==============================================================================
# FUNCIONES ESTATICAS (REEMPLAZO DE LA LÓGICA DE IA)
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
        df = pd.read_csv(
            io.StringIO(csv_data), 
            skiprows=skip_rows_count, 
            header=0, 
            delimiter=';', 
            quotechar='"', 
            engine='python'
        )
    
    df.columns = df.columns.astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
    col_keywords = {'Score': 'Score', 'Correct': 'Correct', 'Answered': 'Answered', 
                    'Card': 'Card Number', 'First': 'First name', 'Last': 'Last Name'}
    new_column_names = {}
    for current_col in df.columns:
        found_match = False
        for keyword, standardized_name in col_keywords.items():
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
    
    answer_key_row = None
    answer_key_index = -1
    
    num_questions = len(question_cols)
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
def generate_generic_topics(question_cols):
    """Asigna tópicos genéricos y cíclicos a las preguntas basándose en su posición."""
    
    generic_topics_pool = [
        'Unidad 1: Conceptos Fundamentales',
        'Unidad 2: Aplicación y Ejercicios',
        'Unidad 3: Análisis Crítico',
        'Unidad 4: Contextualización Histórica'
    ]
    
    topic_map = defaultdict(lambda: 'Tópico General')
    
    # Asignación cíclica para distribuir el análisis
    num_topics = len(generic_topics_pool)
    for i, q_col in enumerate(question_cols):
        topic_index = i % num_topics
        topic_map[q_col] = generic_topics_pool[topic_index]
        
    return dict(topic_map)


@st.cache_data
def generar_generic_recomendaciones_docentes(topicos_criticos_df):
    """Genera recomendaciones pedagógicas genéricas basadas en los tópicos de bajo rendimiento."""
    
    if topicos_criticos_df.empty:
        return "No se encontraron tópicos con rendimiento inferior al 60%. ¡Excelente trabajo en el diseño curricular!"
    
    recomendaciones = []
    
    for index, row in topicos_criticos_df.iterrows():
        topico = row['Tópico']
        acierto = int(row['% Acierto'] * 100)
        
        if acierto < 40:
            # Muy bajo rendimiento
            rec = (
                f"**{topico} ({acierto}%)**: **Refuerzo Máximo.** Se recomienda reestructurar la lección completa. "
                "Utilice métodos de enseñanza activa (ABP o Flipped Classroom) y destine dos sesiones completas para la revisión de los fundamentos."
            )
        elif acierto < 50:
            # Bajo rendimiento
            rec = (
                f"**{topico} ({acierto}%)**: **Revisión profunda.** Ejecute ejercicios prácticos en grupo. "
                "Crear un mapa conceptual colectivo y realizar una prueba corta (quiz) de seguimiento la próxima semana."
            )
        else:
            # Rendimiento aceptable pero mejorable
            rec = (
                f"**{topico} ({acierto}%)**: **Ajuste fino.** Refuerce el tópico con material complementario o un "
                "debate/discusión guiada para consolidar el conocimiento y corregir los errores conceptuales más comunes."
            )
        recomendaciones.append(rec)
    
    return "\n".join([f"{i+1}. {r}" for i, r in enumerate(recomendaciones)])

# ==============================================================================
# VISUALIZACIÓN EN STREAMLIT y PDF (CORRECCIÓN DE ENCODING LATIN-1)
# ==============================================================================

def generate_report_pdf(acierto_por_topico, rendimiento_alumnos, topicos_criticos, docente_recs, alumnos_recs, df_acierto_pregunta):
    """Genera el PDF usando FPDF, asegurando la compatibilidad con versiones estándar."""

    class PDF(FPDF):
        def header(self):
            self.set_y(10)
            self.set_font('Arial', 'B', 16)
            self.cell(40) 
            self.cell(0, 10, 'REPORTE PEDAGÓGICO AVANZADO'.encode('latin-1', 'replace').decode('latin-1'), 0, 1, 'C') 
            self.set_font('Arial', '', 11)
            self.cell(40) 
            self.cell(0, 5, 'Análisis Dinámico de Resultados de Evaluación'.encode('latin-1', 'replace').decode('latin-1'), 0, 1, 'C')
            self.set_line_width(0.5)
            self.line(10, 28, 205, 28)
            self.ln(7)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 9)
            ai_status = "Análisis Estadístico Avanzado y Reglas de Negocio"

            self.cell(0, 10, f'{ai_status} | Página {self.page_no()}'.encode('latin-1', 'replace').decode('latin-1'), 0, 0, 'R')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(230, 230, 230) 
            self.cell(0, 8, title.encode('latin-1', 'replace').decode('latin-1'), 0, 1, 'L', fill=True)
            self.ln(2)

        def chapter_body(self, body):
            self.set_font('Arial', '', 10)
            # Reemplazar **negritas** y *cursivas* antes de codificar
            body_cleaned = body.replace('**', '').replace('* ', '').replace('*', '') 
            # Codificar/Decodificar para asegurar el paso correcto de strings Unicode a FPDF
            self.multi_cell(0, 5, body_cleaned.encode('latin-1', 'replace').decode('latin-1'))
            self.ln(4)

        def print_dataframe(self, df, title, col_widths=None):
            self.chapter_title(title)
            self.set_font('Arial', 'B', 9)
            if col_widths is None:
                col_widths = [self.w / (len(df.columns) + 1)] * len(df.columns)

            for i, header in enumerate(df.columns):
                header_text = header.encode('latin-1', 'replace').decode('latin-1')
                self.cell(col_widths[i], 7, header_text, 1, 0, 'C', fill=True) 
            self.ln()

            self.set_font('Arial', '', 9)
            for _, row in df.iterrows():
                for i, item in enumerate(row.values):
                    item_text = str(item).encode('latin-1', 'replace').decode('latin-1')
                    self.cell(col_widths[i], 7, item_text, 1, 0, 'C')
                self.ln()
            self.ln(5)

    pdf = PDF('P', 'mm', 'Letter')
    pdf.add_page()
    
    # Generar gráficos temporales para el PDF
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
    plt.close(fig1) 

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
    plt.close(fig2) 
    
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
    
    pdf.chapter_title('3. Recomendaciones Docentes (Generadas por Reglas de Negocio)')
    pdf.chapter_body(docente_recs)
    
    pdf.chapter_title('4. Rendimiento Individual y Refuerzo Personalizado')
    rendimiento_alumnos_display = rendimiento_alumnos.copy()
    rendimiento_alumnos_display['Rendimiento Final'] = (rendimiento_alumnos_display['Score_num'] * 100).astype(int).astype(str) + '%'
    rendimiento_alumnos_display = rendimiento_alumnos_display.drop(columns=['Score_num'])
    rendimiento_alumnos_display = rendimiento_alumnos_display.rename(columns={'Nombre Completo': 'Alumno'})
    pdf.print_dataframe(rendimiento_alumnos_display, 'Tabla de Rendimiento de Alumnos', col_widths=[100, 50])
    
    pdf.chapter_title('4.1. Recomendaciones para Alumnos con Rendimiento < 70%')
    pdf.chapter_body(alumnos_recs) 

    # Limpieza de archivos temporales
    os.remove(plot_path_1)
    os.remove(plot_path_2)

    return pdf.output(dest='S').encode('latin-1') 

# ==============================================================================
# FUNCIÓN PRINCIPAL 
# ==============================================================================

def main():
    
    st.set_page_config(
        page_title="Analista Pedagógico Avanzado (Estatístico)",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("👨‍🏫 Analista Pedagógico Avanzado (Modo Estático)")
    st.subheader("Generación de Reportes Dinámicos de Evaluación sin IA Externa")
    
    st.info(gemini_status)

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
        
    # Inicializar el objeto Analizer (solo con datos hashable)
    analizer_instance = EvaluacionAnalizer(df_data, answer_key, question_cols)

    # Llamar al método cacheado para obtener los resultados
    analisis_resultados = analizer_instance.analizar(THRESHOLD_ACIERTO) 
    
    # Desempacar resultados
    alumnos_list = analisis_resultados['alumnos_list']
    acierto_por_topico = analisis_resultados['acierto_por_topico']
    rendimiento_alumnos_df = analisis_resultados['rendimiento_alumnos_df']
    topicos_criticos = analisis_resultados['topicos_criticos']
    df_acierto_pregunta = analisis_resultados['df_acierto_pregunta']
    docente_recs = analisis_resultados['docente_recs']
    alumnos_recs = analisis_resultados['alumnos_recs']
    
    st.success(f"✅ Análisis completado para {len(alumnos_list)} alumnos y {len(analizer_instance.question_cols)} preguntas.")

    st.markdown("---")
    
    # ==========================================================================
    # PESTAÑAS DE VISUALIZACIÓN 
    # ==========================================================================
    tab_docente, tab_visual, tab_alumnos, tab_detalle = st.tabs([
        "✅ Resumen y Recomendaciones Docentes", 
        "📊 Visualizaciones Clave", 
        "🧑‍🎓 Rendimiento Individual", 
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

        st.header("2. Recomendaciones Docentes (Generadas por Reglas de Negocio)")
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
        plt.close(fig1) 

        st.header("Gráfico 2.2: Distribución de Rendimiento por Alumno")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        scores_percent = rendimiento_alumnos_df['Score_num'] * 100
        ax2.hist(scores_percent, bins=range(0, 101, 10), edgecolor='black', color='skyblue')
        ax2.axvline(70, color='red', linestyle='--', linewidth=1, label='Bajo Rendimiento (<70%)')
        ax2.set_title('Distribución de Rendimiento por Alumno')
        ax2.set_xlabel('Rendimiento Final (%)')
        ax2.set_ylabel('Número de Alumnos')
        ax2.set_xticks(range(0, 101, 10))
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2) 

    with tab_alumnos:
        st.header("Rendimiento Individual por Alumno")
        rendimiento_alumnos_display = rendimiento_alumnos_df.copy()
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
    pdf_output = generate_report_pdf(
        acierto_por_topico, 
        rendimiento_alumnos_df, 
        topicos_criticos, 
        docente_recs, 
        alumnos_recs, 
        df_acierto_pregunta
    )
    
    st.download_button(
        label="⬇️ Descargar Reporte Completo en PDF",
        data=pdf_output,
        file_name="Reporte_Pedagogico_Estatistico.pdf",
        mime="application/pdf"
    )

if __name__ == "__main__":
    main()

