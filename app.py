import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
import json
import time
import traceback
from fpdf import FPDF
from io import BytesIO
import tempfile

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------
# Comprobar disponibilidad de Gemini (importar de forma segura)
# -------------------------------------------------------------------
USE_GEMINI = False
client = None
try:
    from google import genai
    from google.genai.errors import APIError
    # Solo habilitamos más adelante si hay clave
    if os.environ.get("GEMINI_API_KEY"):
        try:
            client = genai.Client()
            USE_GEMINI = True
        except Exception:
            # No hay cliente: nos quedamos en modo offline
            USE_GEMINI = False
    else:
        USE_GEMINI = False
except Exception:
    USE_GEMINI = False

# Parámetros
THRESHOLD_ACIERTO = 0.60  # umbral para considerar tópico crítico

# -------------------------------------------------------------------
# Funciones de lectura y limpieza
# -------------------------------------------------------------------
def read_raw_text(uploaded_file):
    """Leer bytes y devolver texto tratando varias codificaciones"""
    raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    for enc in ("utf-8", "utf-8-sig", "latin-1", "utf-16"):
        try:
            text = raw_bytes.decode(enc)
            return text
        except Exception:
            continue
    # fallback: pandas intentar directamente
    return None

def load_and_clean_data(csv_file_content):
    """
    Carga, detecta encabezado y answer key. Retorna:
    df (solo registros de alumnos), answer_key dict, question_cols list
    """
    text = read_raw_text(csv_file_content)
    if text is None:
        # intentar con pandas directamente
        csv_file_content.seek(0)
        raw_df = pd.read_csv(csv_file_content, header=None, dtype=str, on_bad_lines='skip')
        lines = raw_df.astype(str).agg(' '.join, axis=1).tolist()
    else:
        lines = text.splitlines()

    header_idx = -1
    for i, line in enumerate(lines):
        low = str(line).lower()
        if 'card number' in low or 'first name' in low or 'first_name' in low or 'nombre' in low:
            header_idx = i
            break
    if header_idx == -1:
        raise ValueError("No se pudo identificar la fila de encabezado (buscando 'Card Number' / 'First name' / 'Nombre').")

    # Leer con pandas saltando filas previas al encabezado detectado
    try:
        if text is not None:
            df = pd.read_csv(io.StringIO(text), skiprows=header_idx, header=0, dtype=str, on_bad_lines='skip')
        else:
            csv_file_content.seek(0)
            df = pd.read_csv(csv_file_content, skiprows=header_idx, header=0, dtype=str, on_bad_lines='skip')
    except Exception:
        # intentar con punto y coma
        if text is not None:
            df = pd.read_csv(io.StringIO(text), skiprows=header_idx, header=0, dtype=str, sep=';', on_bad_lines='skip')
        else:
            csv_file_content.seek(0)
            df = pd.read_csv(csv_file_content, skiprows=header_idx, header=0, dtype=str, sep=';', on_bad_lines='skip')

    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    df = df.dropna(how='all').reset_index(drop=True)

    # Normalizar nombres de columnas posibles
    col_keywords = {
        'score': ['score', 'puntaje', 'resultado'],
        'correct': ['correct', 'aciertos', 'correctas'],
        'answered': ['answered', 'respondidas', 'contestadas'],
        'first_name': ['first name', 'first_name', 'nombre'],
        'last_name': ['last name', 'last_name', 'apellido'],
        'card': ['card number', 'card_number', 'tarjeta', 'número de tarjeta']
    }
    canonical = {}
    lc = {c.lower(): c for c in df.columns}
    for key, kws in col_keywords.items():
        for kw in kws:
            matches = [orig for low, orig in lc.items() if kw in low]
            if matches:
                canonical[key] = matches[0]
                break

    # required: at least first_name and score or correct/answered
    if ('first_name' not in canonical) or (('score' not in canonical) and not ('correct' in canonical and 'answered' in canonical)):
        # show available columns for debugging
        raise ValueError(f"Columnas esenciales no encontradas. Columnas disponibles: {list(df.columns)}")

    # Determine question columns as those not canonical
    used_cols = set(canonical.values())
    question_cols = [c for c in df.columns if c not in used_cols]

    # Try to detect answer key row: a row where many question-cols contain single-letter A-D
    answer_key_row = None
    answer_key_index = None
    num_questions = len(question_cols)
    threshold = max(3, int(num_questions / 2)) if num_questions>0 else 1

    for idx in range(len(df)):
        row = df.loc[idx, question_cols].astype(str).fillna('').str.upper().str.strip()
        count_letters = row.isin(['A','B','C','D']).sum()
        if count_letters >= threshold:
            answer_key_row = row
            answer_key_index = idx
            break

    if answer_key_row is None:
        # si no se detecta, intentamos buscar en las primeras 6 filas por patrón
        for idx in range(min(6, len(df))):
            row = df.loc[idx, question_cols].astype(str).fillna('').str.upper().str.strip()
            if row.isin(['A','B','C','D']).sum() >= threshold:
                answer_key_row = row
                answer_key_index = idx
                break

    if answer_key_row is None:
        raise ValueError("No se pudo detectar la fila de respuestas correctas (Answer Key).")

    # Construir answer_key dict
    answer_key = {q: str(answer_key_row[q]).strip().upper() for q in question_cols}

    # Drop the answer_key row from df and keep only student rows (a bit heuristic: remove if equals index)
    df_students = df.drop(index=answer_key_index).reset_index(drop=True)

    # Normalize student columns and numeric conversions
    # Score -> number 0-1
    if 'score' in canonical:
        scol = canonical['score']
        df_students[scol] = df_students[scol].astype(str).str.replace('%','').str.replace(',','').str.strip()
        df_students['Score_num'] = pd.to_numeric(df_students[scol], errors='coerce') / 100.0
    else:
        df_students['Score_num'] = np.nan

    # Convert question responses to single uppercase letters
    for q in question_cols:
        df_students[q] = df_students[q].astype(str).fillna('').str.strip().str.upper().str[:1]

    # Drop rows where Score_num is NaN and there are no answers? We keep rows that have at least one non-empty answer
    has_any_answer = df_students[question_cols].apply(lambda r: r.astype(str).str.strip().replace('','').astype(bool).any(), axis=1)
    df_students = df_students[has_any_answer].reset_index(drop=True)

    return df_students, answer_key, question_cols, canonical

# -------------------------------------------------------------------
# Generación de tópicos OFFLINE y con Gemini (si está habilitado)
# -------------------------------------------------------------------
def topic_from_question_text(qtext):
    q = str(qtext).lower()
    keywords = {
        "Marco teórico": ["marco teóric", "antecedent", "bibliogra", "referenc"],
        "Método científico": ["métod", "observación", "hipótes", "experimental", "método"],
        "Estadística": ["media", "mediana", "frecuencia", "estadíst"],
        "Variables": ["variable", "independient", "dependient"],
        "Programación": ["bucle", "for", "while", "función", "variable"],
        "Ética": ["ética", "consentimient", "responsab", "ético"]
    }
    for topic, keys in keywords.items():
        for k in keys:
            if k in q:
                return topic
    # fallback short text
    return qtext[:60]

def generate_topics_offline(question_cols):
    return {q: topic_from_question_text(q) for q in question_cols}

def generate_topics_with_gemini_safe(question_cols):
    """
    Intentar generar tópicos con Gemini cuando esté disponible, si falla usar offline.
    """
    if not USE_GEMINI or client is None:
        return generate_topics_offline(question_cols)

    try:
        # Construir prompt simple (mantener respuesta como JSON)
        qlist = "\n".join([f"- {q}" for q in question_cols])
        prompt = f"Asignar un tópico corto (máx 4 palabras) a cada una de las siguientes preguntas. Devuelve un JSON con clave=texto pregunta, valor=tópico.\n\n{qlist}"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        text = response.text.strip()
        # intento de extraer JSON
        try:
            # limpiar posibles fences
            if text.startswith("```"):
                text = text.split("```",2)[-1]
            topic_map = json.loads(text)
            # map missing to offline
            for q in question_cols:
                if q not in topic_map:
                    topic_map[q] = topic_from_question_text(q)
            return topic_map
        except Exception:
            # si no parsea JSON, fallback a heurística offline
            return generate_topics_offline(question_cols)
    except Exception:
        return generate_topics_offline(question_cols)

# -------------------------------------------------------------------
# Cálculos de acierto y diagnóstico temático
# -------------------------------------------------------------------
def compute_question_stats(df_students, question_cols, answer_key):
    rows = []
    total_students = len(df_students)
    for q in question_cols:
        # contar aciertos exactos respecto de la key
        key = str(answer_key.get(q,"")).upper()
        if key in ['A','B','C','D']:
            corrects = (df_students[q] == key).sum()
            pct = corrects / total_students if total_students>0 else 0
            # distribuciones
            freqs = df_students[q].fillna('').value_counts().to_dict()
            entropy = 0.0
            total_nonblank = sum([v for k,v in freqs.items() if k!=''])
            for v in freqs.values():
                if total_nonblank>0:
                    p = v/total_nonblank
                    if p>0: entropy -= p*np.log2(p)
            rows.append({
                "Question": q,
                "PctCorrect": round(pct,4),
                "MostFreq": max(freqs, key=freqs.get) if freqs else "",
                "MostFreqCount": freqs.get(max(freqs, key=freqs.get)) if freqs else 0,
                "Entropy": round(entropy,3)
            })
        else:
            # si no hay key, inferimos por concordancia
            freqs = df_students[q].fillna('').value_counts()
            most = freqs.index[0] if not freqs.empty else ""
            pct = (freqs.iloc[0]/ len(df_students)) if (not freqs.empty and len(df_students)>0) else 0
            entropy = 0.0
            total_nonblank = sum(freqs)
            for v in freqs:
                if total_nonblank>0:
                    p = v/total_nonblank
                    if p>0: entropy -= p*np.log2(p)
            rows.append({
                "Question": q,
                "PctCorrect": round(pct,4),
                "MostFreq": most,
                "MostFreqCount": int(freqs.iloc[0]) if not freqs.empty else 0,
                "Entropy": round(entropy,3)
            })
    qstats = pd.DataFrame(rows).sort_values(by="PctCorrect")
    return qstats

def generate_offline_diagnostics(df_students, qstats, topics_map):
    diagnostics = []
    crit = qstats[(qstats['PctCorrect'] < THRESHOLD_ACIERTO) | (qstats['Entropy'] > 1.3)]
    for _, r in crit.iterrows():
        q = r['Question']
        topic = topics_map.get(q, topic_from_question_text(q))
        diagnosis = (
            f"Dificultad detectada en '{q[:120]}' (tópico estimado: {topic}). "
            f"Distribución: {r['MostFreq']} ({r['MostFreqCount']} respuestas)."
        )
        rec_teacher = f"Refuerce {topic} con ejemplos concretos, ejercicios guiados y evaluación formativa breve."
        rec_student = f"Repase conceptos de {topic} con ejercicios y tarjetas de estudio; pida retroalimentación al docente."
        diagnostics.append({
            "Question": q,
            "Topic": topic,
            "Diagnosis": diagnosis,
            "TeacherRec": rec_teacher,
            "StudentRec": rec_student,
            "PctCorrect": r['PctCorrect'],
            "Entropy": r['Entropy']
        })
    return diagnostics

# -------------------------------------------------------------------
# Generación PDF/Excel (sin errores)
# -------------------------------------------------------------------
def make_pdf_bytes(asignatura, group_summary, qstats, diagnostics, rendimiento_display, doc_recs, alumnos_recs):
    # usar FPDF para construir PDF similar al original, pero con menor complejidad para evitar problemas de encoding
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, f"Reporte Pedagógico - {asignatura}", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", "", 10)
    for k,v in group_summary.items():
        pdf.cell(0,6, f"{k}: {v}", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,6, "Tópicos cr ́ıticos (resumen)", ln=True)
    pdf.set_font("Arial", "", 10)
    if len(diagnostics)==0:
        pdf.multi_cell(0,5, "No se detectaron tópicos críticos según las heurísticas locales.")
    else:
        for d in diagnostics:
            pdf.multi_cell(0,5, f"- {d['Topic']}: {d['Diagnosis']}")
            pdf.multi_cell(0,5, f"  Recomendación docente: {d['TeacherRec']}")
            pdf.ln(1)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,6, "Recomendaciones docentes (resumen)", ln=True)
    pdf.set_font("Arial", "", 10)
    if isinstance(doc_recs, str):
        pdf.multi_cell(0,5, doc_recs)
    else:
        pdf.multi_cell(0,5, str(doc_recs))
    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,6, "Recomendaciones para alumnos de bajo rendimiento", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0,5, alumnos_recs)
    pdf.ln(6)
    # Añadir tabla pequeña de rendimiento (primeras 40 filas)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0,6, "Rendimiento individual (muestra)", ln=True)
    pdf.set_font("Arial", "", 9)
    for _, r in rendimiento_display.head(40).iterrows():
        pdf.cell(0,5, f"- {r.get('Nombre Completo','')}: {r.get('Rendimiento Final','')}", ln=True)
    return pdf.output(dest='S').encode('latin-1', errors='ignore')

def make_excel_bytes(proc_df, qstats, diagnostics):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        proc_df.to_excel(writer, sheet_name="Processed", index=False)
        qstats.to_excel(writer, sheet_name="QuestionStats", index=False)
        pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)
    out.seek(0)
    return out.getvalue()

# -------------------------------------------------------------------
# Interfaz principal
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Analizador Plickers IA (robusto)", layout="wide")
    st.title("Analizador Plickers IA (modo robusto)")

    st.write("Suba el CSV exportado por Plickers (sin editar). El sistema procesa el archivo, detecta la clave de respuestas, calcula aciertos y genera diagnósticos temáticos. Funciona sin conexión a servicios externos; si tiene GEMINI configurado, intentará mejorar los tópicos.")

    uploaded_file = st.file_uploader("Archivo CSV (exportado por Plickers)", type=["csv"])
    asignatura = st.text_input("Nombre de la asignatura (para el reporte)")

    if uploaded_file is None:
        return

    if not asignatura:
        st.info("Indique el nombre de la asignatura para generar el reporte (campo obligatorio).")
        return

    # Procesar
    try:
        proc_res = load_and_clean_data(uploaded_file)
        # load_and_clean_data defined earlier (keeps caching)
        df_students, answer_key, question_cols, canonical = load_and_clean_data(uploaded_file)
    except Exception as e:
        st.error(f"Error al procesar archivo: {e}")
        return

    st.success(f"Archivo procesado: {len(df_students)} registros de alumnos; {len(question_cols)} preguntas detectadas.")
    # Temas con Gemini fallback offline
    topics_map = generate_topics_with_gemini_safe(question_cols)

    qstats = compute_question_stats(df_students, question_cols, answer_key)
    diagnostics = generate_offline_diagnostics(df_students, qstats, topics_map)

    # Rendimiento por alumno
    if 'Score_num' in df_students.columns and df_students['Score_num'].notna().any():
        df_students['Score_num'] = pd.to_numeric(df_students['Score_num'], errors='coerce').fillna(0)
        df_students['Rendimiento Final'] = (df_students['Score_num'] * 100).round(0).astype(int).astype(str) + '%'
    else:
        df_students['Rendimiento Final'] = 'N/A'
    # Nombre completo
    fname = canonical.get('first_name') if canonical.get('first_name') else (canonical.get('first') if canonical.get('first') else None)
    lname = canonical.get('last_name') if canonical.get('last_name') else (canonical.get('last') if canonical.get('last') else None)
    if fname and lname:
        df_students['Nombre Completo'] = df_students[fname].fillna('') + ' ' + df_students[lname].fillna('')
    else:
        df_students['Nombre Completo'] = df_students.index.astype(str)

    # Resumen grupal
    avg_percent = df_students['Score_num'].mean() if 'Score_num' in df_students.columns else np.nan
    group_summary = {
        "NumAlumnos": int(len(df_students)),
        "AvgPercent": f"{round(avg_percent*100,2)}%" if not np.isnan(avg_percent) else "N/A"
    }

    # Recomendaciones para docente
    # Primero intentamos generar con Gemini si está habilitado (falla con fallback)
    docente_recs = None
    if USE_GEMINI:
        try:
            # construir prompt breve
            top_crit = qstats[qstats['PctCorrect'] < THRESHOLD_ACIERTO].head(10)
            if not top_crit.empty:
                prompt = "Genera recomendaciones docentes breves (lista numerada) para los siguientes tópicos críticos:\n"
                for _, r in top_crit.iterrows():
                    prompt += f"- {r['Question'][:200]}\n"
                if client:
                    resp = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                    docente_recs = resp.text.strip()
        except Exception:
            docente_recs = None

    if not docente_recs:
        # fallback local
        if len(diagnostics)==0:
            docente_recs = "No se detectaron tópicos críticos. Mantener planificación actual y elevar nivel de desafío."
        else:
            lines = []
            for d in diagnostics:
                lines.append(f"- {d['Topic']}: {d['TeacherRec']}")
            docente_recs = "\n".join(lines)

    # Recomendaciones por alumno
    alumnos_recs = generar_recomendaciones_alumnos(df_students[['Nombre Completo', 'Score_num']].rename(columns={'Score_num':'Score_num'})) if True else ""

    # Mostrar pestañas con resultados
    t1, t2, t3, t4 = st.tabs(["Resumen y Recomendaciones","Visualizaciones","Rendimiento Individual","Detalle Preguntas"])
    with t1:
        st.header("Tópicos Críticos")
        if len(diagnostics)==0:
            st.success("No se detectaron tópicos críticos por debajo del umbral.")
        else:
            st.dataframe(pd.DataFrame(diagnostics)[['Topic','Question','PctCorrect','Entropy']].sort_values(by='PctCorrect'))
        st.header("Recomendaciones Docente")
        st.text(docente_recs)

    with t2:
        st.header("Rendimiento colectivo por tópico (gráfico)")
        # preparar dataframe acierto_por_topico
        acierto_por_topico = pd.DataFrame([
            {"Tópico": topics_map.get(row['Question'], row['Question'][:40]), "% Acierto": row['PctCorrect']}
            for _, row in qstats.iterrows()
        ])
        if not acierto_por_topico.empty:
            fig, ax = plt.subplots(figsize=(10,5))
            colors = ['red' if v < THRESHOLD_ACIERTO else 'green' for v in acierto_por_topico['% Acierto']]
            ax.bar(acierto_por_topico['Tópico'], acierto_por_topico['% Acierto'], color=colors)
            ax.set_ylabel('% Acierto')
            ax.set_xticklabels(acierto_por_topico['Tópico'], rotation=45, ha='right', fontsize=8)
            st.pyplot(fig)
        st.header("Distribución de rendimiento por alumno")
        if 'Score_num' in df_students.columns:
            fig2, ax2 = plt.subplots(figsize=(8,4))
            ax2.hist(df_students['Score_num'].dropna()*100, bins=10, color='skyblue', edgecolor='black')
            ax2.set_xlabel('Rendimiento (%)')
            st.pyplot(fig2)

    with t3:
        st.header("Rendimiento Individual")
        display_rend = df_students[['Nombre Completo','Rendimiento Final','Score_num']].sort_values(by='Score_num', ascending=True)
        st.dataframe(display_rend)

        st.header("Recomendaciones para alumnos (<70%)")
        st.markdown(alumnos_recs)

    with t4:
        st.header("Detalle por pregunta")
        df_qdisplay = qstats.copy()
        df_qdisplay['PctCorrect'] = (df_qdisplay['PctCorrect']*100).round(1).astype(str)+'%'
        st.dataframe(df_qdisplay[['Question','PctCorrect','MostFreq','MostFreqCount','Entropy']])

    # Descargas
    pdf_bytes = make_pdf_bytes(asignatura, group_summary, qstats, diagnostics, display_rend, docente_recs, alumnos_recs)
    st.download_button("Descargar Reporte PDF", data=pdf_bytes, file_name=f"Reporte_{asignatura}.pdf", mime="application/pdf")

    excel_bytes = make_excel_bytes(df_students, qstats, diagnostics)
    st.download_button("Descargar Reporte Excel", data=excel_bytes, file_name=f"Reporte_{asignatura}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()




