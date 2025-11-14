# app.py
# Analizador Plickers - Versión robusta offline + reintento de IA externa
# Requisitos: streamlit, pandas, openpyxl, reportlab, requests (opcional)
# Guardar como app.py y ejecutar con: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
import traceback

# Opcional: requests se usa solo si habilitas la consulta a un servicio externo
try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

st.set_page_config(page_title="Analizador Plickers IA", layout="wide")
st.title("Analizador Plickers — Diagnóstico y Rediseño Pedagógico")

# ---------------------------
# Utilidades para lectura robusta
# ---------------------------
def read_raw_csv_any(ufile_bytes):
    """
    Lee un CSV exportado por Plickers que puede incluir filas de metadatos antes del encabezado.
    Devuelve: raw pandas DataFrame leído sin header (header=None), y el decoded text.
    """
    # ufile_bytes may be a UploadedFile or bytes
    if hasattr(ufile_bytes, "read"):
        raw_bytes = ufile_bytes.read()
    else:
        raw_bytes = ufile_bytes
    # Try common encodings
    for enc in ("utf-8", "utf-8-sig", "latin-1", "utf-16"):
        try:
            text = raw_bytes.decode(enc)
            break
        except Exception:
            text = None
    if text is None:
        # fallback binary -> try pandas directly
        raw = pd.read_csv(BytesIO(raw_bytes), header=None, dtype=str, on_bad_lines='skip')
        return raw, ""
    raw = pd.read_csv(StringIO(text), header=None, dtype=str, on_bad_lines='skip')
    return raw, text

def find_header_index(raw_df):
    """
    Buscamos la fila con encabezado real (Card Number / Número de tarjeta / First name / Nombre).
    Devolvemos index de la fila encontrada (0-based). Si no aparece, retornamos None.
    """
    header_keywords = [
        "card number", "número de tarjeta", "número", "tarjeta",
        "first name", "first_name", "nombre", "nombre(s)",
        "last name", "apellido", "apellido(s)",
        "score", "puntaje", "resultado"
    ]
    for i, row in raw_df.iterrows():
        row_text = " ".join([str(x).lower() for x in row.fillna("").values])
        for kw in header_keywords:
            if kw in row_text:
                return i
    return None

def load_processed_df(uploaded_file):
    """
    Retorna dataframe procesado con columnas limpias y preguntas detectadas.
    Si no puede procesar, devuelve (None, mensaje_error).
    """
    try:
        raw_df, raw_text = read_raw_csv_any(uploaded_file)
        hdr_idx = find_header_index(raw_df)
        if hdr_idx is None:
            return None, "No se pudo encontrar la fila de encabezado en el CSV."
        # read again skipping rows before header, letting pandas infer header
        # Use the decoded text if available for accurate reading with special chars
        if raw_text:
            df = pd.read_csv(StringIO(raw_text), skiprows=hdr_idx, dtype=str, on_bad_lines='skip')
        else:
            # fallback use bytes
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, skiprows=hdr_idx, dtype=str, on_bad_lines='skip')
        df = df.dropna(how='all')  # drop completely empty rows
        # clean column names
        df.columns = [str(c).strip() for c in df.columns]
        # normalize common column names to canonical names (case-insensitive)
        cols_lower = {c.lower(): c for c in df.columns}
        canonical = {}
        # mapping possibilities
        mapping_candidates = {
            "card_number": ["card number", "card_number", "card no", "card"],
            "first_name": ["first name", "first_name", "nombre"],
            "last_name": ["last name", "last_name", "apellido"],
            "score": ["score", "puntaje", "resultado"],
            "correct": ["correct", "aciertos", "correctas"],
            "answered": ["answered", "respondidas", "contestadas"]
        }
        for canon, keys in mapping_candidates.items():
            for k in keys:
                for col_lower, col_orig in cols_lower.items():
                    if k in col_lower:
                        canonical[canon] = col_orig
                        break
                if canon in canonical:
                    break
        # If not all base columns found, we still continue but will warn later
        # Determine question columns: those not in canonical values
        used_cols = set(canonical.values())
        question_cols = [c for c in df.columns if c not in used_cols]
        # Build processed df with at least First name, Last name, Score, Correct, Answered
        proc = df.copy()
        # If 'Score' exists but is like '80%' convert to numeric
        if canonical.get("score"):
            scol = canonical["score"]
            proc[scol] = proc[scol].astype(str).str.replace("%","").str.replace(",","").str.strip()
            proc[scol] = pd.to_numeric(proc[scol], errors='coerce')
        if canonical.get("correct"):
            proc[canonical["correct"]] = pd.to_numeric(proc[canonical["correct"]].astype(str).str.replace("%",""), errors='coerce')
        if canonical.get("answered"):
            proc[canonical["answered"]] = pd.to_numeric(proc[canonical["answered"]].astype(str).str.replace("%",""), errors='coerce').replace(0, np.nan)
        # Return tuple
        meta = {
            "canonical": canonical,
            "question_cols": question_cols
        }
        return (proc, meta), None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"Error leyendo el CSV: {e}\n{tb}"

# ---------------------------
# Heurísticas locales para diagnóstico temático
# ---------------------------
def compute_question_stats(proc_df, question_cols, canonical):
    """
    Calcula % aciertos por pregunta y detecta opción correcta si hay fila de correct answers.
    Retorna DataFrame con stats por pregunta.
    """
    rows = []
    # If there is a row that likely is 'correct answers' (row values are single letters A-D), try to detect it:
    # But Plickers sometimes outputs correct answers in a separate row before student rows; our load skips header rows,
    # so we rely on comparing student answers to a detected correct row if present by pattern.
    # For each question column, compute distribution and percent blank, percent most freq etc.
    for q in question_cols:
        vals = proc_df[q].astype(str).fillna("").str.strip()
        total = len(vals[vals != ""])
        freqs = vals[vals != ""].value_counts()
        most_freq = freqs.index[0] if not freqs.empty else ""
        pct_most = (freqs.iloc[0]/total*100) if (not freqs.empty and total>0) else 0
        # guess percent correct: if there is a 'Correct' column per student this is different; we'll use per-question majority correctness if possible
        # If proc_df has 'Correct' column aggregated by question? Plickers original export doesn't label per-question correct; students have overall Correct count.
        # We instead estimate difficulty as dispersion and majority wrong (if majority answer != expected)
        # For offline heuristics we compute entropy-like measure
        entropy = 0.0
        for v,cnt in freqs.items():
            p = cnt/total if total>0 else 0
            if p>0:
                entropy -= p * np.log2(p)
        rows.append({
            "Question": q,
            "TotalAnswered": int(total),
            "MostFrequentAnswer": most_freq,
            "MostFreqPct": round(pct_most,2),
            "Entropy": round(entropy,3),
            "TopCounts": dict(freqs.head(5).to_dict())
        })
    dfq = pd.DataFrame(rows).sort_values(by="MostFreqPct", ascending=True)  # lower most-freq% often means dispersed/wrong
    return dfq

def topic_from_question_text(qtext):
    """
    Heurística básica para inferir tópico a partir del texto de la pregunta.
    Busca palabras clave para agrupar por temas generales.
    """
    q = str(qtext).lower()
    keywords = {
        "metodología": ["marco teóric", "hipótes", "métod", "observación", "investig"],
        "estadística": ["media", "mediana", "desviación", "estadíst", "porcentaje", "frecuencia"],
        "matemáticas": ["factoriz", "ecuaci", "operaci", "algebra", "fraccion"],
        "programación": ["variable", "función", "bucle", "if", "for", "while"],
        "contexto": ["paec", "contexto", "aplicación"],
        "ética": ["ético", "ética", "consentimient", "responsab"]
    }
    for topic, keys in keywords.items():
        for k in keys:
            if k in q:
                return topic.capitalize()
    # fallback: return short preview
    return (qtext[:60] + "...") if len(qtext)>60 else qtext

def generate_offline_diagnostics(proc_df, meta):
    """
    Genera diagnósticos temáticos y micro-sesiones sin conexión.
    """
    question_cols = meta["question_cols"]
    canonical = meta["canonical"]
    qstats = compute_question_stats(proc_df, question_cols, canonical)
    # Identify problematic questions: low MostFreqPct or high entropy
    # We'll mark as problematic if MostFreqPct < 60 or Entropy > 1.3
    problematic = qstats[(qstats["MostFreqPct"] < 60) | (qstats["Entropy"] > 1.3)].copy()
    diagnostics = []
    for _, row in problematic.iterrows():
        q = row["Question"]
        topic = topic_from_question_text(q)
        diagnosis_text = f"Pregunta: {q}. Tema estimado: {topic}. Distribución: {row['TopCounts']}. " \
                         f"Interpretación: alta dispersión de respuestas y baja concordancia; sugiere dificultad en {topic}."
        # Heuristic recommendations by topic
        rec_teacher = f"Reforzar {topic} con ejemplos contextualizados, actividades guiadas y ejercicios de práctica."
        rec_student = f"Repasa conceptos clave de {topic} y realiza ejercicios prácticos con retroalimentación inmediata."
        micro_session = {
            "Topic": topic,
            "Question": q,
            "Diagnosis": diagnosis_text,
            "TeacherRec": rec_teacher,
            "StudentRec": rec_student,
            "Priority": (row["MostFreqPct"], row["Entropy"])
        }
        diagnostics.append(micro_session)
    return qstats, diagnostics

# ---------------------------
# (Opcional) Reintentos para IA externa
# ---------------------------
def try_external_ai_enhancement(prompt_text, max_retries=3, wait_seconds=3):
    """
    Función de reintento para una consulta a un servicio IA externo.
    En esta versión es un stub: si no configuras tu servicio, siempre fallará y
    retornará None para que la app use la versión offline.
    Si quieres conectar un servicio real, edita este bloque para hacer la petición
    (con autenticación) y devolver el texto generado.
    """
    # Si requests no está disponible, saltamos directamente
    if not HAS_REQUESTS:
        return None, "requests library not available"
    # Example stub: (replace URL / headers with your provider)
    # url = "https://api.example.com/generate"
    # headers = {"Authorization": "Bearer YOUR_KEY", "Content-Type": "application/json"}
    # payload = {"prompt": prompt_text}
    last_err = None
    for attempt in range(1, max_retries+1):
        try:
            # Aquí iría la llamada real. Por seguridad por defecto no se llama.
            # response = requests.post(url, headers=headers, json=payload, timeout=15)
            # if response.status_code == 200:
            #     return response.json().get("text"), None
            # else:
            #     last_err = f"status {response.status_code}"
            # Simulamos fallo (por defecto):
            raise RuntimeError("External AI not configured - stub.")
        except Exception as e:
            last_err = str(e)
            time.sleep(wait_seconds)
            continue
    return None, last_err

# ---------------------------
# Export helpers: PDF and Excel
# ---------------------------
def make_pdf_report(asignatura, proc_df, meta, qstats, diagnostics, group_summary):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle(f"Reporte_{asignatura}")
    width, height = letter
    y = height - 50
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, f"Reporte de Análisis - {asignatura}")
    y -= 20
    pdf.setFont("Helvetica", 10)
    for k, v in group_summary.items():
        pdf.drawString(50, y, f"{k}: {v}")
        y -= 12
    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Diagnóstico temático (problemas detectados)")
    y -= 16
    pdf.setFont("Helvetica", 10)
    if len(diagnostics)==0:
        pdf.drawString(50, y, "No se detectaron tópicos problemáticos según las heurísticas.")
        y -= 12
    else:
        for d in diagnostics:
            if y < 120:
                pdf.showPage()
                y = height - 50
                pdf.setFont("Helvetica", 10)
            pdf.drawString(50, y, f"Tópico: {d['Topic']} | Pregunta: {d['Question'][:80]}")
            y -= 12
            pdf.drawString(60, y, f"Diagnóstico: {d['Diagnosis'][:150]}")
            y -= 12
            pdf.drawString(60, y, f"Recomendación docente: {d['TeacherRec']}")
            y -= 16
    # agregar tabla resumen por pregunta (top 10 más problemáticas)
    pdf.showPage()
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, height - 50, "Resumen por pregunta (estadísticas)")
    y = height - 70
    pdf.setFont("Helvetica", 10)
    # write header
    pdf.drawString(50, y, "Pregunta")
    pdf.drawString(360, y, "MostFreq%")
    pdf.drawString(450, y, "Entropy")
    y -= 14
    for _, r in qstats.head(30).iterrows():
        if y < 80:
            pdf.showPage()
            y = height - 50
        pdf.drawString(50, y, str(r["Question"])[:70])
        pdf.drawString(360, y, str(r["MostFreqPct"]))
        pdf.drawString(450, y, str(r["Entropy"]))
        y -= 12
    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer

def make_excel_report(proc_df, qstats, diagnostics):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        proc_df.to_excel(writer, sheet_name="RawProcessed", index=False)
        qstats.to_excel(writer, sheet_name="QuestionStats", index=False)
        pd.DataFrame(diagnostics).to_excel(writer, sheet_name="Diagnostics", index=False)
    output.seek(0)
    return output

# ---------------------------
# Interfaz Streamlit
# ---------------------------
st.markdown("Suba aquí el archivo CSV exportado por Plickers (tal cual). La aplicación detectará encabezados y procesará automáticamente.")
uploaded = st.file_uploader("Archivo CSV de Plickers", type=["csv"])

asignatura = st.text_input("Nombre de la asignatura (para el reporte)")

if uploaded is not None and asignatura:
    st.info("Procesando archivo, por favor espere...")
    processed_res, err = load_processed_df(uploaded)
    if processed_res is None:
        st.error(err)
    else:
        proc_df, meta = processed_res
        st.success("Archivo leido y procesado correctamente.")
        st.subheader("Vista previa (datos procesados)")
        st.dataframe(proc_df.head(10))

        # group summary
        # try compute average from Score if exists, else attempt to estimate per-student percent from Correct/Answered
        canonical = meta["canonical"]
        if canonical.get("score") and canonical.get("correct") and canonical.get("answered"):
            # compute percent correct per student if possible
            score_col = canonical["score"]
            correct_col = canonical["correct"]
            ans_col = canonical["answered"]
            proc_df["Percent"] = proc_df.apply(lambda r: (pd.to_numeric(r[correct_col], errors='coerce') / pd.to_numeric(r[ans_col], errors='coerce') * 100) if pd.to_numeric(r[ans_col], errors='coerce') else pd.to_numeric(r[score_col], errors='coerce'), axis=1)
        elif canonical.get("score"):
            score_col = canonical["score"]
            proc_df["Percent"] = pd.to_numeric(proc_df[score_col], errors='coerce')
        else:
            # fallback: set Percent NaN
            proc_df["Percent"] = np.nan

        group_summary = {
            "NumRecords": int(len(proc_df)),
            "AvgPercent": float(proc_df["Percent"].mean(skipna=True)) if not proc_df["Percent"].isna().all() else "N/A"
        }

        st.subheader("Resumen grupal")
        st.write(group_summary)

        # detect question columns
        question_cols = meta["question_cols"]
        if len(question_cols)==0:
            st.warning("No se detectaron columnas de preguntas. Asegúrese de subir el CSV exportado por Plickers.")
        else:
            st.subheader("Análisis por pregunta")
            qstats, diagnostics = generate_offline_diagnostics(proc_df, meta)
            st.dataframe(qstats.head(20))

            st.subheader("Diagnósticos temáticos sugeridos (offline)")
            if len(diagnostics)==0:
                st.write("No se detectaron tópicos críticos con la heurística actual.")
            else:
                for d in diagnostics:
                    st.markdown(f"**Tema:** {d['Topic']} — **Pregunta:** {d['Question'][:100]}")
                    st.write(f"- Diagnóstico: {d['Diagnosis']}")
                    st.write(f"- Recomendación docente: {d['TeacherRec']}")
                    st.write(f"- Recomendación alumno: {d['StudentRec']}")
                    st.write("")

            # Allow user to try external enhancement with retries (optional)
            st.subheader("Mejorar diagnósticos con IA externa (opcional)")
            st.write("Si habilita este paso y tiene un servicio externo configurado en try_external_ai_enhancement, la app intentará mejorar los diagnósticos (3 reintentos). Si no está configurado, se usará la versión offline.")
            if st.button("Intentar mejorar diagnósticos con IA externa (3 reintentos)"):
                prompt = "Genera diagnóstico pedagógico y micro-sesiones para las siguientes preguntas:\n"
                for _, r in qstats.head(10).iterrows():
                    prompt += f"- {r['Question'][:200]}\n"
                res_text, err_msg = try_external_ai_enhancement(prompt, max_retries=3, wait_seconds=3)
                if res_text:
                    st.success("Diagnóstico mejorado con IA externa (resultado):")
                    st.code(res_text)
                else:
                    st.warning(f"No se pudo contactar al servicio externo: {err_msg}. Se mantienen diagnósticos locales.")

            # Exports
            st.subheader("Descargar reportes")
            pdf_report = make_pdf_report(asignatura, proc_df, meta, qstats, diagnostics, group_summary)
            st.download_button("Descargar reporte (PDF)", data=pdf_report, file_name=f"Reporte_{asignatura}.pdf", mime="application/pdf")
            excel_report = make_excel_report(proc_df, qstats, diagnostics)
            st.download_button("Descargar reporte (Excel)", data=excel_report, file_name=f"Reporte_{asignatura}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.success("Proceso completado.")
else:
    st.info("Suba el CSV exportado desde Plickers y escriba el nombre de la asignatura para comenzar.")


