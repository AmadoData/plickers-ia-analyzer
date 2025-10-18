import streamlit as st
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ----------------------------------------------------------
# Función: identificar automáticamente dónde comienzan los encabezados
# ----------------------------------------------------------
def procesar_archivo(archivo_csv):
    try:
        # Leer el CSV sin encabezados (por si tiene texto previo)
        df_raw = pd.read_csv(archivo_csv, header=None, encoding="utf-8", on_bad_lines="skip")
        
        # Buscar la fila que contiene "Card Number"
        encabezado_idx = None
        for i, row in df_raw.iterrows():
            if any("Card Number" in str(cell) for cell in row):
                encabezado_idx = i
                break

        if encabezado_idx is None:
            raise ValueError("No se encontró la fila de encabezados ('Card Number') en el CSV.")
        
        # Leer nuevamente el archivo desde esa fila
        df = pd.read_csv(archivo_csv, skiprows=encabezado_idx, encoding="utf-8")
        df = df.dropna(how="all")
        
        # Limpiar espacios y caracteres
        df.columns = [str(c).strip() for c in df.columns]
        
        # Validar columnas clave
        columnas_base = ["Card Number", "First name", "Last Name", "Score", "Correct", "Answered"]
        columnas_presentes = [c for c in columnas_base if c in df.columns]
        
        if len(columnas_presentes) < 4:
            raise ValueError("El archivo no tiene las columnas esperadas de Plickers.")
        
        # Filtrar datos válidos
        df = df[df["First name"].notna()]
        df = df[df["First name"] != ""]

        return df

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None


# ----------------------------------------------------------
# Función: análisis del desempeño y generación de sugerencias
# ----------------------------------------------------------
def analizar_resultados(df):
    # Normalizar y limpiar datos
    df["Score"] = df["Score"].replace("-", "0").replace("%", "", regex=True)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0)
    df["Correct"] = pd.to_numeric(df["Correct"], errors="coerce").fillna(0)
    df["Answered"] = pd.to_numeric(df["Answered"], errors="coerce").replace(0, 1)

    promedio = df["Score"].mean()

    sugerencias = []
    for _, row in df.iterrows():
        nombre = f"{row['First name']} {row['Last Name']}".strip()
        aciertos = (row["Correct"] / row["Answered"]) * 100
        if aciertos < 50:
            rec = "Reforzar conceptos fundamentales. Se recomienda repasar los temas básicos y realizar ejercicios guiados."
        elif aciertos < 80:
            rec = "Buen desempeño general. Puede mejorar con actividades prácticas y debates reflexivos."
        else:
            rec = "Excelente comprensión. Se recomienda avanzar a proyectos aplicados o tutorías de apoyo a compañeros."

        sugerencias.append({
            "Alumno": nombre,
            "Aciertos (%)": round(aciertos, 2),
            "Recomendación": rec
        })

    # Detectar preguntas más falladas
    preguntas = [c for c in df.columns if c not in ["Card Number", "First name", "Last Name", "Score", "Correct", "Answered"]]
    resumen_preguntas = []
    for pregunta in preguntas:
        respuestas = df[pregunta].value_counts()
        if len(respuestas) > 0:
            pregunta_info = {
                "Pregunta": pregunta[:80],
                "Opción más frecuente": respuestas.index[0],
                "Veces respondida": respuestas.iloc[0]
            }
            resumen_preguntas.append(pregunta_info)

    return promedio, sugerencias, resumen_preguntas


# ----------------------------------------------------------
# Función: generación del PDF de reporte
# ----------------------------------------------------------
def generar_pdf(promedio, sugerencias, resumen_preguntas, asignatura):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Reporte Plickers - Análisis IA")

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(180, 760, "Reporte de Análisis Plickers con IA")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, 735, f"Asignatura: {asignatura}")
    pdf.drawString(50, 720, f"Promedio general del grupo: {round(promedio, 2)}%")

    y = 700
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Sugerencias Individuales:")
    y -= 20
    pdf.setFont("Helvetica", 10)

    for s in sugerencias:
        if y < 100:
            pdf.showPage()
            y = 750
        pdf.drawString(50, y, f"{s['Alumno']} - {s['Aciertos (%)']}% - {s['Recomendación']}")
        y -= 15

    # Nueva página: resumen docente
    pdf.showPage()
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, 750, "Resumen de Preguntas y Sugerencias para Rediseño Docente")
    pdf.setFont("Helvetica", 10)
    y = 720
    for p in resumen_preguntas:
        if y < 100:
            pdf.showPage()
            y = 750
        pdf.drawString(50, y, f"Pregunta: {p['Pregunta']}")
        pdf.drawString(60, y - 12, f"Opción más frecuente: {p['Opción más frecuente']} | Veces respondida: {p['Veces respondida']}")
        y -= 30

    pdf.save()
    buffer.seek(0)
    return buffer


# ----------------------------------------------------------
# Interfaz principal Streamlit
# ----------------------------------------------------------
st.set_page_config(page_title="Plickers IA Analyzer", layout="centered")

st.title("Plickers IA Analyzer")
st.write("""
Esta aplicación analiza automáticamente los resultados descargados desde **Plickers**, 
sin necesidad de conexión de los alumnos a internet ni de editar el archivo.
Ofrece sugerencias de retroalimentación individual y propone ajustes al docente para rediseñar las sesiones.
""")

archivo = st.file_uploader("Sube el archivo CSV de Plickers", type=["csv"])
nombre_asignatura = st.text_input("Nombre de la asignatura:")

if archivo and nombre_asignatura:
    df = procesar_archivo(archivo)
    if df is not None:
        st.success("Archivo procesado correctamente.")
        st.dataframe(df.head())

        promedio, sugerencias, resumen_preguntas = analizar_resultados(df)

        st.subheader("Análisis General del Grupo")
        st.write(f"Promedio general: {round(promedio, 2)}%")

        st.subheader("Sugerencias Individuales")
        st.dataframe(pd.DataFrame(sugerencias))

        st.subheader("Preguntas más representativas")
        st.dataframe(pd.DataFrame(resumen_preguntas))

        pdf = generar_pdf(promedio, sugerencias, resumen_preguntas, nombre_asignatura)
        st.download_button(
            label="Descargar Reporte en PDF",
            data=pdf,
            file_name=f"Reporte_{nombre_asignatura}.pdf",
            mime="application/pdf"
        )


