import streamlit as st
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Análisis Educativo con IA - Plickers", layout="centered")

st.title("Análisis Educativo Automatizado con IA")
st.write("Cargue el archivo CSV exportado directamente desde Plickers para generar un reporte con análisis, retroalimentación y rediseño de sesiones.")

uploaded_file = st.file_uploader("Cargar archivo CSV de Plickers", type=["csv"])

def generar_pdf(df, resumen_general, recomendaciones_alumnos, recomendaciones_docente):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Reporte de Análisis Educativo - Plickers")

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, 750, "Reporte de Análisis Educativo")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, 730, "Generado a partir del archivo CSV de Plickers")
    pdf.line(50, 725, 560, 725)

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, 705, "Resumen General")
    pdf.setFont("Helvetica", 10)

    y = 690
    for key, value in resumen_general.items():
        pdf.drawString(60, y, f"{key}: {value}")
        y -= 15

    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Recomendaciones para cada alumno")
    y -= 20
    pdf.setFont("Helvetica", 9)
    for alumno, texto in recomendaciones_alumnos.items():
        pdf.drawString(60, y, f"{alumno}: {texto[:100]}...")
        y -= 12
        if y < 100:
            pdf.showPage()
            y = 740

    y -= 20
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Sugerencias para el docente y rediseño de sesiones")
    y -= 20
    pdf.setFont("Helvetica", 9)
    for texto in recomendaciones_docente:
        pdf.drawString(60, y, f"- {texto}")
        y -= 12
        if y < 100:
            pdf.showPage()
            y = 740

    pdf.save()
    buffer.seek(0)
    return buffer


def generar_recomendacion_individual(row):
    score = row["Score"]
    if pd.isna(score):
        return "Sin datos suficientes para analizar."
    if score >= 80:
        return "Excelente desempeño. Puede participar en actividades de tutoría o refuerzo colaborativo."
    elif score >= 60:
        return "Buen desempeño general, pero se recomienda reforzar temas teóricos con recursos visuales."
    elif score >= 40:
        return "Se observan áreas de mejora. Se sugiere repasar los temas conceptuales con ejemplos prácticos."
    else:
        return "Bajo desempeño. Requiere acompañamiento individual y material adaptado a su estilo de aprendizaje."


def generar_recomendaciones_docente(df):
    promedio = df["Score"].mean()
    if promedio >= 80:
        return [
            "El grupo muestra dominio general de los contenidos. Enfocar actividades en pensamiento crítico.",
            "Implementar debates o resolución de problemas para consolidar el aprendizaje."
        ]
    elif promedio >= 60:
        return [
            "El grupo tiene un desempeño aceptable, aunque con vacíos conceptuales.",
            "Diseñar sesiones de refuerzo con apoyo visual y prácticas guiadas."
        ]
    else:
        return [
            "El grupo presenta dificultades generalizadas. Se recomienda un rediseño de las estrategias.",
            "Utilizar ejemplos del contexto real del alumnado y actividades participativas."
        ]


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Archivo cargado correctamente.")
        st.write("Vista previa de los primeros registros:")
        st.dataframe(df.head())

        columnas_estandar = ["First name", "Last Name", "Score", "Correct", "Answered"]
        disponibles = [c for c in columnas_estandar if c in df.columns]
        if not disponibles:
            st.error("El archivo no tiene las columnas esperadas de Plickers.")
        else:
            df["Nombre Completo"] = df["First name"].astype(str) + " " + df["Last Name"].astype(str)
            df["Score"] = df["Score"].replace("-", 0)
            df["Score"] = df["Score"].astype(str).str.replace("%", "").astype(float)

            promedio_general = round(df["Score"].mean(), 2)
            alumnos_aprobados = df[df["Score"] >= 60].shape[0]
            total_alumnos = df.shape[0]

            resumen_general = {
                "Promedio grupal": f"{promedio_general}%",
                "Total de alumnos": total_alumnos,
                "Alumnos aprobados": alumnos_aprobados,
                "Tasa de aprobación": f"{round((alumnos_aprobados / total_alumnos) * 100, 2)}%"
            }

            recomendaciones_alumnos = {
                row["Nombre Completo"]: generar_recomendacion_individual(row)
                for _, row in df.iterrows()
            }

            recomendaciones_docente = generar_recomendaciones_docente(df)

            st.subheader("Resumen General del Grupo")
            st.json(resumen_general)

            st.subheader("Recomendaciones Individuales")
            for alumno, texto in recomendaciones_alumnos.items():
                st.write(f"**{alumno}:** {texto}")

            st.subheader("Sugerencias para el Docente y Rediseño de Sesiones")
            for texto in recomendaciones_docente:
                st.write(f"- {texto}")

            pdf_buffer = generar_pdf(df, resumen_general, recomendaciones_alumnos, recomendaciones_docente)
            st.download_button(
                label="Descargar reporte en PDF",
                data=pdf_buffer,
                file_name="reporte_plickers.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")



