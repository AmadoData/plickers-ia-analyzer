import streamlit as st
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(
    page_title="Analizador de Evaluaciones Plickers",
    layout="wide"
)

st.title("Analizador de Evaluaciones Plickers")
st.write("Sube el archivo CSV exportado directamente desde Plickers para generar un reporte de análisis y recomendaciones pedagógicas automáticas.")

# ---------------------------------------------------------
# CARGA DE ARCHIVO CSV
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Cargar archivo CSV de Plickers", type=["csv"])

if uploaded_file is not None:
    try:
        # Lectura del archivo con tolerancia a codificaciones
        data = pd.read_csv(uploaded_file, encoding="utf-8", skip_blank_lines=True)
        st.success("Archivo cargado correctamente.")
        st.subheader("Vista previa de los datos")
        st.dataframe(data.head())

        # ---------------------------------------------------------
        # DETECCIÓN AUTOMÁTICA DE COLUMNAS
        # ---------------------------------------------------------
        data.columns = data.columns.str.lower()

        posibles_cols_nombre = [c for c in data.columns if 'name' in c or 'student' in c or 'first' in c]
        posibles_cols_pregunta = [c for c in data.columns if 'question' in c]
        posibles_cols_correcto = [c for c in data.columns if 'correct' in c]
        posibles_cols_score = [c for c in data.columns if 'score' in c or 'points' in c]

        col_nombre = posibles_cols_nombre[0] if posibles_cols_nombre else None
        col_pregunta = posibles_cols_pregunta[0] if posibles_cols_pregunta else None
        col_correcta = posibles_cols_correcto[0] if posibles_cols_correcto else None
        col_score = posibles_cols_score[0] if posibles_cols_score else None

        # ---------------------------------------------------------
        # ANÁLISIS DE RESULTADOS
        # ---------------------------------------------------------
        st.header("Resultados Generales del Grupo")

        resumen_texto = ""

        if col_nombre and col_score:
            resumen_alumnos = data.groupby(col_nombre)[col_score].mean().reset_index()
            resumen_alumnos.columns = ["Alumno", "Promedio de aciertos"]
            resumen_alumnos = resumen_alumnos.sort_values(by="Promedio de aciertos", ascending=False)
            st.dataframe(resumen_alumnos)

            promedio_grupal = resumen_alumnos["Promedio de aciertos"].mean()
            st.write(f"Promedio grupal de aciertos: {promedio_grupal:.2f}")
            resumen_texto += f"Promedio grupal: {promedio_grupal:.2f}\n"

        if col_pregunta and col_score:
            resumen_preguntas = data.groupby(col_pregunta)[col_score].mean().reset_index()
            resumen_preguntas.columns = ["Pregunta", "Porcentaje de aciertos"]
            st.write("Porcentaje de aciertos por pregunta:")
            st.dataframe(resumen_preguntas)

        # ---------------------------------------------------------
        # SUGERENCIAS POR ALUMNO
        # ---------------------------------------------------------
        st.header("Análisis Individual y Recomendaciones Personalizadas")

        def sugerencia_porcentaje(p):
            if p >= 0.9:
                return "Excelente desempeño. Se recomienda promover el análisis crítico mediante actividades de profundización."
            elif p >= 0.75:
                return "Buen desempeño. Puede beneficiarse de actividades prácticas que fortalezcan la transferencia de conocimiento."
            elif p >= 0.6:
                return "Desempeño regular. Se sugiere reforzar los conceptos fundamentales y ofrecer tutorías dirigidas."
            else:
                return "Desempeño bajo. Es recomendable implementar sesiones de recuperación, aprendizaje cooperativo y acompañamiento personalizado."

        analisis_alumnos = []
        for _, row in resumen_alumnos.iterrows():
            alumno = row["Alumno"]
            promedio = row["Promedio de aciertos"]
            sugerencia = sugerencia_porcentaje(promedio)
            analisis_alumnos.append({"Alumno": alumno, "Promedio": promedio, "Sugerencia": sugerencia})

        analisis_df = pd.DataFrame(analisis_alumnos)
        st.dataframe(analisis_df)

        # ---------------------------------------------------------
        # REDISEÑO DE SESIONES
        # ---------------------------------------------------------
        st.header("Rediseño Sugerido de Sesiones")

        resumen_texto += "\nRediseño sugerido:\n"

        if col_pregunta and col_score:
            preguntas_bajas = resumen_preguntas[resumen_preguntas["Porcentaje de aciertos"] < 0.6]

            if len(preguntas_bajas) > 0:
                st.write("Temas con menor desempeño detectados:")
                st.dataframe(preguntas_bajas)

                st.write("Sugerencias para el rediseño:")
                for _, row in preguntas_bajas.iterrows():
                    st.markdown(f"- Revisar el tema asociado a la pregunta: **{row['Pregunta']}**.")
                    st.markdown("  Incorporar estrategias activas como debates, estudios de caso y resolución de problemas.")
                    resumen_texto += f"- Tema a reforzar: {row['Pregunta']}\n"
            else:
                st.write("No se detectaron temas con bajo desempeño.")
                resumen_texto += "No se detectaron temas críticos.\n"

        # ---------------------------------------------------------
        # GENERACIÓN DE REPORTE EN PDF
        # ---------------------------------------------------------
        st.header("Descarga del Reporte en PDF")

        def generar_pdf():
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            y = height - 50

            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y, "Reporte de Análisis Plickers")
            y -= 25
            c.setFont("Helvetica", 10)
            c.drawString(50, y, f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            y -= 40

            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Resumen General")
            y -= 20

            for line in resumen_texto.split("\n"):
                c.setFont("Helvetica", 10)
                c.drawString(50, y, line)
                y -= 15
                if y < 80:
                    c.showPage()
                    y = height - 50

            c.showPage()
            c.save()
            buffer.seek(0)
            return buffer

        pdf_buffer = generar_pdf()
        st.download_button(
            label="Descargar reporte en PDF",
            data=pdf_buffer,
            file_name="reporte_plickers.pdf",
            mime="application/pdf"
        )

        st.success("Análisis completado correctamente.")

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

else:
    st.info("Por favor, carga el archivo CSV exportado desde Plickers para generar el reporte completo.")

