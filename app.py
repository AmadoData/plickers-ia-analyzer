import streamlit as st
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ----------------------------------------------------------
# Función para procesar automáticamente el CSV de Plickers
# ----------------------------------------------------------
def procesar_archivo(archivo_csv):
    try:
        # Leer el archivo sin asumir encabezado
        df_raw = pd.read_csv(archivo_csv, encoding="utf-8", header=None, on_bad_lines='skip')

        # Buscar la fila que contiene "Card Number"
        start_index = df_raw[df_raw.apply(lambda row: row.astype(str).str.contains("Card Number").any(), axis=1)].index
        if len(start_index) == 0:
            raise ValueError("No se encontró la fila de encabezados ('Card Number') en el CSV.")
        start_index = start_index[0]

        # Cargar el dataset desde esa fila
        df = pd.read_csv(archivo_csv, encoding="utf-8", skiprows=start_index)

        # Eliminar filas vacías
        df = df.dropna(how="all")
        df.columns = [col.strip() for col in df.columns]

        # Detectar columnas base
        columnas_clave = [c for c in df.columns if c.lower() in [
            "card number", "first name", "last name", "score", "correct", "answered"
        ]]
        columnas_preguntas = [c for c in df.columns if c not in columnas_clave]
        df = df[columnas_clave + columnas_preguntas]

        return df

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None


# ----------------------------------------------------------
# Análisis de resultados y generación de sugerencias
# ----------------------------------------------------------
def analizar_resultados(df):
    # Calcular promedio general
    promedio = df['Score'].mean() if 'Score' in df.columns else None

    # Generar sugerencias por alumno
    sugerencias = []
    for i, row in df.iterrows():
        nombre = f"{row.get('First name', '')} {row.get('Last name', '')}".strip()
        score = row.get('Score', 0)
        correctas = row.get('Correct', 0)
        total = row.get('Answered', 1)
        porcentaje = (correctas / total) * 100 if total > 0 else 0

        if porcentaje < 50:
            sugerencia = "Reforzar conceptos básicos del tema. Se recomienda repasar los fundamentos con ejemplos prácticos y actividades visuales."
        elif porcentaje < 80:
            sugerencia = "Buen desempeño, aunque es recomendable realizar ejercicios de aplicación y análisis de casos."
        else:
            sugerencia = "Excelente resultado. Puede avanzar hacia actividades de síntesis o proyectos de integración."

        sugerencias.append({
            "Alumno": nombre,
            "Aciertos (%)": round(porcentaje, 2),
            "Recomendación": sugerencia
        })

    return promedio, sugerencias


# ----------------------------------------------------------
# Generación del reporte PDF
# ----------------------------------------------------------
def generar_pdf(promedio, sugerencias, nombre_asignatura):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Reporte de Análisis Plickers IA")

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(200, 750, "Reporte de Análisis Plickers IA")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, 720, f"Asignatura: {nombre_asignatura}")
    pdf.drawString(50, 705, f"Promedio general del grupo: {round(promedio,2) if promedio else 'N/A'}%")

    y = 680
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Sugerencias por alumno:")
    y -= 20
    pdf.setFont("Helvetica", 10)

    for s in sugerencias:
        if y < 100:
            pdf.showPage()
            y = 750
        pdf.drawString(50, y, f"{s['Alumno']} - {s['Aciertos (%)']}% - {s['Recomendación']}")
        y -= 15

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer


# ----------------------------------------------------------
# Interfaz principal Streamlit
# ----------------------------------------------------------
st.set_page_config(page_title="Plickers IA Analyzer", layout="centered")

st.title("Plickers IA Analyzer")
st.write("""
Esta herramienta permite analizar automáticamente los resultados exportados desde **Plickers**, 
generando retroalimentación personalizada para cada alumno y sugerencias para rediseñar las sesiones 
de aprendizaje según el desempeño observado.
""")

archivo = st.file_uploader("📁 Sube el archivo CSV exportado desde Plickers", type=["csv"])
nombre_asignatura = st.text_input("Nombre de la asignatura:")

if archivo and nombre_asignatura:
    df = procesar_archivo(archivo)
    if df is not None:
        st.success("Archivo cargado correctamente.")
        st.dataframe(df.head())

        promedio, sugerencias = analizar_resultados(df)

        st.subheader("Análisis general del grupo")
        st.write(f"**Promedio general:** {round(promedio,2)}%")

        st.subheader("Sugerencias por alumno")
        st.dataframe(pd.DataFrame(sugerencias))

        # Generar PDF
        pdf_buffer = generar_pdf(promedio, sugerencias, nombre_asignatura)
        st.download_button(
            label="Descargar reporte PDF",
            data=pdf_buffer,
            file_name=f"Reporte_{nombre_asignatura}.pdf",
            mime="application/pdf"
        )

