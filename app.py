import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from PrescripTech_Model import YOLO_Pred  # Assuming your YOLO_Pred class is in a file named yolo_pred.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import base64
import io

# Main function for Streamlit app
def main():
    st.title('PrescripTech')

    # File uploader to upload the image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Check if an image has been uploaded
    if uploaded_image is not None:
        # Read the uploaded image using PIL
        pil_image = Image.open(uploaded_image)

        # Convert PIL image to NumPy array
        image_np = np.array(pil_image)

        # Display the processed image
        st.image(pil_image, caption='Uploaded Image', use_column_width=True)

        # Create an instance of the YOLO_Pred class
        yolo = YOLO_Pred('best.onnx', 'data.yaml')

        # Button to trigger prediction and display
        if st.button('Convert'):
            df , res = yolo.final_predictions(image_np)

            # Display DataFrames
            st.write('Generic Medicines:')
            st.write(df)

            st.write('Information not available for :')
            st.write(pd.DataFrame(res,columns=['Brand Medicine']))

            if not df.empty:
                # Download button for prescription in PDF format
                st.write('Download Prescription')
                download_pdf(df, res, "prescription.pdf")

def download_pdf(df, res, filename):
    # Create a PDF report
    buffer = io.BytesIO()

    # Set up PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Convert DataFrames to list of lists for tables
    data_df = [df.columns.tolist()] + df.values.tolist()
    data_res = [['Brand Medicine']] + [[item] for item in res]

    #
    elements.append(Table([['Prescription']],style=[
        ('TEXTCOLOR', (0, 0), (0, 0), 'black'),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, 0), 18),
        ('BOTTOMPADDING', (0, 0), (0, 0), 12),
        ('ALIGN', (0, 0), (0, 0), 'CENTER') 
    ]))

    # Title before table_df
    elements.append(Table([['Generic Medicines']],style=[
        ('TEXTCOLOR', (0, 0), (0, 0), 'black'),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, 0), 14),
        ('BOTTOMPADDING', (0, 0), (0, 0), 12),
        ('ALIGN', (0, 0), (0, 0), 'LEFT')  # Align left
    ])) 

    # Create table for df
    table_df = Table(data_df)

    # Style the table_df
    style = TableStyle([('BACKGROUND', (0, 0), (-1, 0), 'grey'),
                        ('TEXTCOLOR', (0, 0), (-1, 0), 'white'),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), 'azure'),
                        ('GRID', (0, 0), (-1, -1), 1, 'BLACK')])

    table_df.setStyle(style)

    # Add table_df to elements
    elements.append(table_df)

    # Add space
    elements.append(Table([[' ']]))

    # Title before table_res
    elements.append(Table([['Information not available for']],style=[
        ('TEXTCOLOR', (0, 0), (0, 0), 'black'),
        ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (0, 0), 14),
        ('BOTTOMPADDING', (0, 0), (0, 0), 12),
        ('ALIGN', (0, 0), (0, 0), 'LEFT')  # Align left
    ])) 

    # Create table for res
    table_res = Table(data_res)

    # Style the table_res
    table_res.setStyle(style)

    # Add table_res to elements
    elements.append(table_res)

    # Build PDF
    doc.build(elements)

    # Save PDF to buffer
    pdf_data = buffer.getvalue()
    buffer.close()

    # Provide download link
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF File</a>'
    st.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()