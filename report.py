from fpdf import FPDF
from datetime import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Road Damage Detection Report', 0, 1, 'C')
        self.cell(0, 10, 'Using YOLO Model', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_summary(self, summary_dict):
        self.chapter_title('Summary of Findings')
        for key, value in summary_dict.items():
            self.cell(0, 10, f'{key}: {value}', 0, 1)
        self.ln()

    def add_detailed_analysis(self, analysis_dict):
        self.chapter_title('Detailed Analysis')
        for damage_type, details in analysis_dict.items():
            self.chapter_title(damage_type)
            for key, value in details.items():
                self.cell(0, 10, f'{key}: {value}', 0, 1)
            self.ln()

# Data to be included in the report
summary_data = {
    "Total damages detected": "X",
    
    "Types Of Damages":'',
    "Alligator Cracks": "X",
    "Damaged Crosswalk": "X",
    "Damaged Paint": "X",
    "Longitudinal Cracks": "X",
    "Manhole Cover": "X",
    "Potholes": "X",
    "Transverse Cracks": "X"
}


# Creating PDF
pdf = PDF()

pdf.add_page()

# Adding Introduction
pdf.chapter_title('Introduction')
intro_text = (
    "This report provides an analysis of road damages detected using the YOLO model. "
    "The primary objective is to identify and categorize different types of road damages to assist in maintenance planning."
)
pdf.chapter_body(intro_text)

# Adding Summary
pdf.add_summary(summary_data)



# Adding placeholder for Visualizations
pdf.chapter_title('Visualizations')
pdf.cell(0, 10, 'Chart 1: Distribution of Road Damages', 0, 1)
pdf.cell(0, 10, '[Insert Bar Chart]', 0, 1)
pdf.ln(10)
pdf.cell(0, 10, 'Chart 2: Trend Analysis of Road Damages Over Time', 0, 1)
pdf.cell(0, 10, '[Insert Line Graph]', 0, 1)
pdf.ln(20)

# Adding Conclusion
pdf.chapter_title('Conclusion')
conclusion_text = (
    "The analysis indicates that the most common type of damage is [Most Common Damage]. "
    "Immediate attention is recommended for areas with severe damages."
)
pdf.chapter_body(conclusion_text)

# Adding Recommendations
pdf.chapter_title('Recommendations')
recommendations_text = (
    "Prioritize repair for [High Priority Areas].\n"
    "Conduct regular monitoring to track changes and new damages."
)
pdf.chapter_body(recommendations_text)

# Save the PDF to a file
pdf_file_name = 'Road_Damage_Detection_Report.pdf'
pdf.output(pdf_file_name)













