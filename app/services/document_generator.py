import io
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# ─── Custom Canvas for Page Numbers (Page X of Y) ────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b")) # Slate 500
        
        # Bottom divider
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(0.75 * inch, 0.75 * inch, 7.75 * inch, 0.75 * inch)
        
        # Footer text
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(7.75 * inch, 0.55 * inch, page_text)
        self.drawString(0.75 * inch, 0.55 * inch, "LuminaThesis AI — Thesis Chapter Report")
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.line(0.75 * inch, 10.25 * inch, 7.75 * inch, 10.25 * inch)
            self.drawString(0.75 * inch, 10.35 * inch, "Thesis Synthesis Draft")
            
        self.restoreState()


# ─── Markdown Parser helper for ReportLab ────────────────────────────────────
def markdown_to_html(text: str) -> str:
    """Converts basic markdown bold/italics/code tags to ReportLab paragraph HTML tags."""
    if not text:
        return ""
    
    # Escape HTML tags (except formatting tags we insert)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Replace bold markdown (**text**)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    
    # Replace italic markdown (*text*)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    
    # Replace inline code (`code`)
    text = re.sub(r"`(.*?)`", r"<font name='Courier'>\1</font>", text)
    
    # Restore the formatting tags we explicitly generated
    text = text.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    text = text.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
    text = text.replace("&lt;font name='Courier'&gt;", "<font name='Courier'>").replace("&lt;/font&gt;", "</font>")
    
    return text


# ─── DOCX Generation ────────────────────────────────────────────────────────
def generate_docx(data: dict) -> io.BytesIO:
    """Generates an academic, styled DOCX file containing the synthesized thesis."""
    doc = Document()
    
    # Page Setup - 1 inch margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
    # Typography Setup
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Arial'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x33, 0x41, 0x55) # Slate 700
    
    # Cover Header
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run("LUMINA THESIS CHAPTER SYNTHESIS\n")
    title_run.font.name = 'Arial'
    title_run.font.size = Pt(20)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1e, 0x1b, 0x4b) # Indigo 950
    
    topic_p = doc.add_paragraph()
    topic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    topic_run = topic_p.add_run(f"Topic: {data.get('topic', 'Untitled Research')}\n")
    topic_run.font.name = 'Arial'
    topic_run.font.size = Pt(13)
    topic_run.font.italic = True
    topic_run.font.color.rgb = RGBColor(0x4f, 0x46, 0xe5) # Indigo 600
    
    doc.add_paragraph() # Spacer
    
    # Separator Line (border)
    p_sep = doc.add_paragraph()
    p_sep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sep_run = p_sep.add_run("―" * 50)
    sep_run.font.color.rgb = RGBColor(0xe2, 0xe8, 0xf0)
    
    doc.add_paragraph()
    
    def add_section(heading_text, content):
        if not content:
            return
        
        # Section Heading
        h = doc.add_heading(level=1)
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(6)
        h_run = h.add_run(heading_text)
        h_run.font.name = 'Arial'
        h_run.font.size = Pt(14)
        h_run.font.bold = True
        h_run.font.color.rgb = RGBColor(0x31, 0x2e, 0x81) # Indigo 900
        
        # Split by double newline for paragraphs
        paragraphs = content.split('\n\n')
        for p_text in paragraphs:
            p_text = p_text.strip()
            if not p_text:
                continue
            
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(10)
            p.paragraph_format.line_spacing = 1.15
            
            # Formatting markdown bold
            parts = p_text.split('**')
            is_bold = False
            for part in parts:
                run = p.add_run(part)
                run.font.bold = is_bold
                is_bold = not is_bold
                
    add_section("1. Background", data.get("background", ""))
    add_section("2. Problem Statement", data.get("problem_statement", ""))
    add_section("3. Research Questions", data.get("research_questions", ""))
    add_section("4. SMART Objectives", data.get("objectives", ""))
    add_section("5. Research Hypotheses", data.get("hypotheses", ""))
    add_section("6. Chapter Scope", data.get("scope", ""))
    add_section("7. Final Output & Detailed Chapter Prose", data.get("final_output", ""))
    
    file_stream = io.BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream


# ─── PDF Generation ─────────────────────────────────────────────────────────
def generate_pdf(data: dict) -> io.BytesIO:
    """Generates an academic, styled PDF file containing the synthesized thesis."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.85*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e1b4b"), # Indigo 950
        alignment=1, # Center
        spaceAfter=15
    )
    
    topic_style = ParagraphStyle(
        'DocTopic',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4f46e5"), # Indigo 600
        alignment=1, # Center
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#312e81"), # Indigo 900
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'AcademicBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#334155"), # Slate 700
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'AcademicBullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=6
    )
    
    story = []
    
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph("LUMINA THESIS CHAPTER SYNTHESIS", title_style))
    story.append(Paragraph(f"Topic: {data.get('topic', 'Untitled Research')}", topic_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Metadata Banner Box
    banner_data = [["Compiled by LuminaThesis PhD AI Assistant System"]]
    banner_table = Table(banner_data, colWidths=[6.5*inch])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#64748b")),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Oblique'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ('LINEABOVE', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 0.3*inch))
    
    def add_section(title_text, content_text):
        if not content_text:
            return
        story.append(Paragraph(title_text, h1_style))
        paragraphs = content_text.split("\n\n")
        for p_text in paragraphs:
            p_text = p_text.strip()
            if not p_text:
                continue
            
            # Check if this paragraph contains a bulleted list or line-by-line list items
            if p_text.startswith("- ") or p_text.startswith("* ") or (p_text[0:1].isdigit() and p_text[1:3] == ". "):
                lines = p_text.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("- ") or line.startswith("* "):
                        clean_line = line[2:]
                        html_line = f"&bull;&nbsp;&nbsp;{markdown_to_html(clean_line)}"
                        story.append(Paragraph(html_line, bullet_style))
                    elif re.match(r"^\d+\.\s+", line):
                        # Numbered list
                        clean_line = re.sub(r"^\d+\.\s+", "", line)
                        prefix = re.match(r"^(\d+\.)\s+", line).group(1)
                        html_line = f"{prefix}&nbsp;&nbsp;{markdown_to_html(clean_line)}"
                        story.append(Paragraph(html_line, bullet_style))
                    else:
                        story.append(Paragraph(markdown_to_html(line), body_style))
            else:
                html_text = markdown_to_html(p_text)
                story.append(Paragraph(html_text, body_style))
                
    add_section("1. Background", data.get("background", ""))
    add_section("2. Problem Statement", data.get("problem_statement", ""))
    add_section("3. Research Questions", data.get("research_questions", ""))
    add_section("4. SMART Objectives", data.get("objectives", ""))
    add_section("5. Research Hypotheses", data.get("hypotheses", ""))
    add_section("6. Chapter Scope", data.get("scope", ""))
    add_section("7. Final Output & Detailed Chapter Prose", data.get("final_output", ""))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer
