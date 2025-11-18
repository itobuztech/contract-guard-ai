# DEPENDENCIES
import os
import math
from typing import Any
from io import BytesIO
from typing import Dict
from typing import List
from typing import Optional
from datetime import datetime
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Image
from reportlab.platypus import Table
from reportlab.lib.units import inch
from reportlab.platypus import Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import Paragraph
from reportlab.platypus import PageBreak
from reportlab.graphics import renderPDF
from reportlab.platypus import TableStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.graphics.shapes import Path
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.platypus import KeepTogether
from reportlab.graphics.shapes import Circle
from reportlab.graphics.shapes import String
from reportlab.lib.pagesizes import landscape
from reportlab.graphics.shapes import Drawing
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus.flowables import PageBreak
from reportlab.platypus.flowables import KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table as PlatypusTable


class PDFReportGenerator:
    """
    Professional-grade PDF report generator matching sample style exactly
    """
    def __init__(self):
        self.styles         = getSampleStyleSheet()
        
        self._setup_custom_styles()

        self.page_width     = letter[0]
        self.page_height    = letter[1]
        self.margin_left    = 0.75 * inch
        self.margin_right   = 0.75 * inch
        self.margin_top     = 1 * inch
        self.margin_bottom  = 1 * inch
        self.content_width  = self.page_width - self.margin_left - self.margin_right
        self.content_height = self.page_height - self.margin_top - self.margin_bottom


    def _setup_custom_styles(self):
        """
        Setup custom paragraph styles with precise control
        """
        # Title style
        self.styles.add(ParagraphStyle(name       = 'ReportTitle',
                                       parent     = self.styles['Heading1'],
                                       fontSize   = 24,
                                       textColor  = colors.HexColor('#1a1a1a'),
                                       spaceAfter = 20,
                                       alignment  = TA_LEFT,
                                       fontName   = 'Helvetica-Bold',
                                      )
                       )

        # Section heading
        self.styles.add(ParagraphStyle(name        = 'SectionHeading',
                                       parent      = self.styles['Heading2'],
                                       fontSize    = 16,
                                       textColor   = colors.HexColor('#1a1a1a'),
                                       spaceAfter  = 12,
                                       spaceBefore = 20,
                                       fontName    = 'Helvetica-Bold',
                                      )
                       )

        # Sub-section heading
        self.styles.add(ParagraphStyle(name        = 'SubSectionHeading',
                                       parent      = self.styles['Normal'],
                                       fontSize    = 12,
                                       textColor   = colors.HexColor('#333333'),
                                       spaceAfter  = 8,
                                       spaceBefore = 12,
                                       fontName    = 'Helvetica-Bold',
                                      )
                       )

        # Body text
        self.styles.add(ParagraphStyle(name        = 'CustomBodyText',
                                       parent      = self.styles['Normal'],
                                       fontSize    = 10,
                                       leading     = 14,
                                       textColor   = colors.HexColor('#333333'),
                                       alignment   = TA_JUSTIFY,
                                       fontName    = 'Helvetica',
                                       leftIndent  = 0,
                                       rightIndent = 0,
                                      )
                       )

        # Small text style
        self.styles.add(ParagraphStyle(name      = 'SmallText',
                                       parent    = self.styles['Normal'],
                                       fontSize  = 8,
                                       leading   = 10,
                                       textColor = colors.HexColor('#666666'),
                                       fontName  = 'Helvetica',
                                      )
                       )

        # Bullet point
        self.styles.add(ParagraphStyle(name           = 'BulletPoint',
                                       parent         = self.styles['Normal'],
                                       fontSize       = 10,
                                       leading        = 14,
                                       textColor      = colors.HexColor('#333333'),
                                       leftIndent     = 20,
                                       bulletIndent   = 10,
                                       bulletFontName = 'Helvetica',
                                       bulletFontSize = 10,
                                       bulletColor    = colors.black,
                                       spaceAfter     = 4,
                                       fontName       = 'Helvetica',
                                      )
                       )

        # Table header
        self.styles.add(ParagraphStyle(name      = 'TableHeader',
                                       parent    = self.styles['Normal'],
                                       fontSize  = 10,
                                       textColor = colors.HexColor('#1a1a1a'),
                                       fontName  = 'Helvetica-Bold',
                                       alignment = TA_LEFT,
                                      )
                       )

        # Table cell
        self.styles.add(ParagraphStyle(name       = 'TableCell',
                                       parent     = self.styles['Normal'],
                                       fontSize   = 9,
                                       textColor  = colors.HexColor('#333333'),
                                       fontName   = 'Helvetica',
                                       alignment  = TA_LEFT,
                                       spaceAfter = 2,
                                      )
                       )

        # Footer
        self.styles.add(ParagraphStyle(name      = 'Footer',
                                       parent    = self.styles['Normal'],
                                       fontSize  = 8,
                                       textColor = colors.HexColor('#666666'),
                                       alignment = TA_CENTER,
                                       fontName  = 'Helvetica',
                                      )
                       )

        # Risk indicator style
        self.styles.add(ParagraphStyle(name       = 'RiskIndicator',
                                       parent     = self.styles['Normal'],
                                       fontSize   = 9,
                                       textColor  = colors.HexColor('#dc2626'),
                                       fontName   = 'Helvetica-Bold',
                                       backColor  = colors.HexColor('#fef2f2'),
                                       borderPadding = 5,
                                       spaceAfter = 4,
                                      )
                       )

        # Keyword style
        self.styles.add(ParagraphStyle(name       = 'Keyword',
                                       parent     = self.styles['Normal'],
                                       fontSize   = 9,
                                       textColor  = colors.HexColor('#1e40af'),
                                       fontName   = 'Helvetica',
                                       backColor  = colors.HexColor('#eff6ff'),
                                       borderPadding = 3,
                                      )
                       )


    def _draw_risk_score_circle(self, score: int) -> Drawing:
        """
        Draw the risk score circle graphic with correct fill percentage
        """
        d                  = Drawing(150, 150)
        
        # Define circle properties
        center_x, center_y = 75, 75
        outer_radius       = 60
        inner_radius       = 45
        thickness          = 15  # Thickness of the colored ring

        # Determine color based on score
        if (score >= 80):
            color = colors.HexColor('#dc2626')  # Red

        elif (score >= 60):
            color = colors.HexColor('#f97316')  # Orange

        elif (score >= 40):
            color = colors.HexColor('#ca8a04')  # Amber

        else:
            color = colors.HexColor('#16a34a')  # Green

        # Draw background circle (light grey)
        bg_circle             = Circle(center_x, center_y, outer_radius)
        bg_circle.fillColor   = colors.HexColor('#f0f0f0')
        bg_circle.strokeColor = None

        d.add(bg_circle)

        # Draw colored arc representing the score percentage: The arc is drawn from 0 degrees (3 o'clock) clockwise
        sweep_angle           = (score / 100.0) * 360

        # Start angle is 90 degrees counter-clockwise from 3 o'clock (i.e., 12 o'clock)
        start_angle           = 90

        # Clockwise direction
        end_angle             = start_angle - sweep_angle  

        # Ensure start angle is greater than end angle for clockwise sweep
        if (start_angle < end_angle):
            end_angle = start_angle - sweep_angle
            extent    = -sweep_angle

        else:
            # Clockwise sweep
            extent = -sweep_angle 

        # Create a path for the arc (ring segment)
        p             = Path()

        # Calculate start and end points using trigonometry
        start_rad     = math.radians(start_angle)

        # Correct end angle for clockwise
        end_rad       = math.radians(start_angle - sweep_angle)

        # Move to the outer perimeter at the start angle
        start_outer_x = center_x + outer_radius * math.cos(start_rad)
        start_outer_y = center_y + outer_radius * math.sin(start_rad)

        p.moveTo(start_outer_x, start_outer_y)
        
        # At least 10 segments, or 1 per 5 degrees of sweep
        num_segments  = max(10, int(sweep_angle / 5)) 
        angle_step    = sweep_angle / num_segments

        # Draw outer arc as line segments
        for i in range(1, num_segments + 1):
            # Clockwise
            current_angle_deg = start_angle - (i * angle_step) 
            current_angle_rad = math.radians(current_angle_deg)
            x                 = center_x + outer_radius * math.cos(current_angle_rad)
            y                 = center_y + outer_radius * math.sin(current_angle_rad)

            p.lineTo(x, y)

        # Draw inner arc as line segments (reverse direction)
        for i in range(num_segments, -1, -1):
            # Clockwise
            current_angle_deg = start_angle - (i * angle_step)
            current_angle_rad = math.radians(current_angle_deg)
            x                 = center_x + inner_radius * math.cos(current_angle_rad)
            y                 = center_y + inner_radius * math.sin(current_angle_rad)
            p.lineTo(x, y)

        p.closePath()
        p.fillColor   = color
        p.strokeColor = None
        d.add(p)

        # Draw inner white circle : Slightly smaller to fit inside the ring
        inner_circle             = Circle(center_x, center_y, inner_radius - 2) 
        inner_circle.fillColor   = colors.white
        inner_circle.strokeColor = None
        d.add(inner_circle)

        # Draw score text in the center
        score_text               = String(center_x, center_y - 10, str(score), textAnchor='middle')
        score_text.fontSize      = 36
        score_text.fontName      = 'Helvetica-Bold'
        score_text.fillColor     = color
        d.add(score_text)

        # Draw "/100" text slightly below the score
        subtitle_text            = String(center_x, center_y - 28, "/100", textAnchor='middle')
        subtitle_text.fontSize   = 16
        subtitle_text.fontName   = 'Helvetica'
        subtitle_text.fillColor  = colors.HexColor('#666666')
        d.add(subtitle_text)

        return d


    def _get_risk_color(self, score: int) -> colors.Color:
        """
        Get color based on risk score
        """
        if (score >= 80):
            return colors.HexColor('#dc2626')

        elif (score >= 60):
            return colors.HexColor('#f97316')

        elif (score >= 40):
            return colors.HexColor('#ca8a04')

        else:
            return colors.HexColor('#16a34a')


    def _create_header_footer(self, canvas, doc):
        """
        Add header and footer to each page with consistent positioning
        """
        canvas.saveState()

        # Header
        canvas.setFont('Helvetica-Bold', 12)
        canvas.setFillColor(colors.black)
        canvas.drawString(self.margin_left, self.page_height - 0.5 * inch, "AI Powered Contract Risk Analysis Report")

        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#666666'))

        # Page number
        page_num = f"Page {doc.page}"
        canvas.drawString(self.page_width - self.margin_right - 1*inch, 0.5 * inch, page_num)

        # Disclaimer
        disclaimer = "For informational purposes only. Not legal advice."
        canvas.drawCentredString(self.page_width / 2.0, 0.5 * inch, disclaimer)

        canvas.restoreState()


    def generate_report(self, analysis_result: Dict[str, Any], output_path: Optional[str] = None) -> BytesIO:
        """
        Generate PDF report from analysis results

        Arguments:
        ----------
            analysis_result { dict } : Analysis result dictionary from the API

            output_path      { str } : Optional file path to save PDF

        Returns:
        --------
                  { BytesIO }        : Buffer containing the PDF
        """
        # Create buffer
        buffer = BytesIO()

        # Create document
        doc    = SimpleDocTemplate(buffer if not output_path else output_path,
                                   pagesize     = letter,
                                   rightMargin  = self.margin_right,
                                   leftMargin   = self.margin_left,
                                   topMargin    = self.margin_top,
                                   bottomMargin = self.margin_bottom,
                                  )

        # Build story
        story   = list()

        # Page 1: Title, Risk Score, Executive Summary, Keywords
        story.extend(self._build_page_1(analysis_result))
        story.append(PageBreak())

        # Page 2: Unfavorable Terms, Missing Protections
        story.extend(self._build_page_2(analysis_result))
        story.append(PageBreak())

        # Page 3: Negotiation Points
        story.extend(self._build_page_3(analysis_result))
        story.append(PageBreak())

        # Page 4: Risk Category Breakdown Table
        story.extend(self._build_page_4(analysis_result))
        story.append(PageBreak())

        # Page 5: Clause Interpretations
        story.extend(self._build_page_5(analysis_result))
        story.append(PageBreak())

        # Page 6+: Detailed Clause Analysis
        story.extend(self._build_clause_analysis_pages(analysis_result))

        # Build PDF
        doc.build(story, onFirstPage = self._create_header_footer, onLaterPages = self._create_header_footer)

        # If using buffer, seek to beginning
        if not output_path:
            buffer.seek(0)
            return buffer

        return buffer


    def _build_page_1(self, result: Dict) -> List:
        """
        Build page 1 content: Title, Risk Score, Executive Summary, Keywords
        """
        elements       = list()

        # Title
        elements.append(Paragraph("AI Contract Risk Analysis Report", self.styles['ReportTitle']))
        elements.append(Spacer(1, 0.1*inch))

        # Contract Info
        classification = result.get('classification', {})
        contract_type  = classification.get('subcategory', 'Unknown')
        confidence     = classification.get('confidence', 0) * 100
        
        info_text      = f"<b>Contract Type:</b> {contract_type.replace('_', ' ').title()} | <b>Confidence:</b> {confidence:.1f}%"
        elements.append(Paragraph(info_text, self.styles['CustomBodyText']))
        elements.append(Spacer(1, 0.2*inch))

        # Risk Score Circle and Executive Summary Side-by-Side
        risk_analysis = result.get('risk_analysis', {})
        overall_score = risk_analysis.get('overall_score', 0)
        risk_level    = risk_analysis.get('risk_level', 'UNKNOWN')
        
        score_frame   = KeepInFrame(1.5*inch, 1.5*inch, [self._draw_risk_score_circle(overall_score)])
        summary_para  = Paragraph(f"<b>Overall Risk Score: {overall_score}/100 ({risk_level})</b><br/><br/>{result.get('executive_summary', 'No executive summary available.')}",
                                 self.styles['CustomBodyText']
                                )

        top_row       = PlatypusTable([[score_frame, summary_para]], colWidths=[1.6*inch, 4.5*inch])

        top_row.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                     ('LEFTPADDING', (0, 0), (-1, -1), 0),
                                     ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                                     ('TOPPADDING', (0, 0), (-1, -1), 0),
                                     ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                                   ])
                         )

        elements.append(top_row)
        elements.append(Spacer(1, 0.3*inch))

        # Detected Keywords
        detected_keywords = result.get('classification', {}).get('detected_keywords', [])
        
        if detected_keywords:
            elements.append(Paragraph("Detected Keywords", self.styles['SectionHeading']))
            keywords_text = ", ".join([f"<font color='#1e40af'><b>{kw}</b></font>" for kw in detected_keywords[:15]])  # Show first 15 keywords
            elements.append(Paragraph(keywords_text, self.styles['CustomBodyText']))

        return elements


    def _build_page_2(self, result: Dict) -> List:
        """
        Build page 2: Unfavorable Terms and Missing Protections
        """
        elements    = list()

        # Unfavorable Terms
        elements.append(Paragraph("Unfavorable Terms", self.styles['SectionHeading']))

        unfav_terms = result.get('unfavorable_terms', [])

        if unfav_terms:
            # Sort by severity and risk score
            sorted_terms = sorted(unfav_terms, key=lambda x: (x.get('severity', 'low') != 'high', -x.get('risk_score', 0)))
            
            for term in sorted_terms:
                severity    = term.get('severity', 'unknown').upper()
                risk_score  = term.get('risk_score', 0)
                clause_ref  = term.get('clause_reference', 'N/A')
                explanation = term.get('explanation', 'No explanation provided.')
                
                term_text   = f"<b>{clause_ref}</b> | <font color='{self._get_severity_color(severity).hexval()}'>{severity} SEVERITY</font> | Risk Score: {risk_score}<br/>{explanation}"
                
                elements.append(Paragraph(term_text, self.styles['BulletPoint']))
        
        else:
            elements.append(Paragraph("No unfavorable terms identified.", self.styles['CustomBodyText']))

        elements.append(Spacer(1, 0.2*inch))
        
        # Missing Protections
        elements.append(Paragraph("Missing Protections", self.styles['SectionHeading']))

        missing_protections = result.get('missing_protections', [])
        
        if missing_protections:
            # Sort by importance and risk score
            sorted_protections = sorted(missing_protections, key=lambda x: (x.get('importance', 'medium') != 'critical', -x.get('risk_score', 0)))
            
            for prot in sorted_protections:
                importance      = prot.get('importance', 'medium').upper()
                risk_score      = prot.get('risk_score', 0)
                protection_name = prot.get('protection', 'N/A')
                explanation     = prot.get('explanation', 'No explanation provided.')
                
                prot_text       = f"<b>{protection_name}</b> | <font color='{self._get_importance_color(importance).hexval()}'>{importance} IMPORTANCE</font> | Risk Score: {risk_score}<br/>{explanation}"
                
                elements.append(Paragraph(prot_text, self.styles['BulletPoint']))
        
        else:
            elements.append(Paragraph("No missing protections identified.", self.styles['CustomBodyText']))

        return elements


    def _build_page_3(self, result: Dict) -> List:
        """
        Build page 3: Negotiation Points as a structured table
        """
        elements           = list()

        elements.append(Paragraph("Negotiation Strategy", self.styles['SectionHeading']))
        
        negotiation_points = result.get('negotiation_points', [])

        if negotiation_points:
            # Prepare table data: Priority, Issue, Current Language, Proposed Language
            table_data    = [[Paragraph('<b>Priority</b>', self.styles['TableHeader']),
                              Paragraph('<b>Issue</b>', self.styles['TableHeader']),
                              Paragraph('<b>Current</b>', self.styles['TableHeader']),
                              Paragraph('<b>Proposed</b>', self.styles['TableHeader']),
                            ]]
            
            # Sort by priority
            sorted_points = sorted(negotiation_points, key=lambda x: x.get('priority', 999))
            
            for point in sorted_points:
                priority  = str(point.get('priority', 'N/A'))
                issue     = Paragraph(point.get('issue', 'N/A'), self.styles['TableCell'])
                current   = Paragraph(self._truncate_text(point.get('current_language', 'Not specified'), 100), self.styles['TableCell'])
                proposed  = Paragraph(self._truncate_text(point.get('proposed_language', 'Request balanced language'), 100), self.styles['TableCell'])
                
                table_data.append([Paragraph(priority, self.styles['TableCell']), issue, current, proposed])

            # Create the table with appropriate column widths
            col_widths = [0.5*inch, 1.5*inch, 1.5*inch, 2*inch]
            table      = Table(table_data, colWidths=col_widths)
            
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f5f5f5')),
                                       ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1a1a1a')),
                                       ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                                       ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                       ('FONTSIZE', (0,0), (-1,0), 10),
                                       ('BOTTOMPADDING', (0,0), (-1,0), 12),
                                       ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#d1d5db')),
                                       ('VALIGN', (0,0), (-1,-1), 'TOP'),
                                       ('LEFTPADDING', (0,0), (-1,-1), 6),
                                       ('RIGHTPADDING', (0,0), (-1,-1), 6),
                                       ('TOPPADDING', (0,0), (-1,-1), 6),
                                       ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                                     ])
                          )

            elements.append(table)

        else:
            elements.append(Paragraph("No negotiation points available.", self.styles['CustomBodyText']))

        return elements

    
    def _build_page_4(self, result: Dict) -> List:
        """
        Build page 4: Risk Category Breakdown Table
        """
        elements       = list()

        elements.append(Paragraph("Risk Category Breakdown", self.styles['SectionHeading']))
        
        risk_breakdown = result.get('risk_analysis', {}).get('risk_breakdown', [])
        
        if risk_breakdown:
            # Prepare table data
            table_data = [[Paragraph('<b>Category</b>', self.styles['TableHeader']),
                            Paragraph('<b>Score</b>', self.styles['TableHeader']),
                            Paragraph('<b>Summary</b>', self.styles['TableHeader']),
                          ]]

            for item in risk_breakdown:
                category     = item.get('category', 'N/A').replace('_', ' ').title()
                score        = item.get('score', 0)
                summary      = item.get('summary', 'No summary available.')
                score_color  = self._get_risk_color(score)
                score_para   = Paragraph(f'<font color="{score_color.hexval()}">{score}/100</font>', self.styles['TableHeader'])
                summary_para = Paragraph(summary, self.styles['TableCell'])
                
                table_data.append([Paragraph(category, self.styles['TableCell']), score_para, summary_para])

            # Create table
            col_widths = [2*inch, 1*inch, 3.5*inch] 
            table      = Table(table_data, colWidths = col_widths)

            # Table Style
            table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f5f5')),
                                       ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
                                       ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                       ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                                       ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                       ('FONTSIZE', (0, 0), (-1, 0), 10),
                                       ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                       ('TOPPADDING', (0, 1), (-1, -1), 8),
                                       ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
                                       ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d1d5db')),
                                       ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                     ])
                          )

            elements.append(table)

        else:
            elements.append(Paragraph("No risk breakdown data available.", self.styles['CustomBodyText']))
        
        return elements


    def _build_page_5(self, result: Dict) -> List:
        """
        Build page 5: Clause Interpretations
        """
        elements               = list()

        elements.append(Paragraph("Clause Interpretations", self.styles['SectionHeading']))
        
        clause_interpretations = result.get('clause_interpretations', [])

        if clause_interpretations:
            for i, clause in enumerate(clause_interpretations): 
                ref           = clause.get('clause_reference', 'N/A')
                plain_english = clause.get('plain_english_summary', 'No summary available.')
                favorability  = clause.get('favorability', 'neutral')
                
                # Color code based on favorability
                fav_color     = self._get_favorability_color(favorability)
                
                clause_text   = f"<b>{ref}</b> | <font color='{fav_color.hexval()}'>{favorability.upper()}</font><br/>{plain_english}"
                
                elements.append(Paragraph(clause_text, self.styles['BulletPoint']))
                
                # Add key points if available
                key_points = clause.get('key_points', [])
                if key_points:
                    for point in key_points:  
                        elements.append(Paragraph(f"• {point}", self.styles['SmallText']))
                
                elements.append(Spacer(1, 0.1*inch))
                
                # Break if we've added enough content: After 8 clauses, it's usually enough for one page
                if i >= 7:  
                    remaining = len(clause_interpretations) - 8
                    if (remaining > 0):
                        elements.append(Paragraph(f"... and {remaining} more clause interpretations", self.styles['SmallText']))
                    break

        else:
            elements.append(Paragraph("No clause interpretations available.", self.styles['CustomBodyText']))

        return elements


    def _build_clause_analysis_pages(self, analysis_result):
        """
        Build dynamic pages for detailed clause-by-clause analysis
        """
        story   = list()
        clauses = analysis_result.get('clauses', [])

        if not clauses:
            story.append(Paragraph("No clauses analyzed.", self.styles['CustomBodyText']))
            return story

        story.append(Paragraph("Detailed Clause Analysis", self.styles['SectionHeading']))

        for i, clause in enumerate(clauses):
            # Use KeepTogether to ensure a clause block stays on one page if possible
            clause_elements = list()

            # Clause Reference and Category as Header
            ref           = clause.get('reference', 'N/A')
            category      = clause.get('category', 'N/A').replace('_', ' ').title()
            confidence    = clause.get('confidence', 0)
            risk_score    = clause.get('risk_score', 0)
            
            ref_cat_text  = f"{ref} • {category} | Confidence: {confidence:.1f} | Risk Score: {risk_score}"
            clause_header = Paragraph(ref_cat_text, self.styles['SubSectionHeading'])
            
            clause_elements.append(clause_header)

            # Original Clause Text
            clause_text = clause.get('text', 'No text available.')
            
            # Truncate very long clause text
            if len(clause_text) > 500:
                clause_text = clause_text[:500] + "... [truncated]"
            
            clause_para = Paragraph(f"<b>Original Text:</b> {clause_text}", self.styles['CustomBodyText'])
            
            clause_elements.append(clause_para)

            # Risk Indicators (if any)
            risk_inds = clause.get('risk_indicators', [])
            if risk_inds:
                ri_text = f"<b>Risk Indicators:</b> {', '.join(risk_inds)}"
                ri_para = Paragraph(ri_text, self.styles['RiskIndicator'])
                
                clause_elements.append(ri_para)

            # Add Spacer between clauses
            clause_elements.append(Spacer(1, 0.15 * inch))

            # Wrap in KeepTogether
            kt_flowable = KeepTogether(clause_elements)
            story.append(kt_flowable)

            # Add page break every 5 clauses to prevent overflow
            if (i + 1) % 5 == 0 and (i + 1) < len(clauses):
                story.append(PageBreak())

        return story


    def _get_severity_color(self, severity: str) -> colors.Color:
        """
        Get color based on severity level
        """
        severity = severity.lower()
        if (severity == 'high'):
            return colors.HexColor('#dc2626')

        elif (severity == 'medium'):
            return colors.HexColor('#f97316')

        else:
            return colors.HexColor('#16a34a')


    def _get_importance_color(self, importance: str) -> colors.Color:
        """
        Get color based on importance level
        """
        importance = importance.lower()

        if (importance == 'critical'):
            return colors.HexColor('#dc2626')

        elif (importance == 'high'):
            return colors.HexColor('#f97316')

        elif (importance == 'medium'):
            return colors.HexColor('#ca8a04')

        else:
            return colors.HexColor('#16a34a')


    def _get_favorability_color(self, favorability: str) -> colors.Color:
        """
        Get color based on favorability
        """
        favorability = favorability.lower()
        if favorability == 'favorable':
            return colors.HexColor('#16a34a')
        elif favorability == 'unfavorable':
            return colors.HexColor('#dc2626')
        else:
            return colors.HexColor('#ca8a04')


    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text to specified length
        """
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."


def generate_pdf_report(analysis_result: Dict[str, Any], output_path: Optional[str] = None) -> BytesIO:
    """
    Convenience function to generate PDF report

    Arguments:
    ----------
        analysis_result { dict } : Complete analysis result from the API

        output_path      { str } : Optional file path to save PDF

    Returns:
    --------
              { BytesIO }        : Buffer containing the PDF
    """
    generator = PDFReportGenerator()

    return generator.generate_report(analysis_result, output_path)