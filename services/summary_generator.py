# DEPENDENCIES
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from utils.logger import ContractAnalyzerLogger
from model_manager.llm_manager import LLMManager, LLMProvider
from services.risk_analyzer import RiskScore
from services.llm_interpreter import RiskInterpretation
from services.negotiation_engine import NegotiationPlaybook
from services.contract_classifier import ContractCategory

logger = ContractAnalyzerLogger.get_logger()


@dataclass
class SummaryContext:
    """
    Context data for comprehensive summary generation
    """
    contract_type: str
    risk_score: int
    risk_level: str
    category_scores: Dict[str, int]
    unfavorable_terms: List[Dict]
    missing_protections: List[Dict]
    clauses: List
    key_findings: List[str]
    # NEW: Full pipeline integration
    risk_interpretation: Optional[RiskInterpretation] = None
    negotiation_playbook: Optional[NegotiationPlaybook] = None
    contract_text_preview: Optional[str] = None
    contract_metadata: Optional[Dict[str, Any]] = None


class SummaryGenerator:
    """
    LLM-powered executive summary generator for contract analysis
    Generates professional, detailed executive summaries using ALL pipeline outputs
    """
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """
        Initialize the summary generator
        
        Args:
            llm_manager: LLM manager instance (if None, creates one with default settings)
        """
        self.llm_manager = llm_manager or LLMManager()
        self.logger = ContractAnalyzerLogger.get_logger()
        
        logger.info("Summary generator initialized")

    # ENHANCED: Main entry point with full pipeline integration
    def generate_comprehensive_summary(self,
                                     contract_text: str,
                                     classification: ContractCategory,
                                     risk_analysis: RiskScore,
                                     risk_interpretation: RiskInterpretation,
                                     negotiation_playbook: NegotiationPlaybook,
                                     unfavorable_terms: List[Dict],
                                     missing_protections: List[Dict],
                                     clauses: List) -> str:
        """
        Generate comprehensive executive summary using ALL pipeline outputs
        
        Args:
            contract_text: Original contract text (for context)
            classification: Contract classification results
            risk_analysis: Complete risk analysis
            risk_interpretation: LLM-enhanced risk explanations
            negotiation_playbook: Comprehensive negotiation strategy
            unfavorable_terms: Detected unfavorable terms
            missing_protections: Missing protections
            clauses: Extracted clauses
            
        Returns:
            Generated executive summary string
        """
        try:
            # Prepare enhanced context with ALL pipeline data
            context = self._prepare_comprehensive_context(
                contract_text=contract_text,
                classification=classification,
                risk_analysis=risk_analysis,
                risk_interpretation=risk_interpretation,
                negotiation_playbook=negotiation_playbook,
                unfavorable_terms=unfavorable_terms,
                missing_protections=missing_protections,
                clauses=clauses
            )
            
            # Generate enhanced summary using LLM
            summary = self._generate_enhanced_summary(context)
            
            logger.info(f"Comprehensive executive summary generated - Risk: {context.risk_score}/100 ({context.risk_level})")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive summary: {e}")
            
            # Enhanced fallback with available data
            return self._generate_enhanced_fallback_summary(
                contract_text=contract_text,
                classification=classification,
                risk_analysis=risk_analysis,
                unfavorable_terms=unfavorable_terms,
                missing_protections=missing_protections
            )
    
    def _prepare_comprehensive_context(self,
                                    contract_text: str,
                                    classification: ContractCategory,
                                    risk_analysis: RiskScore,
                                    risk_interpretation: RiskInterpretation,
                                    negotiation_playbook: NegotiationPlaybook,
                                    unfavorable_terms: List[Dict],
                                    missing_protections: List[Dict],
                                    clauses: List) -> SummaryContext:
        """Prepare comprehensive context with ALL pipeline data"""
        
        # Extract contract preview
        contract_preview = contract_text[:500] + "..." if len(contract_text) > 500 else contract_text
        
        # Extract key findings from ALL sources
        key_findings = self._extract_comprehensive_findings(
            risk_analysis=risk_analysis,
            risk_interpretation=risk_interpretation,
            negotiation_playbook=negotiation_playbook,
            unfavorable_terms=unfavorable_terms,
            missing_protections=missing_protections,
            clauses=clauses
        )
        
        # Prepare metadata
        metadata = {
            "contract_length": len(contract_text),
            "clauses_analyzed": len(clauses),
            "critical_issues": len([t for t in unfavorable_terms if self._get_severity(t) == "critical"]),
            "walk_away_items": len(negotiation_playbook.walk_away_items) if negotiation_playbook else 0
        }
        
        return SummaryContext(
            contract_type=classification.category,
            risk_score=risk_analysis.overall_score,
            risk_level=risk_analysis.risk_level,
            category_scores=risk_analysis.category_scores,
            unfavorable_terms=unfavorable_terms,
            missing_protections=missing_protections,
            clauses=clauses,
            key_findings=key_findings,
            risk_interpretation=risk_interpretation,
            negotiation_playbook=negotiation_playbook,
            contract_text_preview=contract_preview,
            contract_metadata=metadata
        )
    
    def _extract_comprehensive_findings(self,
                                      risk_analysis: RiskScore,
                                      risk_interpretation: RiskInterpretation,
                                      negotiation_playbook: NegotiationPlaybook,
                                      unfavorable_terms: List[Dict],
                                      missing_protections: List[Dict],
                                      clauses: List) -> List[str]:
        """Extract comprehensive findings from ALL analysis components"""
        
        findings = []
        
        # 1. Overall risk context
        if risk_analysis.overall_score >= 80:
            findings.append("CRITICAL RISK LEVEL: Contract presents unacceptable risk requiring immediate attention")
        elif risk_analysis.overall_score >= 60:
            findings.append("HIGH RISK LEVEL: Significant concerns requiring substantial negotiation")
        
        # 2. Critical unfavorable terms
        critical_terms = [t for t in unfavorable_terms if self._get_severity(t) == "critical"]
        if critical_terms:
            findings.append(f"{len(critical_terms)} CRITICAL unfavorable terms identified")
            for term in critical_terms[:2]:
                term_name = self._get_term_name(term)
                findings.append(f"Critical: {term_name}")
        
        # 3. Critical missing protections
        critical_protections = [p for p in missing_protections if self._get_importance(p) == "critical"]
        if critical_protections:
            findings.append(f"{len(critical_protections)} CRITICAL protections missing")
            for prot in critical_protections[:2]:
                prot_name = self._get_protection_name(prot)
                findings.append(f"Missing: {prot_name}")
        
        # 4. High-risk categories
        high_risk_categories = [cat for cat, score in risk_analysis.category_scores.items() 
                               if score >= 70]
        if high_risk_categories:
            findings.append(f"High-risk categories: {', '.join(high_risk_categories)}")
        
        # 5. Walk-away items from negotiation playbook
        if negotiation_playbook and negotiation_playbook.walk_away_items:
            findings.append(f"{len(negotiation_playbook.walk_away_items)} potential deal-breakers identified")
        
        # 6. Key concerns from risk interpretation
        if risk_interpretation and risk_interpretation.key_concerns:
            top_concerns = risk_interpretation.key_concerns[:2]
            for concern in top_concerns:
                findings.append(f"Key concern: {concern}")
        
        return findings[:8]  # Return top 8 findings
    
    def _generate_enhanced_summary(self, context: SummaryContext) -> str:
        """Generate enhanced summary using comprehensive context"""
        
        prompt = self._build_enhanced_summary_prompt(context)
        system_prompt = self._build_enhanced_system_prompt()
        
        try:
            response = self.llm_manager.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=500,  # Increased for comprehensive summary
                json_mode=False
            )
            
            if response.success and response.text.strip():
                return self._clean_summary_response(response.text)
            else:
                raise ValueError(f"LLM generation failed: {response.error_message}")
                
        except Exception as e:
            logger.error(f"Enhanced LLM summary generation failed: {e}")
            # Fallback to basic summary
            return self._generate_enhanced_fallback_summary_from_context(context)
    
    def _build_enhanced_system_prompt(self) -> str:
        """Build enhanced system prompt for comprehensive summary generation"""
        
        return """You are a senior legal analyst and strategic advisor specializing in contract risk assessment. 

YOUR ROLE:
Generate comprehensive, professional executive summaries that synthesize technical risk analysis with practical business implications.

KEY REQUIREMENTS:
1. Write in formal, professional business language suitable for executives
2. Synthesize ALL analysis components into cohesive narrative
3. Focus on strategic implications and decision-making
4. Maintain objective, factual tone while highlighting critical risks
5. Keep summary length between 150-300 words
6. Structure: Overall assessment → Critical risks → Strategic implications → Recommended approach

CONTENT FOCUS:
- Start with overall risk assessment and contract type context
- Highlight 2-3 most critical risks with practical consequences  
- Mention key missing protections and their business impact
- Reference negotiation strategy and deal-breakers
- Conclude with clear recommended next steps

WRITING STYLE:
- Executive-level business language
- Focus on decision-making implications
- Avoid markdown formatting
- Be direct, actionable, and strategic
- Connect legal risks to business outcomes

OUTPUT FORMAT:
Return only the executive summary text, no headings, no bullet points, no role rescription, just clean paragraph text. Also write the summary in passive voice only."""

    def _build_enhanced_summary_prompt(self, context: SummaryContext) -> str:
        """Build detailed prompt for comprehensive summary generation"""
        
        # Build comprehensive context sections
        risk_context = self._build_enhanced_risk_context(context)
        critical_issues = self._build_critical_issues_context(context)
        strategic_context = self._build_strategic_context(context)
        negotiation_context = self._build_negotiation_context(context)
        
        prompt = f"""
COMPREHENSIVE CONTRACT ANALYSIS:

{risk_context}

{critical_issues}

{strategic_context}

{negotiation_context}

GENERATION INSTRUCTIONS:
Based on the comprehensive analysis above, write a professional executive summary that:

1. Starts with overall risk assessment for this {context.contract_type} agreement
2. Highlights the most critical risks and their business implications
3. Mentions key missing protections and unfavorable terms
4. References the negotiation strategy and potential deal-breakers  
5. Provides clear, actionable recommendations for next steps

Focus on synthesizing all analysis components into a cohesive strategic assessment that supports executive decision-making.
"""
        return prompt
    
    def _build_enhanced_risk_context(self, context: SummaryContext) -> str:
        """Build enhanced risk assessment context"""
        
        risk_level_descriptions = {
            "CRITICAL": "CRITICAL level of risk requiring immediate executive attention",
            "HIGH": "HIGH level of risk requiring significant review and negotiation",
            "MEDIUM": "MODERATE level of risk with specific concerns to address",
            "LOW": "LOW level of risk, generally favorable with minor improvements needed"
        }
        
        risk_desc = risk_level_descriptions.get(context.risk_level, "Requires professional review")
        
        text = f"OVERALL RISK ASSESSMENT:\n"
        text += f"- Risk Score: {context.risk_score}/100 ({risk_desc})\n"
        text += f"- Contract Type: {context.contract_type.replace('_', ' ').title()}\n"
        text += f"- Analysis Scope: {context.contract_metadata.get('clauses_analyzed', 0)} clauses analyzed\n"
        
        # Top risk categories
        if context.category_scores:
            high_risk_categories = [(cat, score) for cat, score in context.category_scores.items() 
                                   if score >= 60]
            if high_risk_categories:
                text += "- Highest Risk Categories:\n"
                for category, score in sorted(high_risk_categories, key=lambda x: x[1], reverse=True)[:3]:
                    category_name = category.replace('_', ' ').title()
                    text += f"  * {category_name}: {score}/100\n"
        
        return text
    
    def _build_critical_issues_context(self, context: SummaryContext) -> str:
        """Build context about critical issues"""
        
        text = "CRITICAL ISSUES IDENTIFIED:\n"
        
        # Critical unfavorable terms
        critical_terms = [t for t in context.unfavorable_terms if self._get_severity(t) == "critical"]
        if critical_terms:
            text += f"- Critical Unfavorable Terms: {len(critical_terms)}\n"
            for term in critical_terms[:2]:
                term_name = self._get_term_name(term)
                explanation = self._get_explanation(term)
                text += f"  * {term_name}: {explanation}\n"
        
        # Critical missing protections
        critical_protections = [p for p in context.missing_protections if self._get_importance(p) == "critical"]
        if critical_protections:
            text += f"- Critical Missing Protections: {len(critical_protections)}\n"
            for prot in critical_protections[:2]:
                prot_name = self._get_protection_name(prot)
                explanation = self._get_explanation(prot)
                text += f"  * {prot_name}: {explanation}\n"
        
        # Key concerns from risk interpretation
        if context.risk_interpretation and context.risk_interpretation.key_concerns:
            text += f"- Key Strategic Concerns: {len(context.risk_interpretation.key_concerns)}\n"
            for concern in context.risk_interpretation.key_concerns[:2]:
                text += f"  * {concern}\n"
        
        if not critical_terms and not critical_protections:
            text += "- No critical issues identified\n"
        
        return text
    
    def _build_strategic_context(self, context: SummaryContext) -> str:
        """Build strategic context from risk interpretation"""
        
        text = "STRATEGIC ASSESSMENT:\n"
        
        if context.risk_interpretation:
            text += f"- Overall Risk Explanation: {context.risk_interpretation.overall_risk_explanation}\n"
            
            if context.risk_interpretation.market_comparison:
                text += f"- Market Context: {context.risk_interpretation.market_comparison}\n"
        
        # Contract complexity context
        if context.contract_metadata:
            if context.contract_metadata['contract_length'] > 10000:
                text += "- Complex Agreement: Extensive contract requiring detailed review\n"
            elif context.contract_metadata['critical_issues'] > 0:
                text += "- High Attention Required: Contains critical issues needing resolution\n"
        
        return text
    
    def _build_negotiation_context(self, context: SummaryContext) -> str:
        """Build negotiation strategy context"""
        
        text = "NEGOTIATION STRATEGY:\n"
        
        if context.negotiation_playbook:
            text += f"- Overall Approach: {context.negotiation_playbook.overall_strategy}\n"
            
            if context.negotiation_playbook.walk_away_items:
                text += f"- Deal-Breakers: {len(context.negotiation_playbook.walk_away_items)} critical items\n"
                for item in context.negotiation_playbook.walk_away_items[:2]:
                    text += f"  * {item}\n"
            
            if context.negotiation_playbook.critical_points:
                text += f"- Priority Negotiation Points: {len(context.negotiation_playbook.critical_points)}\n"
            
            text += f"- Timing Guidance: {context.negotiation_playbook.timing_guidance}\n"
        else:
            text += "- Standard negotiation approach recommended\n"
        
        return text
    
    def _clean_summary_response(self, text: str) -> str:
        """Clean and format the LLM response"""
        
        # Remove any markdown formatting
        text = text.replace('**', '').replace('*', '').replace('#', '')
        
        # Remove common LLM artifacts and empty lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith(('executive summary', 'summary:', 'here is', 'based on', 'certainly')):
                cleaned_lines.append(line)
        
        # Join into coherent paragraph
        summary = ' '.join(cleaned_lines)
        
        # Ensure proper sentence structure
        if summary:
            if not summary[0].isupper():
                summary = summary[0].upper() + summary[1:]
            
            if not summary.endswith(('.', '!', '?')):
                summary += '.'
        
        return summary
    
    def _generate_enhanced_fallback_summary(self,
                                          contract_text: str,
                                          classification: ContractCategory,
                                          risk_analysis: RiskScore,
                                          unfavorable_terms: List[Dict],
                                          missing_protections: List[Dict]) -> str:
        """Generate enhanced fallback summary"""
        
        contract_type_display = classification.category.replace('_', ' ').title()
        
        # Count critical items
        critical_terms = len([t for t in unfavorable_terms if self._get_severity(t) == "critical"])
        critical_protections = len([p for p in missing_protections if self._get_importance(p) == "critical"])
        
        # Enhanced risk assessment
        if risk_analysis.overall_score >= 80:
            risk_assessment = f"This {contract_type_display} presents a CRITICAL level of risk"
            action = "requires immediate executive attention and significant revision before consideration"
        elif risk_analysis.overall_score >= 60:
            risk_assessment = f"This {contract_type_display} presents a HIGH level of risk" 
            action = "requires careful legal review and substantial negotiation to mitigate key concerns"
        elif risk_analysis.overall_score >= 40:
            risk_assessment = f"This {contract_type_display} presents a MODERATE level of risk"
            action = "requires professional review and selective negotiation on specific provisions"
        else:
            risk_assessment = f"This {contract_type_display} presents a LOW level of risk"
            action = "appears generally reasonable but should undergo standard legal review"
        
        summary = f"{risk_assessment} with an overall risk score of {risk_analysis.overall_score}/100. "
        summary += f"The agreement {action}. "
        
        # Add critical items context
        if critical_terms > 0:
            summary += f"Analysis identified {critical_terms} critical unfavorable terms "
            if critical_protections > 0:
                summary += f"and {critical_protections} critical missing protections. "
            else:
                summary += f"and {len(missing_protections)} missing standard protections. "
        else:
            summary += f"Review identified {len(unfavorable_terms)} areas for improvement. "
        
        # Add high-risk categories context
        high_risk_categories = [cat for cat, score in risk_analysis.category_scores.items() if score >= 60]
        if high_risk_categories:
            category_names = [cat.replace('_', ' ').title() for cat in high_risk_categories[:2]]
            summary += f"Particular attention should be given to {', '.join(category_names)} provisions. "
        
        summary += "Proceed with the detailed negotiation strategy and risk mitigation recommendations provided in the full analysis."
        
        return summary
    
    def _generate_enhanced_fallback_summary_from_context(self, context: SummaryContext) -> str:
        """Generate fallback summary from context object"""
        return self._generate_enhanced_fallback_summary(
            contract_text=context.contract_text_preview or "",
            classification=type('MockClassification', (), {'category': context.contract_type})(),
            risk_analysis=type('MockRiskAnalysis', (), {
                'overall_score': context.risk_score,
                'risk_level': context.risk_level,
                'category_scores': context.category_scores
            })(),
            unfavorable_terms=context.unfavorable_terms,
            missing_protections=context.missing_protections
        )
    
    # Helper methods for safe attribute access
    def _get_severity(self, term) -> str:
        """Safely get severity from term object or dict"""
        try:
            if hasattr(term, 'severity'):
                return term.severity
            else:
                return term.get('severity', 'unknown')
        except (AttributeError, KeyError):
            return 'unknown'
    
    def _get_importance(self, protection) -> str:
        """Safely get importance from protection object or dict"""
        try:
            if hasattr(protection, 'importance'):
                return protection.importance
            else:
                return protection.get('importance', 'unknown')
        except (AttributeError, KeyError):
            return 'unknown'
    
    def _get_term_name(self, term) -> str:
        """Safely get term name"""
        try:
            if hasattr(term, 'term'):
                return term.term
            else:
                return term.get('term', 'Unknown Term')
        except (AttributeError, KeyError):
            return 'Unknown Term'
    
    def _get_protection_name(self, protection) -> str:
        """Safely get protection name"""
        try:
            if hasattr(protection, 'protection'):
                return protection.protection
            else:
                return protection.get('protection', 'Unknown Protection')
        except (AttributeError, KeyError):
            return 'Unknown Protection'
    
    def _get_explanation(self, item) -> str:
        """Safely get explanation"""
        try:
            if hasattr(item, 'explanation'):
                return item.explanation
            else:
                return item.get('explanation', 'No explanation available')
        except (AttributeError, KeyError):
            return 'No explanation available'

    # Keep original method for backward compatibility
    def generate_executive_summary(self, 
                                 classification: Dict,
                                 risk_analysis: Dict,
                                 unfavorable_terms: List[Dict],
                                 missing_protections: List[Dict],
                                 clauses: List) -> str:
        """
        Original method for backward compatibility
        """
        # Convert dict inputs to appropriate types for the new method
        contract_category = type('ContractCategory', (), {
            'category': classification.get('category', 'contract')
        })()
        
        risk_score_obj = type('RiskScore', (), {
            'overall_score': risk_analysis.get('overall_score', 0),
            'risk_level': risk_analysis.get('risk_level', 'unknown'),
            'category_scores': risk_analysis.get('category_scores', {})
        })()
        
        return self.generate_comprehensive_summary(
            contract_text="",  # Not available in original method
            classification=contract_category,
            risk_analysis=risk_score_obj,
            risk_interpretation=None,
            negotiation_playbook=None,
            unfavorable_terms=unfavorable_terms,
            missing_protections=missing_protections,
            clauses=clauses
        )