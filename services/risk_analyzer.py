# DEPENDENCIES
import sys
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from pathlib import Path
from typing import Optional
from dataclasses import field
from dataclasses import dataclass
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import log_info
from utils.logger import log_error
from config.risk_rules import RiskRules
from config.risk_rules import ContractType
from utils.logger import ContractAnalyzerLogger
from services.term_analyzer import TermAnalyzer
from services.term_analyzer import UnfavorableTerm
from services.clause_extractor import RiskClauseExtractor
from services.protection_checker import ProtectionChecker
from services.protection_checker import MissingProtection
from services.contract_classifier import ContractCategory
from services.contract_classifier import ContractClassifier
from services.clause_extractor import ComprehensiveClauseExtractor



@dataclass
class RiskBreakdownItem:
    """
    Individual risk category breakdown
    """
    category : str
    score    : int  # 0-100
    summary  : str
    findings : List[str] = field(default_factory = list)
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        """
        return {"category" : self.category,
                "score"    : self.score,
                "summary"  : self.summary,
                "findings" : self.findings,
               }


@dataclass
class RiskScore:
    """
    Comprehensive risk score with detailed breakdown
    """
    overall_score        : int                       # 0-100
    risk_level           : str                       # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    category_scores      : Dict[str, int]            # Scores for each risk category
    risk_factors         : List[str]                 # Top risk factors
    detailed_findings    : Dict[str, List[str]]      # Findings by category
    benchmark_comparison : Dict[str, str]            # Market comparisons
    risk_breakdown       : List[RiskBreakdownItem]   # Detailed breakdown
    contract_type        : str                       # Detected contract type
    unfavorable_terms    : List[Dict]                # Unfavorable terms found
    missing_protections  : List[Dict]                # Missing critical protections
    

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization
        """
        return {"overall_score"        : self.overall_score,
                "risk_level"           : self.risk_level,
                "category_scores"      : self.category_scores,
                "risk_factors"         : self.risk_factors,
                "detailed_findings"    : self.detailed_findings,
                "benchmark_comparison" : self.benchmark_comparison,
                "risk_breakdown"       : [item.to_dict() for item in self.risk_breakdown],
                "contract_type"        : self.contract_type,
                "unfavorable_terms"    : self.unfavorable_terms,
                "missing_protections"  : self.missing_protections,
               }


class RiskAnalyzer:
    """
    Orchestrates all analysis components and calculates comprehensive risk scores
    
    Analysis Pipeline:
    1. Contract Classification
    2. Clause Extraction 
    3.Term Analysis 
    4. Protection Checking 
    5. Risk Scoring
    """
    def __init__(self, model_loader):
        """
        Initialize the risk analyzer with all required components
        
        Arguments:
        ----------
            model_loader : ModelLoader instance for accessing AI models
        """
        self.model_loader           = model_loader
        self.rules                  = RiskRules()
        self.logger                 = ContractAnalyzerLogger.get_logger()
        
        # Initialize all analysis components
        self.contract_classifier    = ContractClassifier(model_loader = model_loader)
        self.clause_extractor       = ComprehensiveClauseExtractor(model_loader = model_loader)
        self.risk_clause_extractor  = None  # Lazy initialization
        self.term_analyzer          = TermAnalyzer()
        self.protection_checker     = ProtectionChecker()
        
        log_info("RiskAnalyzer initialized - All components ready")
    

    @ContractAnalyzerLogger.log_execution_time("analyze_contract_risk")
    def analyze_contract_risk(self, contract_text: str) -> RiskScore:
        """
        MAIN ENTRY POINT: Comprehensive contract risk analysis
        
        Arguments:
        ----------
            contract_text { str } : Full contract text
        
        Returns:
        --------
               { RiskScore }      : Complete risk assessment with 0-100 score and detailed breakdown
        """
        
        log_info("Starting Comprehensive Contract Risk Analysis...", text_length = len(contract_text))
        
        # CONTRACT CLASSIFICATION
        contract_category   = self._classify_contract(contract_text = contract_text)
        log_info("Phase 1 complete: Contract classified", contract_type = contract_category.category)
        
        # CLAUSE EXTRACTION  
        clauses             = self._extract_clauses(contract_text     = contract_text, 
                                                    contract_category = contract_category,
                                                   )

        log_info("Phase 2 complete: Clauses extracted", num_clauses = len(clauses))
        
        # UNFAVORABLE TERM ANALYSIS
        unfavorable_terms   = self._analyze_unfavorable_terms(contract_text     = contract_text,
                                                              clauses           = clauses, 
                                                              contract_category = contract_category,
                                                             )

        log_info("Phase 3 complete: Unfavorable terms analyzed", num_unfavorable_terms = len(unfavorable_terms))
        
        # MISSING PROTECTIONS ANALYSIS
        missing_protections = self._analyze_missing_protections(contract_text     = contract_text, 
                                                                clauses           = clauses, 
                                                                contract_category = contract_category,
                                                               )

        log_info("Phase 4 complete: Missing protections analyzed", num_missing_protections = len(missing_protections))
        
        # RISK SCORING & AGGREGATION
        risk_score          = self._calculate_comprehensive_risk(contract_category    = contract_category,
                                                                 clauses              = clauses,
                                                                 unfavorable_terms    = unfavorable_terms,
                                                                 missing_protections  = missing_protections,
                                                                 contract_text        = contract_text,
                                                                )
        
        log_info("Risk Analysis Complete", 
                 overall_score = risk_score.overall_score,
                 risk_level    = risk_score.risk_level,
                 contract_type = risk_score.contract_type,
                )
        
        return risk_score
    

    def _classify_contract(self, contract_text: str):
        """
        Classify contract type
        """
        log_info("Classifying contract type...")
        
        try:
            classification = self.contract_classifier.classify_contract(contract_text = contract_text)
            
            log_info("Contract classification successful",
                     category    = classification.category,
                     confidence  = classification.confidence,
                     subcategory = classification.subcategory)
            
            return classification
            
        except Exception as e:
            log_error(e, context = {"component": "RiskAnalyzer", "operation": "contract_classification"})
            
            # Fallback to general classification
            return ContractCategory(category          = "general",
                                    subcategory       = None,
                                    confidence        = 0.5,
                                    reasoning         = ["Classification failed, using general fallback"],
                                    detected_keywords = [],
                                   )
    

    def _extract_clauses(self, contract_text: str, contract_category) -> List:
        """
        Extract clauses from contract
        """
        log_info("Extracting clauses from contract...")
        
        try:
            # Use comprehensive extractor for broad coverage
            clauses = self.clause_extractor.extract_clauses(contract_text = contract_text, 
                                                            max_clauses   = 25)
            
            log_info("Clause extraction successful",
                     total_clauses = len(clauses),
                     categories    = [c.category for c in clauses])
            
            return clauses
            
        except Exception as e:
            log_error(e, context = {"component": "RiskAnalyzer", "operation": "clause_extraction"})
            return []
    

    def _analyze_unfavorable_terms(self, contract_text: str, clauses: List, contract_category) -> List[UnfavorableTerm]:
        """
        Phase 3: Analyze for unfavorable terms
        """
        log_info("Analyzing unfavorable terms...")
        
        try:
            # Initialize term analyzer with contract type
            contract_type_enum = self._get_contract_type_enum(contract_category.category)
            self.term_analyzer = TermAnalyzer(contract_type = contract_type_enum)
            
            unfavorable_terms = self.term_analyzer.analyze_unfavorable_terms(contract_text = contract_text,
                                                                             clauses       = clauses)
            
            log_info("Unfavorable terms analysis successful",
                     total_terms = len(unfavorable_terms),
                     critical    = sum(1 for t in unfavorable_terms if t.severity == "critical"))
            
            return unfavorable_terms
            
        except Exception as e:
            log_error(e, context = {"component": "RiskAnalyzer", "operation": "unfavorable_terms_analysis"})
            return []
    

    def _analyze_missing_protections(self, contract_text: str, clauses: List, contract_category) -> List[MissingProtection]:
        """
        Phase 4: Analyze for missing protections
        """
        log_info("Analyzing missing protections...")
        
        try:
            # Initialize protection checker with contract type
            contract_type_enum = self._get_contract_type_enum(contract_category.category)
            self.protection_checker = ProtectionChecker(contract_type = contract_type_enum)
            
            missing_protections = self.protection_checker.check_missing_protections(contract_text = contract_text,
                                                                                    clauses       = clauses)
            
            log_info("Missing protections analysis successful",
                     total_missing = len(missing_protections),
                     critical      = sum(1 for p in missing_protections if p.importance == "critical"))
            
            return missing_protections
            
        except Exception as e:
            log_error(e, context = {"component": "RiskAnalyzer", "operation": "missing_protections_analysis"})
            return []
    

    def _calculate_comprehensive_risk(self, contract_category, clauses: List, unfavorable_terms: List[UnfavorableTerm], missing_protections: List[MissingProtection],
                                      contract_text: str) -> RiskScore:
        """
        Phase 5: Calculate comprehensive risk score using all analysis results
        """
        log_info("Calculating comprehensive risk score...")
        
        # Get contract type for risk rule adjustments
        contract_type_enum = self._get_contract_type_enum(contract_category.category)
        adjusted_weights   = self.rules.get_adjusted_weights(contract_type_enum)
        
        # Initialize scoring containers
        category_scores    = defaultdict(int)
        detailed_findings  = defaultdict(list)
        risk_factors       = list()
        
        # Calculate risk for each category
        for risk_category in adjusted_weights.keys():
            category_risk = self._calculate_category_risk(risk_category        = risk_category,
                                                          contract_type        = contract_type_enum,
                                                          clauses              = clauses,
                                                          unfavorable_terms    = unfavorable_terms,
                                                          missing_protections  = missing_protections,
                                                          contract_text        = contract_text,
                                                         )
            
            category_scores[risk_category]   = category_risk["score"]
            detailed_findings[risk_category] = category_risk["findings"]
            
            # Add to risk factors if high risk
            if (category_risk["score"] >= self.rules.RISK_THRESHOLDS["high"]):
                risk_factors.append(risk_category)
        
        # Calculate weighted overall score
        overall_score            = self._calculate_weighted_score(category_scores   = category_scores,
                                                                  adjusted_weights  = adjusted_weights)
        
        risk_level               = self._get_risk_level(score = overall_score)
        
        # Create risk breakdown
        risk_breakdown           = self._create_risk_breakdown(category_scores   = dict(category_scores),
                                                               detailed_findings = dict(detailed_findings))
        
        # Benchmark comparison
        benchmark_comparison     = self._compare_to_benchmarks(category_scores = category_scores,
                                                               contract_type   = contract_type_enum)
        
        # Prepare output data
        unfavorable_terms_dict   = [term.to_dict() for term in unfavorable_terms]
        missing_protections_dict = [protection.to_dict() for protection in missing_protections]
        
        return RiskScore(overall_score        = overall_score,
                         risk_level           = risk_level,
                         category_scores      = dict(category_scores),
                         risk_factors         = risk_factors,
                         detailed_findings    = dict(detailed_findings),
                         benchmark_comparison = benchmark_comparison,
                         risk_breakdown       = risk_breakdown,
                         contract_type        = contract_category.category,
                         unfavorable_terms    = unfavorable_terms_dict,
                         missing_protections  = missing_protections_dict,
                        )
    

    def _calculate_category_risk(self, risk_category: str, contract_type: ContractType, clauses: List, unfavorable_terms: List[UnfavorableTerm],
                                 missing_protections: List[MissingProtection], contract_text: str) -> Dict:
        """
        Calculate risk score for a specific category using all available data
        """
        base_score     = 0
        findings       = list()
        
        # Score from unfavorable terms in this category
        category_terms = [t for t in unfavorable_terms if (t.category == risk_category)]
        
        for term in category_terms:
            # Scale appropriately
            base_score += term.risk_score * 0.1  

            findings.append(f"{term.term}: {term.explanation}")
        
        # Score from missing protections affecting this category
        category_protections = [p for p in missing_protections if risk_category in p.categories]
        
        for protection in category_protections:
            base_score += protection.risk_score * 0.15
            findings.append(f"Missing: {protection.protection}")
        
        # Score from clauses in this category
        category_clauses = self._get_clauses_for_risk_category(clauses         = clauses,
                                                               risk_category   = risk_category,
                                                              )
        
        for clause in category_clauses:
            clause_risk = self._analyze_clause_risk(clause          = clause,
                                                    risk_category   = risk_category,
                                                    contract_type   = contract_type,
                                                   )
            base_score += clause_risk["score"]

            findings.extend(clause_risk["findings"])
        
        # Apply contract-type specific adjustments
        category_weight = self.rules.CONTRACT_TYPE_ADJUSTMENTS.get(contract_type.value, {}).get(risk_category, 1.0)
        adjusted_score  = base_score * category_weight
        
        # Cap score between 0-100
        final_score     = max(0, min(100, int(adjusted_score)))
        
        # Top 25 findings
        return {"score"    : final_score,
                "findings" : findings[:25] 
               }
    

    def _get_clauses_for_risk_category(self, clauses: List, risk_category: str) -> List:
        """
        Map clauses to risk categories
        """
        category_mapping = {"restrictive_covenants" : ["non_compete", "confidentiality"],
                            "termination_rights"    : ["termination"],
                            "penalties_liability"   : ["indemnification", "liability"],
                            "compensation_benefits" : ["compensation"],
                            "intellectual_property" : ["intellectual_property"],
                            "confidentiality"       : ["confidentiality"],
                            "liability_indemnity"   : ["indemnification", "liability"],
                            "governing_law"         : ["dispute_resolution"],
                            "payment_terms"         : ["compensation"],
                            "warranties"            : ["warranty"],
                            "dispute_resolution"    : ["dispute_resolution"],
                            "assignment_change"     : ["assignment", "amendment"],
                            "insurance"             : ["insurance"],
                            "force_majeure"         : ["force_majeure"],
                           }
        
        target_categories = category_mapping.get(risk_category, [])
        
        return [c for c in clauses if c.category in target_categories]
    

    def _analyze_clause_risk(self, clause, risk_category: str, contract_type: ContractType) -> Dict:
        """
        Analyze individual clause risk using RiskRules factors
        """
        risk_factors = self.rules.CLAUSE_RISK_FACTORS
        
        # Map clause category to risk factors
        factor_mapping = {"non_compete"           : "restrictive_covenants",
                          "termination"           : "termination_rights", 
                          "indemnification"       : "liability_indemnity",
                          "compensation"          : "compensation_benefits",
                          "intellectual_property" : "intellectual_property",
                          "confidentiality"       : "confidentiality",
                          "liability"             : "penalties_liability",
                          "warranty"              : "warranties",
                          "dispute_resolution"    : "dispute_resolution",
                          "assignment"            : "assignment_change",
                          "amendment"             : "assignment_change",
                          "insurance"             : "insurance",
                          "force_majeure"         : "force_majeure",
                         }
        
        risk_factor_key = factor_mapping.get(clause.category)
        
        if not risk_factor_key or risk_factor_key not in risk_factors:
            return {"score": 0, "findings": []}
        
        factor_config = risk_factors[risk_factor_key]
        base_risk     = factor_config.get("base_risk", 50)
        text_lower    = clause.text.lower()
        
        risk_score    = base_risk
        findings      = list()
        
        # Check red flags
        for red_flag, adjustment in factor_config["red_flags"].items():
            if red_flag in text_lower:
                risk_score += adjustment
                severity    = "increases" if adjustment > 0 else "decreases"

                findings.append(f"Red flag: '{red_flag}' ({severity} risk by {abs(adjustment)})")
        
        # Apply contract-type specific multiplier
        type_adjustments = self.rules.CONTRACT_TYPE_ADJUSTMENTS.get(contract_type.value, {})
        category_multiplier = type_adjustments.get(risk_category, 1.0)
        
        risk_score *= category_multiplier
        
        return {"score"    : max(0, min(100, risk_score)),
                "findings" : findings}
    

    def _calculate_weighted_score(self, category_scores: Dict[str, int], adjusted_weights: Dict[str, float]) -> int:
        """
        Calculate weighted overall risk score
        """
        total_score  = 0
        total_weight = 0
        
        for category, score in category_scores.items():
            weight        = adjusted_weights.get(category, 1.0)
            total_score  += score * weight
            total_weight += weight
        
        return int(total_score / total_weight) if (total_weight > 0) else 50
    

    def _get_risk_level(self, score: int) -> str:
        """
        Convert numeric score to risk level
        """
        if (score >= self.rules.RISK_THRESHOLDS["critical"]):
            return "CRITICAL"
        
        elif (score >= self.rules.RISK_THRESHOLDS["high"]):
            return "HIGH"
        
        elif (score >= self.rules.RISK_THRESHOLDS["medium"]):
            return "MEDIUM"
        
        elif (score >= self.rules.RISK_THRESHOLDS["low"]):
            return "LOW"
        
        return "VERY LOW"
    

    def _create_risk_breakdown(self, category_scores: Dict[str, int], detailed_findings: Dict[str, List[str]]) -> List[RiskBreakdownItem]:
        """
        Create detailed risk breakdown for reporting
        """
        breakdown             = list()
        
        category_descriptions = self.rules.CATEGORY_DESCRIPTIONS
        
        for category, score in category_scores.items():
            if category in category_descriptions:
                # Get appropriate description based on score
                if (score >= 70):
                    risk_level = "high"
                
                elif (score >= 40):
                    risk_level = "medium"
                
                else:
                    risk_level = "low"
                
                summary = category_descriptions[category][risk_level]
            
            else:
                summary = f"Risk assessment for {category.replace('_', ' ')}"
            
            findings = detailed_findings.get(category, [])
            
            breakdown.append(RiskBreakdownItem(category = category.replace('_', ' ').title(),
                                               score    = score,
                                               summary  = summary,
                                               findings = findings[:25],  # Top 25 findings
                                              )
                            )
        
        # Sort by score (highest risk first)
        breakdown.sort(key = lambda x: x.score, reverse = True)
        
        return breakdown
    

    def _compare_to_benchmarks(self, category_scores: Dict[str, int], contract_type: ContractType) -> Dict[str, str]:
        """
        Compare risk scores to industry benchmarks
        """
        comparisons   = dict()
        
        # Overall risk comparison
        overall_score = sum(category_scores.values()) / len(category_scores) if category_scores else 50
        
        if (overall_score >= 70):
            comparisons["overall"] = "✗ Significantly above market risk levels"
        
        elif (overall_score >= 55):
            comparisons["overall"] = "⚠ Above typical market risk levels"
        
        elif (overall_score >= 45):
            comparisons["overall"] = "✓ Within typical market risk range"
        
        else:
            comparisons["overall"] = "✓ Below market risk levels (favorable)"
        
        # Key category comparisons
        high_risk_categories = [cat for cat, score in category_scores.items() if score >= 60]
        
        if high_risk_categories:
            comparisons["high_risk_areas"] = f"High risk in: {', '.join(high_risk_categories)}"
        
        return comparisons
    

    def _get_contract_type_enum(self, category_str: str) -> ContractType:
        """
        Convert category string to ContractType enum
        """
        mapping = {"employment"  : ContractType.EMPLOYMENT,
                   "consulting"  : ContractType.CONSULTING,
                   "nda"         : ContractType.NDA,
                   "software"    : ContractType.SOFTWARE,
                   "service"     : ContractType.SERVICE,
                   "partnership" : ContractType.PARTNERSHIP,
                   "lease"       : ContractType.LEASE,
                   "purchase"    : ContractType.PURCHASE,
                   "general"     : ContractType.GENERAL,
                  }
        
        return mapping.get(category_str, ContractType.GENERAL)