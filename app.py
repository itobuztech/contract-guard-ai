"""
FastAPI Application for AI Contract Risk Analyzer - UPDATED
Complete integration with new services pipeline and frontend requirements
"""
import signal
import os
import time
import json
import uuid
from typing import Any, List, Dict, Optional
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import all services - UPDATED WITH NEW SERVICES
from config.settings import settings
from config.risk_rules import ContractType
from model_manager.model_loader import ModelLoader
from model_manager.llm_manager import LLMManager, LLMProvider
from utils.document_reader import DocumentReader
from utils.validators import ContractValidator
from utils.text_processor import TextProcessor
from utils.logger import ContractAnalyzerLogger, log_info, log_error

# UPDATED SERVICE IMPORTS
from services.contract_classifier import ContractClassifier, ContractCategory
from services.clause_extractor import ComprehensiveClauseExtractor, RiskClauseExtractor, ExtractedClause
from services.risk_analyzer import RiskAnalyzer, RiskScore
from services.term_analyzer import TermAnalyzer, UnfavorableTerm
from services.protection_checker import ProtectionChecker, MissingProtection
from services.llm_interpreter import LLMClauseInterpreter, ClauseInterpretation, RiskInterpretation
from services.negotiation_engine import NegotiationEngine, NegotiationPlaybook, NegotiationPoint
from services.summary_generator import SummaryGenerator

# Import PDF generator
from reporter.pdf_generator import generate_pdf_report

# ============================================================================
# CUSTOM SERIALIZATION (UNCHANGED)
# ============================================================================

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int8, np.uint8)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        return super().default(obj)

class NumpyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=NumpyJSONEncoder,
        ).encode("utf-8")

def convert_numpy_types(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int8, np.uint8)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'to_dict'):
        return convert_numpy_types(obj.to_dict())
    elif hasattr(obj, 'dict'):
        return convert_numpy_types(obj.dict())
    else:
        return obj

def safe_serialize_response(data: Any) -> Any:
    return convert_numpy_types(data)

# ============================================================================
# PYDANTIC SCHEMAS - UPDATED FOR FRONTEND COMPATIBILITY
# ============================================================================

class SerializableBaseModel(BaseModel):
    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        data = super().dict(*args, **kwargs)
        return convert_numpy_types(data)
    
    def json(self, *args, **kwargs) -> str:
        data = self.dict(*args, **kwargs)
        return json.dumps(data, cls=NumpyJSONEncoder, *args, **kwargs)

class HealthResponse(SerializableBaseModel):
    status: str
    version: str
    timestamp: str
    models_loaded: int
    services_loaded: int
    memory_usage_mb: float

class AnalysisOptions(SerializableBaseModel):
    max_clauses: int = Field(default=15, ge=5, le=30)
    interpret_clauses: bool = Field(default=True)
    generate_negotiation_points: bool = Field(default=True)
    compare_to_market: bool = Field(default=False)  # Disabled for now

class AnalysisResult(SerializableBaseModel):
    analysis_id: str
    timestamp: str
    classification: Dict[str, Any]
    clauses: List[Dict[str, Any]]
    risk_analysis: Dict[str, Any]
    unfavorable_terms: List[Dict[str, Any]]
    missing_protections: List[Dict[str, Any]]
    clause_interpretations: Optional[List[Dict[str, Any]]] = None
    negotiation_points: Optional[List[Dict[str, Any]]] = None
    market_comparisons: Optional[List[Dict[str, Any]]] = None
    executive_summary: str
    metadata: Dict[str, Any]
    pdf_available: bool = True

class ErrorResponse(SerializableBaseModel):
    error: str
    detail: str
    timestamp: str

class FileValidationResponse(SerializableBaseModel):
    valid: bool
    message: str
    confidence: Optional[float] = None
    report: Optional[Dict[str, Any]] = None

# ============================================================================
# SERVICE INITIALIZATION WITH FULL PIPELINE INTEGRATION
# ============================================================================

class PreloadedAnalysisService:
    """Analysis service with complete pipeline integration"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.llm_manager = LLMManager()
        self.services = {}
        self.service_status = {}
        self.memory_usage_mb = 0
        self._preload_all_services()
    
    def _preload_all_services(self):
        """Pre-load ALL services and models at initialization"""
        log_info("PRE-LOADING ALL AI MODELS AND SERVICES")
        
        try:
            initial_memory = self._get_memory_usage()
            
            # 1. Pre-load Contract Classifier
            log_info("🔄 Pre-loading Contract Classifier...")
            self.services["classifier"] = ContractClassifier(self.model_loader)
            self.service_status["classifier"] = "loaded"
            log_info("✅ Contract Classifier loaded")
            
            # 2. Pre-load Comprehensive Clause Extractor
            log_info("🔄 Pre-loading Comprehensive Clause Extractor...")
            self.services["clause_extractor"] = ComprehensiveClauseExtractor(self.model_loader)
            self.service_status["clause_extractor"] = "loaded"
            log_info("✅ Comprehensive Clause Extractor loaded")
            
            # 3. Pre-load Risk Analyzer (Main Orchestrator)
            log_info("🔄 Pre-loading Risk Analyzer...")
            self.services["risk_analyzer"] = RiskAnalyzer(self.model_loader)
            self.service_status["risk_analyzer"] = "loaded"
            log_info("✅ Risk Analyzer loaded")
            
            # 4. Pre-load LLM Interpreter
            log_info("🔄 Pre-loading LLM Interpreter...")
            try:
                self.services["llm_interpreter"] = LLMClauseInterpreter(self.llm_manager)
                self.service_status["llm_interpreter"] = "loaded"
                log_info("✅ LLM Interpreter loaded")
            except Exception as e:
                self.services["llm_interpreter"] = None
                self.service_status["llm_interpreter"] = f"failed: {str(e)}"
                log_info("⚠️  LLM Interpreter not available")
            
            # 5. Pre-load Negotiation Engine
            log_info("🔄 Pre-loading Negotiation Engine...")
            try:
                self.services["negotiation_engine"] = NegotiationEngine(self.llm_manager)
                self.service_status["negotiation_engine"] = "loaded"
                log_info("✅ Negotiation Engine loaded")
            except Exception as e:
                self.services["negotiation_engine"] = None
                self.service_status["negotiation_engine"] = f"failed: {str(e)}"
                log_info("⚠️  Negotiation Engine not available")
            
            # 6. Pre-load Summary Generator
            log_info("🔄 Pre-loading Summary Generator...")
            try:
                self.services["summary_generator"] = SummaryGenerator(self.llm_manager)
                self.service_status["summary_generator"] = "loaded"
                log_info("✅ Summary Generator loaded")
            except Exception as e:
                self.services["summary_generator"] = SummaryGenerator()
                self.service_status["summary_generator"] = "fallback_loaded"
                log_info("⚠️  Summary Generator using fallback mode")
            
            # Calculate memory usage
            final_memory = self._get_memory_usage()
            self.memory_usage_mb = final_memory - initial_memory
            
            log_info("🎉 ALL SERVICES PRE-LOADED SUCCESSFULLY!")
            log_info(f"📊 Memory Usage: {self.memory_usage_mb:.2f} MB")
            log_info(f"🔧 Services Loaded: {len(self.service_status)}")
            
        except Exception as e:
            log_error(f"CRITICAL: Failed to pre-load services: {e}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get detailed service status"""
        model_stats = self.model_loader.get_registry_stats()
        return {
            "services": self.service_status,
            "models": model_stats,
            "memory_usage_mb": self.memory_usage_mb,
            "total_services_loaded": len([s for s in self.service_status.values() if "loaded" in str(s)]),
            "total_models_loaded": model_stats.get("loaded_models", 0)
        }
    
    def analyze_contract(self, contract_text: str, options: AnalysisOptions) -> Dict[str, Any]:
        """Complete contract analysis using full pipeline"""
        try:
            log_info("Starting comprehensive contract analysis pipeline...")
            
            # Step 1: Classify contract
            classification = self.services["classifier"].classify_contract(contract_text)
            classification_dict = safe_serialize_response(classification.to_dict())
            log_info(f"Contract classified as: {classification.category}")
            
            # Step 2: Extract clauses
            clauses = self.services["clause_extractor"].extract_clauses(
                contract_text, options.max_clauses
            )
            clauses_dict = [safe_serialize_response(clause.to_dict()) for clause in clauses]
            log_info(f"Extracted {len(clauses)} clauses")
            
            # Step 3: Map to ContractType
            contract_type = self._get_contract_type_enum(classification.category)
            
            # Step 4: Complete Risk Analysis (Main Orchestrator)
            risk_score = self.services["risk_analyzer"].analyze_contract_risk(contract_text)
            risk_dict = safe_serialize_response(risk_score.to_dict())
            log_info(f"Risk analysis completed: {risk_score.overall_score}/100")
            
            # Extract components from risk analysis for further processing
            unfavorable_terms = risk_score.unfavorable_terms
            missing_protections = risk_score.missing_protections
            
            # Step 5: Generate LLM Interpretations (if enabled and available)
            interpretations_dict = None
            risk_interpretation = None
            
            if options.interpret_clauses and self.services["llm_interpreter"]:
                try:
                    risk_interpretation = self.services["llm_interpreter"].interpret_with_risk_context(
                        clauses=clauses,
                        unfavorable_terms=unfavorable_terms,
                        missing_protections=missing_protections,
                        contract_type=contract_type,
                        overall_risk_score=risk_score.overall_score,
                        max_clauses=min(10, options.max_clauses)
                    )
                    interpretations_dict = [
                        safe_serialize_response(interp.to_dict()) 
                        for interp in risk_interpretation.clause_interpretations
                    ]
                    log_info(f"Generated {len(interpretations_dict)} clause interpretations")
                except Exception as e:
                    log_error(f"LLM interpretation failed: {e}")
                    interpretations_dict = []
            
            # Step 6: Generate Negotiation Points (if enabled and available)
            negotiation_dict     = []
            negotiation_playbook = None

            if options.generate_negotiation_points and self.services["negotiation_engine"]:
                try:
                    negotiation_playbook = self.services["negotiation_engine"].generate_comprehensive_playbook(
                        risk_analysis=risk_score,
                        risk_interpretation=risk_interpretation or RiskInterpretation(
                            overall_risk_explanation="",
                            key_concerns=[],
                            negotiation_strategy="",
                            market_comparison="",
                            clause_interpretations=[]
                        ),
                        unfavorable_terms=unfavorable_terms,
                        missing_protections=missing_protections,
                        clauses=clauses,
                        contract_type=contract_type,
                        max_points=8  # Match frontend limit
                    )
                    
                    negotiation_dict = [
                        safe_serialize_response(point.to_dict()) 
                        for point in negotiation_playbook.critical_points
                    ]
                    log_info(f"Generated {len(negotiation_dict)} negotiation points")

                except Exception as e:
                    log_error(f"Negotiation engine failed: {e}")
                    print(f"🔍 DEBUG: Negotiation engine exception: {e}")
                    import traceback
                    print(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
                    negotiation_dict = []
            
            # Step 7: Generate Executive Summary
            executive_summary = self.services["summary_generator"].generate_comprehensive_summary(
                contract_text=contract_text,
                classification=classification,
                risk_analysis=risk_score,
                risk_interpretation=risk_interpretation or RiskInterpretation(
                    overall_risk_explanation="",
                    key_concerns=[],
                    negotiation_strategy="",
                    market_comparison="",
                    clause_interpretations=[]
                ),
                negotiation_playbook=negotiation_playbook or NegotiationPlaybook(
                    overall_strategy="",
                    critical_points=[],
                    walk_away_items=[],
                    concession_items=[],
                    timing_guidance="",
                    risk_mitigation_plan=""
                ),
                unfavorable_terms=unfavorable_terms,
                missing_protections=missing_protections,
                clauses=clauses
            )
            
            # Build final result matching frontend expectations
            result = {
                "analysis_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "classification": classification_dict,
                "clauses": clauses_dict,
                "risk_analysis": risk_dict,  # Contains overall_score, risk_level, category_scores, risk_breakdown
                "unfavorable_terms": [safe_serialize_response(term) for term in unfavorable_terms],
                "missing_protections": [safe_serialize_response(prot) for prot in missing_protections],
                "clause_interpretations": interpretations_dict,
                "negotiation_points": negotiation_dict,
                "market_comparisons": [],  # Disabled for now
                "executive_summary": executive_summary,
                "metadata": {
                    "text_length": len(contract_text),
                    "word_count": len(contract_text.split()),
                    "num_clauses": len(clauses),
                    "contract_type": contract_type.value,
                    "actual_category": classification.category,
                    "options": options.dict()
                },
                "pdf_available": True
            }
            
            log_info("Contract analysis completed successfully")
            return result
            
        except Exception as e:
            log_error(f"Contract analysis failed: {e}")
            raise
    
    def _get_contract_type_enum(self, category_str: str) -> ContractType:
        """Convert category string to ContractType enum"""
        mapping = {
            'employment': ContractType.EMPLOYMENT,
            'consulting': ContractType.CONSULTING,
            'nda': ContractType.NDA,
            'software': ContractType.SOFTWARE,
            'service': ContractType.SERVICE,
            'partnership': ContractType.PARTNERSHIP,
            'lease': ContractType.LEASE,
            'purchase': ContractType.PURCHASE,
            'general': ContractType.GENERAL,
        }
        return mapping.get(category_str, ContractType.GENERAL)

# ============================================================================
# FASTAPI APPLICATION (UNCHANGED STRUCTURE, UPDATED IMPLEMENTATION)
# ============================================================================

# Global instances
analysis_service: Optional[PreloadedAnalysisService] = None
app_start_time = time.time()

# Initialize logger
ContractAnalyzerLogger.setup(log_dir="logs", app_name="contract_analyzer")
logger = ContractAnalyzerLogger.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global analysis_service
    
    log_info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} STARTING UP...")
    log_info("=" * 80)
    
    try:
        analysis_service = PreloadedAnalysisService()
        log_info("✅ All services initialized successfully")
    except Exception as e:
        log_error(f"Startup failed: {e}")
        raise
    
    log_info(f"📍 Server: {settings.HOST}:{settings.PORT}")
    log_info("=" * 80)
    log_info("✅ AI Contract Risk Analyzer Ready!")
    
    try:
        yield
    finally:
        log_info("🛑 Shutting down server...")
        log_info("✅ Server shutdown complete")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered contract risk analysis with complete model pre-loading",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    default_response_class=NumpyJSONResponse,
    lifespan=lifespan
)

# Get absolute paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HELPER FUNCTIONS (UNCHANGED)
# ============================================================================

def validate_file(file: UploadFile) -> tuple[bool, str]:
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
    
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    
    if size > settings.MAX_UPLOAD_SIZE:
        return False, f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / (1024*1024)}MB"
    
    if size == 0:
        return False, "File is empty"
    
    return True, "OK"


def read_contract_file(file) -> str:
    """
    Read contract file and return text content.
    """
    reader         = DocumentReader()
    
    # Extract file extension without dot
    filename       = file.filename.lower()
    file_extension = Path(filename).suffix.lower().lstrip('.')
    
    # If no extension found, try to detect from content or default to pdf
    if not file_extension:
        file_extension = "pdf"
        print(f"📁 DEBUG app.py - No extension found, defaulting to: '{file_extension}'")
    
    file_contents = reader.read_file(file.file, file_extension)
    
    if (not file_contents or not file_contents.strip()):
        raise ValueError("Could not extract text from file")
        
    return file_contents



def validate_contract_text(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "Contract text is empty"
    
    if len(text) < settings.MIN_CONTRACT_LENGTH:
        return False, f"Contract text too short. Minimum {settings.MIN_CONTRACT_LENGTH} characters required."
    
    if len(text) > settings.MAX_CONTRACT_LENGTH:
        return False, f"Contract text too long. Maximum {settings.MAX_CONTRACT_LENGTH} characters allowed."
    
    return True, "OK"

# ============================================================================
# API ROUTES (UNCHANGED INTERFACE, UPDATED IMPLEMENTATION)
# ============================================================================

@app.get("/")
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    if not analysis_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    service_status = analysis_service.get_service_status()
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.now().isoformat(),
        models_loaded=service_status["total_models_loaded"],
        services_loaded=service_status["total_services_loaded"],
        memory_usage_mb=service_status["memory_usage_mb"]
    )

@app.get("/api/v1/status")
async def get_detailed_status():
    if not analysis_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return analysis_service.get_service_status()

@app.post("/api/v1/analyze/file", response_model=AnalysisResult)
async def analyze_contract_file(
    file: UploadFile = File(...),
    max_clauses: int = Form(15),
    interpret_clauses: bool = Form(True),
    generate_negotiation_points: bool = Form(True),
    compare_to_market: bool = Form(False)  # Disabled for now
):
    if not analysis_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Validate file
        is_valid, message = validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Read contract text
        contract_text = read_contract_file(file)
        
        # Validate contract text
        is_valid_text, text_message = validate_contract_text(contract_text)
        if not is_valid_text:
            raise HTTPException(status_code=400, detail=text_message)
        
        # Validate contract structure using ContractValidator
        validator = ContractValidator()
        is_valid_contract, contract_type, confidence = validator.is_valid_contract(contract_text)
        
        if not is_valid_contract:
            raise HTTPException(status_code=400, detail=f"Invalid contract: {confidence}")
        
        # Create analysis options
        options = AnalysisOptions(
            max_clauses=min(max_clauses, settings.MAX_CLAUSES_TO_ANALYZE),
            interpret_clauses=interpret_clauses,
            generate_negotiation_points=generate_negotiation_points,
            compare_to_market=compare_to_market
        )
        
        # Perform analysis
        result = analysis_service.analyze_contract(contract_text, options)
        
        log_info(f"File analysis completed", 
                filename=file.filename,
                analysis_id=result["analysis_id"],
                risk_score=result["risk_analysis"]["overall_score"])
        
        return AnalysisResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"File analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/analyze/text", response_model=AnalysisResult)
async def analyze_contract_text(
    contract_text: str = Form(..., description="Contract text to analyze"),
    max_clauses: int = Form(15),
    interpret_clauses: bool = Form(True),
    generate_negotiation_points: bool = Form(True),
    compare_to_market: bool = Form(False)  # Disabled for now
):
    if not analysis_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Validate contract text length first
        is_valid, message = validate_contract_text(contract_text)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Validate contract structure using ContractValidator
        validator = ContractValidator()
        is_valid_contract, validation_type, message = validator.is_valid_contract(contract_text)
        
        if not is_valid_contract:
            error_message = message if "does not appear to be a legal contract" in message else "The provided document does not appear to be a legal contract. Please upload a valid contract for analysis."
            raise HTTPException(status_code=400, detail=error_message)
                
        # Create analysis options
        options = AnalysisOptions(
            max_clauses=min(max_clauses, settings.MAX_CLAUSES_TO_ANALYZE),
            interpret_clauses=interpret_clauses,
            generate_negotiation_points=generate_negotiation_points,
            compare_to_market=compare_to_market
        )
        
        # Perform analysis
        result = analysis_service.analyze_contract(contract_text, options)
        
        log_info(f"Text analysis completed", 
                analysis_id=result["analysis_id"],
                risk_score=result["risk_analysis"]["overall_score"])
        
        return AnalysisResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Text analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/generate-pdf")
async def generate_pdf_from_analysis(analysis_result: Dict[str, Any]):
    try:
        pdf_buffer = generate_pdf_report(analysis_result)
        
        analysis_id = analysis_result.get('analysis_id', 'report')
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=contract_analysis_{analysis_id}.pdf"
            }
        )
    except Exception as e:
        log_error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@app.get("/api/v1/categories")
async def get_contract_categories():
    if not analysis_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        categories = analysis_service.services["classifier"].get_all_categories()
        return {"categories": categories}
    except Exception as e:
        log_error(f"Categories fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@app.post("/api/v1/validate/file", response_model=FileValidationResponse)
async def validate_contract_file(file: UploadFile = File(...)):
    try:
        is_valid, message = validate_file(file)
        if not is_valid:
            return FileValidationResponse(valid=False, message=message)
        
        contract_text = read_contract_file(file)
        
        # Validate text length
        is_valid_text, text_message = validate_contract_text(contract_text)
        if not is_valid_text:
            return FileValidationResponse(valid=False, message=text_message)
        
        # Validate contract structure using ContractValidator
        validator = ContractValidator()
        report = validator.get_validation_report(contract_text)
        
        return FileValidationResponse(
            valid=report["scores"]["total"] > 50 and is_valid_text,
            message="Contract appears valid" if report["scores"]["total"] > 50 else "May not be a valid contract",
            confidence=report["scores"]["total"],
            report=report
        )
        
    except Exception as e:
        log_error(f"File validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

@app.post("/api/v1/validate/text", response_model=FileValidationResponse)
async def validate_contract_text_endpoint(contract_text: str = Form(...)):
    try:
        # Validate text length
        is_valid, message = validate_contract_text(contract_text)
        if not is_valid:
            return FileValidationResponse(valid=False, message=message)
        
        # Validate contract structure using ContractValidator
        validator = ContractValidator()
        report = validator.get_validation_report(contract_text)
        
        return FileValidationResponse(
            valid=report["scores"]["total"] > 50 and is_valid,
            message="Contract appears valid" if report["scores"]["total"] > 50 else "May not be a valid contract",
            confidence=report["scores"]["total"],
            report=report
        )
        
    except Exception as e:
        log_error(f"Text validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

# ============================================================================
# ERROR HANDLERS AND MIDDLEWARE (UNCHANGED)
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return NumpyJSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    log_error(f"Unhandled exception: {exc}")
    return NumpyJSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    log_info(f"API Request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.3f}s")
    
    return response

# ============================================================================
# MAIN (UNCHANGED)
# ============================================================================
if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n👋 Received Ctrl+C, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        uvicorn.run(
            "app:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.RELOAD,
            workers=1,
            log_level=settings.LOG_LEVEL.lower()
        )
    except KeyboardInterrupt:
        print("\n🎯 Server stopped by user")
    except Exception as e:
        log_error(f"Server error: {e}")
        sys.exit(1)