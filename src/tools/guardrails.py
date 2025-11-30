"""
Guardrails for LLM Call Validation and Safety.

Provides input/output validation, content filtering, and domain-specific
guardrails for LLM API calls to ensure safe and reliable operation.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Guardrails AI imports with fallback
try:
    from guardrails import Guard
    from guardrails.hub import (
        DetectPII,
        DetectSecrets,
        ToxicLanguage,
        BanSubstrings,
        BanTopics,
        ReadingTime,
        PolitenessCheck
    )
    GUARDRAILS_AI_AVAILABLE = True
except ImportError:
    GUARDRAILS_AI_AVAILABLE = False
    logger.warning("Guardrails AI is not installed. Install with: pip install guardrails-ai>=0.4.0")


class GuardrailResult(Enum):
    """Result status for guardrail checks."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class GuardrailCheck:
    """Result of a guardrail check."""
    name: str
    status: GuardrailResult
    message: str
    details: Optional[Dict[str, Any]] = None


class LLMGuardrails:
    """
    Comprehensive guardrails for LLM calls.
    
    Provides:
    - Input validation and sanitization
    - Prompt injection detection
    - Output validation
    - Content filtering
    - Domain-specific validation (financial)
    """
    
    def __init__(self, enable_guardrails_ai: bool = True):
        """
        Initialize LLM Guardrails.
        
        Args:
            enable_guardrails_ai: Whether to use Guardrails AI library if available
        """
        self.enable_guardrails_ai = enable_guardrails_ai and GUARDRAILS_AI_AVAILABLE
        self.guard = None
        self._initialize_guardrails()
        
        # Prompt injection patterns
        self.injection_patterns = [
            r"(?i)ignore\s+(previous|all|your)\s+(instructions?|prompts?)",
            r"(?i)forget\s+(previous|all|your)\s+(instructions?|prompts?)",
            r"(?i)you\s+are\s+now\s+(a|an)\s+",
            r"(?i)system\s*:\s*",
            r"(?i)assistant\s*:\s*",
            r"(?i)user\s*:\s*",
            r"(?i)new\s+instructions?\s*:",
            r"(?i)override\s+(previous|instructions?)",
            r"(?i)disregard\s+(previous|all|your)\s+",
            r"(?i)pretend\s+you\s+are",
            r"(?i)act\s+as\s+if",
            r"(?i)you\s+must\s+now",
            r"(?i)your\s+new\s+(role|task|instructions?)",
        ]
        
        # Harmful content keywords (financial domain specific)
        self.harmful_keywords = [
            "financial fraud",
            "market manipulation",
            "insider trading",
            "pump and dump",
            "ponzi scheme",
        ]
        
        # Financial domain validation patterns
        self.financial_patterns = {
            "stock_symbol": r"^[A-Z]{1,5}$",
            "price": r"^\d+\.?\d*$",
            "percentage": r"^-?\d+\.?\d*%?$",
        }
    
    def _initialize_guardrails(self):
        """Initialize Guardrails AI if available."""
        if not self.enable_guardrails_ai:
            logger.info("Guardrails AI disabled or not available. Using custom guardrails only.")
            return
        
        try:
            # Create a Guard instance with basic validations
            # Note: Guardrails AI configuration can be customized based on needs
            self.guard = Guard()
            logger.info("Guardrails AI initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Guardrails AI: {e}. Using custom guardrails only.")
            self.enable_guardrails_ai = False
    
    def validate_input(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[GuardrailCheck]]:
        """
        Validate LLM input for safety and compliance.
        
        Args:
            messages: List of message dictionaries
            context: Optional context for validation
        
        Returns:
            Tuple of (is_valid, list_of_checks)
        """
        checks = []
        is_valid = True
        
        # Extract user content from messages
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content += msg.get("content", "") + " "
        
        user_content = user_content.strip()
        
        # Check 1: Prompt injection detection
        injection_check = self._check_prompt_injection(user_content)
        checks.append(injection_check)
        if injection_check.status == GuardrailResult.FAIL:
            is_valid = False
        
        # Check 2: Input length validation
        length_check = self._check_input_length(user_content)
        checks.append(length_check)
        if length_check.status == GuardrailResult.FAIL:
            is_valid = False
        
        # Check 3: Harmful content detection
        harmful_check = self._check_harmful_content(user_content)
        checks.append(harmful_check)
        if harmful_check.status == GuardrailResult.FAIL:
            is_valid = False
        
        # Check 4: Guardrails AI validation (if available)
        if self.enable_guardrails_ai and self.guard:
            try:
                ai_check = self._check_with_guardrails_ai(user_content, "input")
                if ai_check:
                    checks.append(ai_check)
                    if ai_check.status == GuardrailResult.FAIL:
                        is_valid = False
            except Exception as e:
                logger.debug(f"Guardrails AI input check failed: {e}")
        
        return is_valid, checks
    
    def validate_output(
        self,
        output: str,
        expected_format: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[GuardrailCheck]]:
        """
        Validate LLM output for quality and compliance.
        
        Args:
            output: LLM output text
            expected_format: Expected format (e.g., "json", "markdown")
            context: Optional context for validation
        
        Returns:
            Tuple of (is_valid, list_of_checks)
        """
        checks = []
        is_valid = True
        
        # Check 1: Output not empty
        if not output or not output.strip():
            checks.append(GuardrailCheck(
                name="output_empty",
                status=GuardrailResult.FAIL,
                message="LLM output is empty"
            ))
            return False, checks
        
        # Check 2: Output length validation
        length_check = self._check_output_length(output)
        checks.append(length_check)
        if length_check.status == GuardrailResult.FAIL:
            is_valid = False
        
        # Check 3: Format validation
        if expected_format:
            format_check = self._check_output_format(output, expected_format)
            checks.append(format_check)
            if format_check.status == GuardrailResult.FAIL:
                is_valid = False
        
        # Check 4: Harmful content in output
        harmful_check = self._check_harmful_content(output)
        checks.append(harmful_check)
        if harmful_check.status == GuardrailResult.FAIL:
            is_valid = False
        
        # Check 5: Guardrails AI validation (if available)
        if self.enable_guardrails_ai and self.guard:
            try:
                ai_check = self._check_with_guardrails_ai(output, "output")
                if ai_check:
                    checks.append(ai_check)
                    if ai_check.status == GuardrailResult.FAIL:
                        is_valid = False
            except Exception as e:
                logger.debug(f"Guardrails AI output check failed: {e}")
        
        return is_valid, checks
    
    def _check_prompt_injection(self, content: str) -> GuardrailCheck:
        """Check for prompt injection attempts."""
        content_lower = content.lower()
        
        for pattern in self.injection_patterns:
            if re.search(pattern, content):
                return GuardrailCheck(
                    name="prompt_injection",
                    status=GuardrailResult.FAIL,
                    message=f"Potential prompt injection detected: {pattern}",
                    details={"pattern": pattern, "content_snippet": content[:100]}
                )
        
        return GuardrailCheck(
            name="prompt_injection",
            status=GuardrailResult.PASS,
            message="No prompt injection detected"
        )
    
    def _check_input_length(self, content: str) -> GuardrailCheck:
        """Check input length constraints."""
        max_length = 50000  # Maximum input length
        min_length = 1
        
        if len(content) > max_length:
            return GuardrailCheck(
                name="input_length",
                status=GuardrailResult.FAIL,
                message=f"Input too long: {len(content)} characters (max: {max_length})",
                details={"length": len(content), "max_length": max_length}
            )
        
        if len(content) < min_length:
            return GuardrailCheck(
                name="input_length",
                status=GuardrailResult.FAIL,
                message=f"Input too short: {len(content)} characters (min: {min_length})",
                details={"length": len(content), "min_length": min_length}
            )
        
        return GuardrailCheck(
            name="input_length",
            status=GuardrailResult.PASS,
            message=f"Input length valid: {len(content)} characters"
        )
    
    def _check_output_length(self, content: str) -> GuardrailCheck:
        """Check output length constraints."""
        max_length = 100000  # Maximum output length
        min_length = 1
        
        if len(content) > max_length:
            return GuardrailCheck(
                name="output_length",
                status=GuardrailResult.WARNING,
                message=f"Output very long: {len(content)} characters",
                details={"length": len(content), "max_length": max_length}
            )
        
        if len(content) < min_length:
            return GuardrailCheck(
                name="output_length",
                status=GuardrailResult.FAIL,
                message="Output is empty",
                details={"length": len(content)}
            )
        
        return GuardrailCheck(
            name="output_length",
            status=GuardrailResult.PASS,
            message=f"Output length valid: {len(content)} characters"
        )
    
    def _check_harmful_content(self, content: str) -> GuardrailCheck:
        """Check for harmful or inappropriate content."""
        content_lower = content.lower()
        
        for keyword in self.harmful_keywords:
            if keyword.lower() in content_lower:
                return GuardrailCheck(
                    name="harmful_content",
                    status=GuardrailResult.FAIL,
                    message=f"Potentially harmful content detected: {keyword}",
                    details={"keyword": keyword}
                )
        
        return GuardrailCheck(
            name="harmful_content",
            status=GuardrailResult.PASS,
            message="No harmful content detected"
        )
    
    def _check_output_format(self, content: str, expected_format: str) -> GuardrailCheck:
        """Check if output matches expected format."""
        if expected_format.lower() == "json":
            try:
                json.loads(content)
                return GuardrailCheck(
                    name="output_format",
                    status=GuardrailResult.PASS,
                    message="Output is valid JSON"
                )
            except json.JSONDecodeError as e:
                return GuardrailCheck(
                    name="output_format",
                    status=GuardrailResult.FAIL,
                    message=f"Output is not valid JSON: {str(e)}",
                    details={"expected": "json", "error": str(e)}
                )
        
        elif expected_format.lower() == "markdown":
            # Basic markdown validation (check for markdown patterns)
            markdown_patterns = [r"^#+\s", r"\*\*.*\*\*", r"`.*`", r"\[.*\]\(.*\)"]
            has_markdown = any(re.search(pattern, content, re.MULTILINE) for pattern in markdown_patterns)
            
            if has_markdown or len(content) > 100:  # Allow plain text for short outputs
                return GuardrailCheck(
                    name="output_format",
                    status=GuardrailResult.PASS,
                    message="Output appears to be valid markdown or text"
                )
            else:
                return GuardrailCheck(
                    name="output_format",
                    status=GuardrailResult.WARNING,
                    message="Output may not be valid markdown",
                    details={"expected": "markdown"}
                )
        
        return GuardrailCheck(
            name="output_format",
            status=GuardrailResult.PASS,
            message=f"Format check passed for {expected_format}"
        )
    
    def _check_with_guardrails_ai(self, content: str, check_type: str) -> Optional[GuardrailCheck]:
        """Use Guardrails AI for additional validation."""
        if not self.enable_guardrails_ai or not self.guard:
            return None
        
        try:
            # This is a placeholder - actual Guardrails AI integration would depend on
            # specific validators you want to use
            # Example: self.guard.validate(content, validators=[...])
            return None  # Implement based on specific Guardrails AI validators needed
        except Exception as e:
            logger.debug(f"Guardrails AI check failed: {e}")
            return None
    
    def sanitize_input(self, content: str) -> str:
        """
        Sanitize input content to remove potentially dangerous patterns.
        
        Args:
            content: Input content to sanitize
        
        Returns:
            Sanitized content
        """
        sanitized = content
        
        # Remove common injection patterns (aggressive sanitization)
        for pattern in self.injection_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Trim
        sanitized = sanitized.strip()
        
        return sanitized
    
    def validate_financial_content(
        self,
        content: str,
        content_type: str = "general"
    ) -> Tuple[bool, List[GuardrailCheck]]:
        """
        Validate financial domain-specific content.
        
        Args:
            content: Content to validate
            content_type: Type of financial content (e.g., "stock_symbol", "price", "report")
        
        Returns:
            Tuple of (is_valid, list_of_checks)
        """
        checks = []
        is_valid = True
        
        # Check for financial data patterns
        if content_type == "stock_symbol":
            if not re.match(self.financial_patterns["stock_symbol"], content):
                checks.append(GuardrailCheck(
                    name="financial_validation",
                    status=GuardrailResult.FAIL,
                    message=f"Invalid stock symbol format: {content}",
                    details={"content_type": content_type}
                ))
                is_valid = False
        
        # Check for financial disclaimers in reports
        if content_type == "report":
            required_disclaimers = ["disclaimer", "risk", "investment"]
            has_disclaimer = any(
                disclaimer.lower() in content.lower() 
                for disclaimer in required_disclaimers
            )
            
            if not has_disclaimer and len(content) > 1000:
                checks.append(GuardrailCheck(
                    name="financial_validation",
                    status=GuardrailResult.WARNING,
                    message="Financial report may be missing required disclaimers",
                    details={"content_type": content_type}
                ))
        
        if is_valid and not checks:
            checks.append(GuardrailCheck(
                name="financial_validation",
                status=GuardrailResult.PASS,
                message=f"Financial content validation passed for {content_type}"
            ))
        
        return is_valid, checks


# Global guardrails instance
_guardrails: Optional[LLMGuardrails] = None


def initialize_guardrails(enable_guardrails_ai: bool = True) -> LLMGuardrails:
    """
    Initialize global guardrails instance.
    
    Args:
        enable_guardrails_ai: Whether to use Guardrails AI library
    
    Returns:
        LLMGuardrails instance
    """
    global _guardrails
    _guardrails = LLMGuardrails(enable_guardrails_ai=enable_guardrails_ai)
    logger.info("Guardrails initialized")
    return _guardrails


def get_guardrails() -> Optional[LLMGuardrails]:
    """
    Get the global guardrails instance.
    
    Returns:
        LLMGuardrails instance or None if not initialized
    """
    return _guardrails


