"""
Validation - Schema validation and data validation utilities.

Provides:
- JSON Schema validation
- Type checking
- Custom validators
- Error collection

From ROADMAP: Data validation features
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class ValidationErrorType(Enum):
    """Types of validation errors."""

    REQUIRED = "required"
    TYPE = "type"
    FORMAT = "format"
    RANGE = "range"
    LENGTH = "length"
    PATTERN = "pattern"
    ENUM = "enum"
    CUSTOM = "custom"


@dataclass
class ValidationError:
    """A validation error."""

    field: str
    message: str
    error_type: ValidationErrorType
    value: Any = None
    constraint: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "type": self.error_type.value,
            "value": self.value,
            "constraint": self.constraint,
        }


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    def add_error(
        self,
        field: str,
        message: str,
        error_type: ValidationErrorType,
        value: Any = None,
        constraint: Any = None,
    ) -> None:
        """Add an error."""
        self.errors.append(
            ValidationError(
                field=field,
                message=message,
                error_type=error_type,
                value=value,
                constraint=constraint,
            )
        )
        self.valid = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
        }

    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed."""
        if not self.valid:
            messages = [f"{e.field}: {e.message}" for e in self.errors]
            raise ValidationException(messages)


class ValidationException(Exception):
    """Exception for validation failures."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


class Validator:
    """
    Validates data against schemas and rules.

    Usage:
        v = Validator()

        # Simple validation
        result = v.validate_type(value, str)

        # Schema validation
        schema = {
            "name": {"type": "string", "required": True, "min_length": 1},
            "age": {"type": "integer", "min": 0, "max": 150},
            "email": {"type": "string", "pattern": "email"},
        }
        result = v.validate(data, schema)

        if not result.valid:
            for error in result.errors:
                print(error.message)
    """

    # Type mappings
    TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    # Format patterns
    FORMAT_PATTERNS = {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "url": r"^https?://[^\s<>\"{}|\\^`\[\]]+$",
        "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        "date": r"^\d{4}-\d{2}-\d{2}$",
        "datetime": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
        "phone": r"^\+?[1-9]\d{1,14}$",
    }

    def __init__(self):
        self._custom_validators: dict[str, Callable] = {}

    def validate(
        self,
        data: dict[str, Any],
        schema: dict[str, dict[str, Any]],
    ) -> ValidationResult:
        """Validate data against schema."""
        result = ValidationResult(valid=True)

        # Check required fields
        for field_name, rules in schema.items():
            if rules.get("required", False) and field_name not in data:
                result.add_error(
                    field=field_name,
                    message=f"Field '{field_name}' is required",
                    error_type=ValidationErrorType.REQUIRED,
                )

        # Validate each field
        for field_name, value in data.items():
            if field_name not in schema:
                continue

            rules = schema[field_name]
            self._validate_field(field_name, value, rules, result)

        return result

    def _validate_field(
        self,
        field_name: str,
        value: Any,
        rules: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate a single field."""
        # Type validation
        if "type" in rules:
            type_name = rules["type"]
            expected_type = self.TYPE_MAP.get(type_name)

            if expected_type and not isinstance(value, expected_type):
                result.add_error(
                    field=field_name,
                    message=f"Expected {type_name}, got {type(value).__name__}",
                    error_type=ValidationErrorType.TYPE,
                    value=value,
                    constraint=type_name,
                )
                return  # Skip further validation if type is wrong

        # String validations
        if isinstance(value, str):
            self._validate_string(field_name, value, rules, result)

        # Number validations
        if isinstance(value, (int, float)):
            self._validate_number(field_name, value, rules, result)

        # Array validations
        if isinstance(value, list):
            self._validate_array(field_name, value, rules, result)

        # Format validation
        if "pattern" in rules:
            self._validate_pattern(field_name, value, rules["pattern"], result)

        # Enum validation
        if "enum" in rules:
            if value not in rules["enum"]:
                result.add_error(
                    field=field_name,
                    message=f"Value must be one of: {rules['enum']}",
                    error_type=ValidationErrorType.ENUM,
                    value=value,
                    constraint=rules["enum"],
                )

        # Custom validator
        if "validator" in rules:
            validator_name = rules["validator"]
            if validator_name in self._custom_validators:
                validator = self._custom_validators[validator_name]
                if not validator(value):
                    result.add_error(
                        field=field_name,
                        message=f"Custom validation '{validator_name}' failed",
                        error_type=ValidationErrorType.CUSTOM,
                        value=value,
                    )

    def _validate_string(
        self,
        field_name: str,
        value: str,
        rules: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate string value."""
        if "min_length" in rules and len(value) < rules["min_length"]:
            result.add_error(
                field=field_name,
                message=f"Length must be at least {rules['min_length']}",
                error_type=ValidationErrorType.LENGTH,
                value=len(value),
                constraint=rules["min_length"],
            )

        if "max_length" in rules and len(value) > rules["max_length"]:
            result.add_error(
                field=field_name,
                message=f"Length must be at most {rules['max_length']}",
                error_type=ValidationErrorType.LENGTH,
                value=len(value),
                constraint=rules["max_length"],
            )

    def _validate_number(
        self,
        field_name: str,
        value: int | float,
        rules: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate number value."""
        if "min" in rules and value < rules["min"]:
            result.add_error(
                field=field_name,
                message=f"Value must be at least {rules['min']}",
                error_type=ValidationErrorType.RANGE,
                value=value,
                constraint=rules["min"],
            )

        if "max" in rules and value > rules["max"]:
            result.add_error(
                field=field_name,
                message=f"Value must be at most {rules['max']}",
                error_type=ValidationErrorType.RANGE,
                value=value,
                constraint=rules["max"],
            )

    def _validate_array(
        self,
        field_name: str,
        value: list,
        rules: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate array value."""
        if "min_items" in rules and len(value) < rules["min_items"]:
            result.add_error(
                field=field_name,
                message=f"Array must have at least {rules['min_items']} items",
                error_type=ValidationErrorType.LENGTH,
                value=len(value),
                constraint=rules["min_items"],
            )

        if "max_items" in rules and len(value) > rules["max_items"]:
            result.add_error(
                field=field_name,
                message=f"Array must have at most {rules['max_items']} items",
                error_type=ValidationErrorType.LENGTH,
                value=len(value),
                constraint=rules["max_items"],
            )

    def _validate_pattern(
        self,
        field_name: str,
        value: Any,
        pattern: str,
        result: ValidationResult,
    ) -> None:
        """Validate against pattern."""
        if not isinstance(value, str):
            return

        # Check if it's a named format
        regex = self.FORMAT_PATTERNS.get(pattern, pattern)

        if not re.match(regex, value):
            result.add_error(
                field=field_name,
                message=f"Value does not match pattern '{pattern}'",
                error_type=ValidationErrorType.PATTERN,
                value=value,
                constraint=pattern,
            )

    def add_custom_validator(
        self,
        name: str,
        validator: Callable[[Any], bool],
    ) -> None:
        """Add a custom validator."""
        self._custom_validators[name] = validator

    def validate_type(self, value: Any, expected_type: type) -> bool:
        """Simple type validation."""
        return isinstance(value, expected_type)


def validate_dict_schema(
    data: dict[str, Any],
    schema: dict[str, dict[str, Any]],
) -> ValidationResult:
    """Convenience function for schema validation."""
    return Validator().validate(data, schema)


def assert_type(value: Any, expected_type: type, name: str = "value") -> None:
    """Assert value is of expected type."""
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__}, got {type(value).__name__}")


def assert_not_none(value: Any, name: str = "value") -> None:
    """Assert value is not None."""
    if value is None:
        raise ValueError(f"{name} cannot be None")


def assert_not_empty(value: Any, name: str = "value") -> None:
    """Assert value is not empty."""
    if not value:
        raise ValueError(f"{name} cannot be empty")
