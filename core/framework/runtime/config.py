"""
Configuration Manager - Centralized configuration for agents.

Provides:
- Environment-based configuration
- Config validation with defaults
- Multiple config sources (env, files, code)
- Type-safe configuration access

From ROADMAP: Configuration management system
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    api_key: str = ""
    base_url: str | None = None
    timeout_seconds: int = 60

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load from environment variables."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL"),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT", "60")),
        )


@dataclass
class RuntimeConfig:
    """Agent runtime configuration."""

    max_iterations: int = 100
    timeout_seconds: int = 300
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_tracing: bool = False

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Load from environment variables."""
        return cls(
            max_iterations=int(os.getenv("RUNTIME_MAX_ITERATIONS", "100")),
            timeout_seconds=int(os.getenv("RUNTIME_TIMEOUT", "300")),
            enable_logging=os.getenv("RUNTIME_LOGGING", "true").lower() == "true",
            log_level=os.getenv("RUNTIME_LOG_LEVEL", "INFO"),
            enable_metrics=os.getenv("RUNTIME_METRICS", "true").lower() == "true",
            enable_tracing=os.getenv("RUNTIME_TRACING", "false").lower() == "true",
        )


@dataclass
class GuardrailsConfig:
    """Guardrails configuration."""

    enabled: bool = True
    max_tokens: int = 100000
    max_cost_usd: float = 10.0
    enable_pii_detection: bool = True
    enable_content_filter: bool = True
    max_response_length: int = 10000

    @classmethod
    def from_env(cls) -> "GuardrailsConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true",
            max_tokens=int(os.getenv("GUARDRAILS_MAX_TOKENS", "100000")),
            max_cost_usd=float(os.getenv("GUARDRAILS_MAX_COST", "10.0")),
            enable_pii_detection=os.getenv("GUARDRAILS_PII", "true").lower() == "true",
            enable_content_filter=os.getenv("GUARDRAILS_FILTER", "true").lower() == "true",
            max_response_length=int(os.getenv("GUARDRAILS_MAX_LENGTH", "10000")),
        )


@dataclass
class DashboardConfig:
    """Dashboard configuration."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    enable_websocket: bool = True
    auto_open_browser: bool = False

    @classmethod
    def from_env(cls) -> "DashboardConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("DASHBOARD_ENABLED", "true").lower() == "true",
            host=os.getenv("DASHBOARD_HOST", "127.0.0.1"),
            port=int(os.getenv("DASHBOARD_PORT", "8000")),
            enable_websocket=os.getenv("DASHBOARD_WS", "true").lower() == "true",
            auto_open_browser=os.getenv("DASHBOARD_AUTO_OPEN", "false").lower() == "true",
        )


@dataclass
class CacheConfig:
    """Caching configuration."""

    enabled: bool = True
    ttl_hours: int = 24
    max_entries: int = 1000
    storage_path: str | None = None
    enable_semantic_cache: bool = False

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Load from environment variables."""
        return cls(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            ttl_hours=int(os.getenv("CACHE_TTL_HOURS", "24")),
            max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "1000")),
            storage_path=os.getenv("CACHE_STORAGE_PATH"),
            enable_semantic_cache=os.getenv("CACHE_SEMANTIC", "false").lower() == "true",
        )


@dataclass
class AgentConfig:
    """Complete agent configuration."""

    name: str = "agent"
    version: str = "1.0.0"
    description: str = ""

    llm: LLMConfig = field(default_factory=LLMConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    custom: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load all config from environment."""
        return cls(
            name=os.getenv("AGENT_NAME", "agent"),
            version=os.getenv("AGENT_VERSION", "1.0.0"),
            description=os.getenv("AGENT_DESCRIPTION", ""),
            llm=LLMConfig.from_env(),
            runtime=RuntimeConfig.from_env(),
            guardrails=GuardrailsConfig.from_env(),
            dashboard=DashboardConfig.from_env(),
            cache=CacheConfig.from_env(),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Create config from dictionary."""
        return cls(
            name=data.get("name", "agent"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            llm=LLMConfig(**data.get("llm", {})) if "llm" in data else LLMConfig(),
            runtime=RuntimeConfig(**data.get("runtime", {}))
            if "runtime" in data
            else RuntimeConfig(),
            guardrails=GuardrailsConfig(**data.get("guardrails", {}))
            if "guardrails" in data
            else GuardrailsConfig(),
            dashboard=DashboardConfig(**data.get("dashboard", {}))
            if "dashboard" in data
            else DashboardConfig(),
            cache=CacheConfig(**data.get("cache", {})) if "cache" in data else CacheConfig(),
            custom=data.get("custom", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
                "timeout_seconds": self.llm.timeout_seconds,
            },
            "runtime": {
                "max_iterations": self.runtime.max_iterations,
                "timeout_seconds": self.runtime.timeout_seconds,
                "enable_logging": self.runtime.enable_logging,
                "log_level": self.runtime.log_level,
            },
            "guardrails": {
                "enabled": self.guardrails.enabled,
                "max_tokens": self.guardrails.max_tokens,
                "max_cost_usd": self.guardrails.max_cost_usd,
            },
            "dashboard": {
                "enabled": self.dashboard.enabled,
                "host": self.dashboard.host,
                "port": self.dashboard.port,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "ttl_hours": self.cache.ttl_hours,
                "max_entries": self.cache.max_entries,
            },
            "custom": self.custom,
        }

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ConfigManager:
    """
    Manager for agent configuration.

    Usage:
        # Load from environment
        config = ConfigManager.load_from_env()

        # Load from file
        config = ConfigManager.load_from_file("config.json")

        # Access config
        print(config.llm.model)
        print(config.runtime.max_iterations)

        # Override specific values
        ConfigManager.set("llm.model", "claude-3-opus")
    """

    _instance: "ConfigManager | None" = None
    _config: AgentConfig

    def __init__(self, config: AgentConfig | None = None):
        self._config = config or AgentConfig()

    @classmethod
    def instance(cls) -> "ConfigManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(AgentConfig.from_env())
        return cls._instance

    @classmethod
    def load_from_env(cls) -> AgentConfig:
        """Load config from environment."""
        config = AgentConfig.from_env()
        cls._instance = cls(config)
        return config

    @classmethod
    def load_from_file(cls, path: str | Path) -> AgentConfig:
        """Load config from file."""
        config = AgentConfig.from_file(path)
        cls._instance = cls(config)
        return config

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a config value by dot-notation key."""
        instance = cls.instance()
        parts = key.split(".")

        obj = instance._config
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default

        return obj

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a config value by dot-notation key."""
        instance = cls.instance()
        parts = key.split(".")

        obj = instance._config
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return

        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)

    @property
    def config(self) -> AgentConfig:
        """Get the config object."""
        return self._config


# Convenience function
def get_config() -> AgentConfig:
    """Get the current agent configuration."""
    return ConfigManager.instance().config


def load_config(path: str | Path | None = None) -> AgentConfig:
    """
    Load configuration from file or environment.

    Args:
        path: Optional path to config file

    Returns:
        Loaded AgentConfig
    """
    if path:
        return ConfigManager.load_from_file(path)
    return ConfigManager.load_from_env()
