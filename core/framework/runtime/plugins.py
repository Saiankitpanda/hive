"""
Plugin System - Extensible plugin architecture for agents.

Provides:
- Plugin discovery and loading
- Plugin lifecycle management
- Hook system for extensibility
- Plugin configuration

From ROADMAP: Extensibility features
"""

import importlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class PluginState(Enum):
    """Plugin lifecycle states."""

    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class PluginMetadata:
    """Metadata about a plugin."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    config_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plugin:
    """A loaded plugin."""

    metadata: PluginMetadata
    module: Any
    state: PluginState = PluginState.LOADED
    config: dict[str, Any] = field(default_factory=dict)
    instance: Any = None

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version


class HookManager:
    """
    Manages hooks for plugin extensibility.

    Usage:
        hooks = HookManager()

        # Register hook handler
        @hooks.register("before_execute")
        def my_handler(context):
            print("Before execute!")

        # Trigger hook
        hooks.trigger("before_execute", context=ctx)
    """

    def __init__(self):
        self._hooks: dict[str, list[Callable]] = {}
        self._priorities: dict[str, list[tuple[int, Callable]]] = {}

    def register(
        self,
        hook_name: str,
        priority: int = 100,
    ) -> Callable:
        """Decorator to register a hook handler."""

        def decorator(func: Callable) -> Callable:
            if hook_name not in self._priorities:
                self._priorities[hook_name] = []

            self._priorities[hook_name].append((priority, func))
            self._priorities[hook_name].sort(key=lambda x: x[0])

            # Update hooks list
            self._hooks[hook_name] = [f for _, f in self._priorities[hook_name]]

            return func

        return decorator

    def add_handler(
        self,
        hook_name: str,
        handler: Callable,
        priority: int = 100,
    ) -> None:
        """Add a hook handler programmatically."""
        if hook_name not in self._priorities:
            self._priorities[hook_name] = []

        self._priorities[hook_name].append((priority, handler))
        self._priorities[hook_name].sort(key=lambda x: x[0])
        self._hooks[hook_name] = [f for _, f in self._priorities[hook_name]]

    def remove_handler(self, hook_name: str, handler: Callable) -> bool:
        """Remove a hook handler."""
        if hook_name not in self._priorities:
            return False

        self._priorities[hook_name] = [
            (p, f) for p, f in self._priorities[hook_name] if f != handler
        ]
        self._hooks[hook_name] = [f for _, f in self._priorities[hook_name]]
        return True

    def trigger(
        self,
        hook_name: str,
        *args,
        stop_on_false: bool = False,
        **kwargs,
    ) -> list[Any]:
        """Trigger a hook and collect results."""
        results = []

        for handler in self._hooks.get(hook_name, []):
            try:
                result = handler(*args, **kwargs)
                results.append(result)

                if stop_on_false and result is False:
                    break
            except Exception as e:
                results.append(e)

        return results

    async def trigger_async(
        self,
        hook_name: str,
        *args,
        **kwargs,
    ) -> list[Any]:
        """Trigger a hook asynchronously."""
        import asyncio

        results = []

        for handler in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(*args, **kwargs)
                else:
                    result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(e)

        return results

    def get_hooks(self) -> list[str]:
        """Get all registered hook names."""
        return list(self._hooks.keys())

    def get_handlers(self, hook_name: str) -> list[Callable]:
        """Get handlers for a hook."""
        return self._hooks.get(hook_name, [])


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.

    Usage:
        plugins = PluginManager()

        # Discover plugins
        plugins.discover("./plugins")

        # Load a plugin
        plugins.load("my_plugin")

        # Initialize all
        plugins.initialize_all()

        # Use hooks
        plugins.hooks.trigger("on_agent_start", agent=agent)
    """

    def __init__(self):
        self._plugins: dict[str, Plugin] = {}
        self._plugin_paths: list[Path] = []
        self.hooks = HookManager()

        # Built-in hooks
        self._builtin_hooks = [
            "before_agent_start",
            "after_agent_start",
            "before_execute",
            "after_execute",
            "before_tool_call",
            "after_tool_call",
            "on_error",
            "on_llm_request",
            "on_llm_response",
        ]

    def discover(self, path: str | Path) -> list[str]:
        """
        Discover plugins in a directory.

        Plugins should have a plugin.py with a PluginInfo class.
        """
        path = Path(path)
        if not path.exists():
            return []

        discovered = []
        self._plugin_paths.append(path)

        for item in path.iterdir():
            if item.is_dir() and (item / "plugin.py").exists():
                plugin_name = item.name
                discovered.append(plugin_name)

                # Create placeholder
                self._plugins[plugin_name] = Plugin(
                    metadata=PluginMetadata(
                        name=plugin_name,
                        version="0.0.0",
                        description=f"Plugin: {plugin_name}",
                    ),
                    module=None,
                    state=PluginState.DISCOVERED,
                )

        return discovered

    def load(self, name: str) -> Plugin | None:
        """Load a plugin by name."""
        if name not in self._plugins:
            return None

        plugin = self._plugins[name]

        if plugin.state not in (PluginState.DISCOVERED, PluginState.ERROR):
            return plugin

        # Find plugin path
        plugin_path = None
        for base_path in self._plugin_paths:
            candidate = base_path / name / "plugin.py"
            if candidate.exists():
                plugin_path = candidate
                break

        if not plugin_path:
            plugin.state = PluginState.ERROR
            return None

        try:
            # Load module
            spec = importlib.util.spec_from_file_location(
                f"plugins.{name}",
                plugin_path,
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Get metadata
                if hasattr(module, "PLUGIN_INFO"):
                    info = module.PLUGIN_INFO
                    plugin.metadata = PluginMetadata(
                        name=info.get("name", name),
                        version=info.get("version", "1.0.0"),
                        description=info.get("description", ""),
                        author=info.get("author", ""),
                        dependencies=info.get("dependencies", []),
                        hooks=info.get("hooks", []),
                    )

                plugin.module = module
                plugin.state = PluginState.LOADED

                return plugin

        except Exception as e:
            plugin.state = PluginState.ERROR
            plugin.config["error"] = str(e)

        return None

    def initialize(self, name: str) -> bool:
        """Initialize a loaded plugin."""
        plugin = self._plugins.get(name)
        if not plugin or plugin.state != PluginState.LOADED:
            return False

        try:
            # Call plugin's init function
            if hasattr(plugin.module, "init"):
                plugin.instance = plugin.module.init(self.hooks, plugin.config)

            # Register hooks
            if hasattr(plugin.module, "register_hooks"):
                plugin.module.register_hooks(self.hooks)

            plugin.state = PluginState.INITIALIZED
            return True

        except Exception as e:
            plugin.state = PluginState.ERROR
            plugin.config["error"] = str(e)
            return False

    def activate(self, name: str) -> bool:
        """Activate an initialized plugin."""
        plugin = self._plugins.get(name)
        if not plugin or plugin.state != PluginState.INITIALIZED:
            return False

        try:
            if hasattr(plugin.module, "activate"):
                plugin.module.activate(plugin.instance)

            plugin.state = PluginState.ACTIVE
            return True

        except Exception as e:
            plugin.state = PluginState.ERROR
            plugin.config["error"] = str(e)
            return False

    def deactivate(self, name: str) -> bool:
        """Deactivate an active plugin."""
        plugin = self._plugins.get(name)
        if not plugin or plugin.state != PluginState.ACTIVE:
            return False

        try:
            if hasattr(plugin.module, "deactivate"):
                plugin.module.deactivate(plugin.instance)

            plugin.state = PluginState.DISABLED
            return True

        except Exception:
            return False

    def initialize_all(self) -> dict[str, bool]:
        """Initialize all loaded plugins."""
        results = {}
        for name, plugin in self._plugins.items():
            if plugin.state == PluginState.LOADED:
                results[name] = self.initialize(name)
        return results

    def activate_all(self) -> dict[str, bool]:
        """Activate all initialized plugins."""
        results = {}
        for name, plugin in self._plugins.items():
            if plugin.state == PluginState.INITIALIZED:
                results[name] = self.activate(name)
        return results

    def get_plugin(self, name: str) -> Plugin | None:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> list[str]:
        """List all plugin names."""
        return list(self._plugins.keys())

    def get_active_plugins(self) -> list[Plugin]:
        """Get all active plugins."""
        return [p for p in self._plugins.values() if p.state == PluginState.ACTIVE]

    def configure(self, name: str, config: dict[str, Any]) -> bool:
        """Configure a plugin."""
        plugin = self._plugins.get(name)
        if not plugin:
            return False

        plugin.config.update(config)

        if hasattr(plugin.module, "configure"):
            plugin.module.configure(plugin.instance, config)

        return True


def create_plugin_template() -> str:
    """Generate a plugin template."""
    return '''"""
Example Plugin for Hive Agent Framework.

Create a directory with this file as plugin.py to make a plugin.
"""

PLUGIN_INFO = {
    "name": "my_plugin",
    "version": "1.0.0",
    "description": "My custom plugin",
    "author": "Your Name",
    "dependencies": [],
    "hooks": ["before_execute", "after_execute"],
}


class MyPlugin:
    """Plugin implementation."""

    def __init__(self, config: dict):
        self.config = config
        print(f"MyPlugin initialized with config: {config}")

    def before_execute(self, context):
        """Called before agent execution."""
        print(f"Before execute: {context}")

    def after_execute(self, context, result):
        """Called after agent execution."""
        print(f"After execute: {result}")


def init(hooks, config):
    """Initialize the plugin."""
    return MyPlugin(config)


def register_hooks(hooks):
    """Register plugin hooks."""
    pass  # Hooks are registered via PLUGIN_INFO


def activate(instance):
    """Called when plugin is activated."""
    print("Plugin activated!")


def deactivate(instance):
    """Called when plugin is deactivated."""
    print("Plugin deactivated!")


def configure(instance, config):
    """Update plugin configuration."""
    instance.config.update(config)
'''
