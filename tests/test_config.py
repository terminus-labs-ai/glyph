"""Tests for the config module — global, per-repo, and auto-discovery."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from glyph.config import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXCLUDE_PATTERNS,
    DefaultsConfig,
    discover_source,
    load_config,
    load_global_config,
    load_repo_config,
    resolve_config_for_repo,
)


# --- Fixtures ---


@pytest.fixture
def full_config_yaml(tmp_path):
    """Write a full glyph.yaml and return its path."""
    cfg = {
        "database": {"url": "postgresql://test:test@localhost:5432/glyph_test"},
        "embedder": {
            "type": "llama",
            "url": "http://localhost:11434",
            "model": "test-model",
            "dimensions": 256,
            "batch_size": 2,
        },
        "sources": [
            {
                "name": "my-lib",
                "version": "1.0",
                "ingestors": [
                    {"type": "source_code", "path": "/src", "extensions": [".py"]},
                ],
            },
        ],
        "output": {"directory": "/tmp/output", "formats": ["markdown"]},
        "defaults": {
            "include_bodies": True,
            "exclude_dirs": [".git", "node_modules"],
            "exclude_patterns": ["_test."],
        },
    }
    config_file = tmp_path / "glyph.yaml"
    config_file.write_text(yaml.dump(cfg))
    return config_file


@pytest.fixture
def global_config_dir(tmp_path):
    """Create a global config directory with config.yaml."""
    config_dir = tmp_path / ".config" / "glyph"
    config_dir.mkdir(parents=True)
    cfg = {
        "database": {"url": "postgresql://global:global@localhost:5432/glyph"},
        "embedder": {"model": "global-model", "dimensions": 512},
        "output": {"directory": "/global/output"},
    }
    (config_dir / "config.yaml").write_text(yaml.dump(cfg))
    return config_dir / "config.yaml"


@pytest.fixture
def repo_with_glyph_yaml(tmp_path):
    """Create a fake repo with .glyph.yaml."""
    repo = tmp_path / "my-project"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("print('hello')")

    glyph_cfg = {
        "name": "my-project",
        "version": "2.0",
        "ingestors": [
            {
                "type": "source_code",
                "path": "./src",
                "extensions": [".py"],
                "include_bodies": False,
            },
        ],
    }
    (repo / ".glyph.yaml").write_text(yaml.dump(glyph_cfg))
    return repo


@pytest.fixture
def repo_without_config(tmp_path):
    """Create a fake repo with source files but no .glyph.yaml."""
    repo = tmp_path / "auto-detect"
    repo.mkdir()
    (repo / "main.py").write_text("print('hello')")
    (repo / "utils.py").write_text("def foo(): pass")
    (repo / "lib").mkdir()
    (repo / "lib" / "helper.ts").write_text("export function bar() {}")
    # Docs dir
    (repo / "docs").mkdir()
    (repo / "docs" / "guide.md").write_text("# Guide")
    return repo


# --- load_config (backwards compat) ---


class TestLoadConfig:
    def test_loads_full_config(self, full_config_yaml):
        cfg = load_config(full_config_yaml)
        assert cfg.database.url == "postgresql://test:test@localhost:5432/glyph_test"
        assert cfg.embedder.model == "test-model"
        assert cfg.embedder.dimensions == 256
        assert len(cfg.sources) == 1
        assert cfg.sources[0].name == "my-lib"
        assert cfg.output.directory == "/tmp/output"

    def test_loads_defaults_section(self, full_config_yaml):
        cfg = load_config(full_config_yaml)
        assert cfg.defaults.include_bodies is True
        assert cfg.defaults.exclude_dirs == [".git", "node_modules"]
        assert cfg.defaults.exclude_patterns == ["_test."]

    def test_defaults_have_sane_fallbacks(self, tmp_path):
        """Config without defaults section uses built-in defaults."""
        cfg_file = tmp_path / "minimal.yaml"
        cfg_file.write_text(yaml.dump({
            "database": {"url": "postgresql://x"},
        }))
        cfg = load_config(cfg_file)
        assert cfg.defaults.include_bodies is False
        assert ".git" in cfg.defaults.exclude_dirs
        assert cfg.embedder.model == "nomic-embed-text"


# --- load_global_config ---


class TestLoadGlobalConfig:
    def test_loads_from_explicit_path(self, global_config_dir):
        cfg = load_global_config(global_config_dir)
        assert cfg is not None
        assert cfg.database.url == "postgresql://global:global@localhost:5432/glyph"
        assert cfg.embedder.model == "global-model"

    def test_returns_none_for_missing_path(self, tmp_path):
        cfg = load_global_config(tmp_path / "nonexistent.yaml")
        assert cfg is None

    def test_returns_none_when_no_well_known_paths(self):
        with patch("glyph.config.GLOBAL_CONFIG_PATHS", [Path("/nonexistent/path/config.yaml")]):
            cfg = load_global_config()
            assert cfg is None

    def test_finds_well_known_path(self, global_config_dir):
        with patch("glyph.config.GLOBAL_CONFIG_PATHS", [global_config_dir]):
            cfg = load_global_config()
            assert cfg is not None
            assert cfg.embedder.model == "global-model"


# --- load_repo_config ---


class TestLoadRepoConfig:
    def test_loads_glyph_yaml(self, repo_with_glyph_yaml):
        src_cfg = load_repo_config(repo_with_glyph_yaml)
        assert src_cfg is not None
        assert src_cfg.name == "my-project"
        assert src_cfg.version == "2.0"
        assert len(src_cfg.ingestors) == 1
        assert src_cfg.ingestors[0].type == "source_code"

    def test_resolves_relative_paths(self, repo_with_glyph_yaml):
        src_cfg = load_repo_config(repo_with_glyph_yaml)
        # ./src should be resolved to absolute
        ingestor_path = src_cfg.ingestors[0].settings["path"]
        assert Path(ingestor_path).is_absolute()
        assert ingestor_path.endswith("src")

    def test_returns_none_without_config(self, repo_without_config):
        src_cfg = load_repo_config(repo_without_config)
        assert src_cfg is None

    def test_version_auto_fallback(self, tmp_path):
        """version: auto without git falls back to 'latest'."""
        repo = tmp_path / "no-git"
        repo.mkdir()
        (repo / ".glyph.yaml").write_text(yaml.dump({
            "name": "test",
            "version": "auto",
            "ingestors": [{"type": "source_code", "path": "."}],
        }))
        src_cfg = load_repo_config(repo)
        assert src_cfg.version == "latest"

    def test_defaults_name_from_dirname(self, tmp_path):
        """Missing name defaults to directory name."""
        repo = tmp_path / "cool-project"
        repo.mkdir()
        (repo / ".glyph.yaml").write_text(yaml.dump({
            "version": "1.0",
            "ingestors": [{"type": "source_code", "path": "."}],
        }))
        src_cfg = load_repo_config(repo)
        assert src_cfg.name == "cool-project"


# --- discover_source ---


class TestDiscoverSource:
    def test_discovers_python_and_typescript(self, repo_without_config):
        src_cfg = discover_source(repo_without_config)
        assert src_cfg.name == "auto-detect"
        # Should find source_code ingestor with .py and .ts
        code_ingestor = next(i for i in src_cfg.ingestors if i.type == "source_code")
        extensions = code_ingestor.settings["extensions"]
        assert ".py" in extensions
        assert ".ts" in extensions

    def test_discovers_docs_directory(self, repo_without_config):
        src_cfg = discover_source(repo_without_config)
        docs_ingestor = next((i for i in src_cfg.ingestors if i.type == "docs"), None)
        assert docs_ingestor is not None
        assert ".md" in docs_ingestor.settings["extensions"]

    def test_skips_excluded_dirs(self, tmp_path):
        repo = tmp_path / "with-excluded"
        repo.mkdir()
        # Only python files are in node_modules — should be skipped
        nm = repo / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.ts").write_text("export default {}")
        # One real file
        (repo / "app.py").write_text("print()")

        src_cfg = discover_source(repo)
        code_ingestor = next(i for i in src_cfg.ingestors if i.type == "source_code")
        extensions = code_ingestor.settings["extensions"]
        assert ".py" in extensions
        assert ".ts" not in extensions  # only in node_modules

    def test_fallback_on_empty_repo(self, tmp_path):
        repo = tmp_path / "empty"
        repo.mkdir()
        src_cfg = discover_source(repo)
        # Should still return a config with defaults
        assert src_cfg.name == "empty"
        assert len(src_cfg.ingestors) >= 1

    def test_uses_custom_defaults(self, tmp_path):
        repo = tmp_path / "custom"
        repo.mkdir()
        (repo / "main.py").write_text("x = 1")

        defaults = DefaultsConfig(
            include_bodies=True,
            exclude_dirs=[".git"],
            exclude_patterns=["_test."],
        )
        src_cfg = discover_source(repo, defaults=defaults)
        code_ingestor = next(i for i in src_cfg.ingestors if i.type == "source_code")
        assert code_ingestor.settings["include_bodies"] is True
        assert code_ingestor.settings["exclude_dirs"] == [".git"]

    def test_version_from_git_branch(self, tmp_path):
        """In a git repo with no tags, version comes from branch name."""
        repo = tmp_path / "git-repo"
        repo.mkdir()
        (repo / "main.py").write_text("x = 1")
        subprocess.run(["git", "init"], cwd=repo, capture_output=True)
        subprocess.run(["git", "checkout", "-b", "develop"], cwd=repo, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init", "--allow-empty"],
            cwd=repo, capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@test.com",
                 "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@test.com"},
        )

        src_cfg = discover_source(repo)
        assert src_cfg.version == "develop"


# --- resolve_config_for_repo ---


class TestResolveConfigForRepo:
    def test_with_repo_config(self, repo_with_glyph_yaml, full_config_yaml):
        global_cfg = load_config(full_config_yaml)
        config, source_cfg = resolve_config_for_repo(
            repo_with_glyph_yaml, global_config=global_cfg,
        )
        assert source_cfg.name == "my-project"
        assert source_cfg.version == "2.0"
        assert config.database.url == global_cfg.database.url

    def test_auto_discovers_without_repo_config(self, repo_without_config, full_config_yaml):
        global_cfg = load_config(full_config_yaml)
        config, source_cfg = resolve_config_for_repo(
            repo_without_config, global_config=global_cfg,
        )
        assert source_cfg.name == "auto-detect"
        assert len(source_cfg.ingestors) >= 1

    def test_name_override(self, repo_with_glyph_yaml, full_config_yaml):
        global_cfg = load_config(full_config_yaml)
        _, source_cfg = resolve_config_for_repo(
            repo_with_glyph_yaml, global_config=global_cfg,
            name_override="custom-name",
        )
        assert source_cfg.name == "custom-name"

    def test_version_override(self, repo_with_glyph_yaml, full_config_yaml):
        global_cfg = load_config(full_config_yaml)
        _, source_cfg = resolve_config_for_repo(
            repo_with_glyph_yaml, global_config=global_cfg,
            version_override="99.0",
        )
        assert source_cfg.version == "99.0"

    def test_raises_without_global_config(self, repo_without_config):
        with patch("glyph.config.load_global_config", return_value=None):
            with pytest.raises(ValueError, match="No global config found"):
                resolve_config_for_repo(repo_without_config)
