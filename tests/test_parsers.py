from __future__ import annotations

from pathlib import Path

import pytest

from glyph.chunkers._parsers import get_parser
from glyph.domain.models import ChunkType

FIXTURES = Path(__file__).parent / "fixtures"


def _parse_fixture(language: str, filename: str, include_bodies: bool = False):
    parser = get_parser(language)
    assert parser is not None, f"No parser for {language}"
    source = (FIXTURES / filename).read_text()
    return parser.parse(source, include_bodies=include_bodies)


def _find(symbols, name, parent=None):
    for s in symbols:
        if s.name == name and s.parent == parent:
            return s
    names = [(s.name, s.parent) for s in symbols]
    raise AssertionError(f"Symbol {name!r} (parent={parent!r}) not found in {names}")


# ── TypeScript ─────────────────────────────────────────────────────────


class TestTypeScriptParser:
    @pytest.fixture()
    def symbols(self):
        return _parse_fixture("typescript", "sample.ts")

    @pytest.fixture()
    def symbols_with_bodies(self):
        return _parse_fixture("typescript", "sample.ts", include_bodies=True)

    def test_class_extracted(self, symbols):
        cls = _find(symbols, "MyClass")
        assert cls.chunk_type == ChunkType.CLASS_OVERVIEW
        assert "BaseClass" in cls.metadata.get("bases", [])
        assert "Serializable" in cls.metadata.get("implements", [])
        assert cls.metadata.get("is_exported") is True
        assert "A sample class with docs" in cls.summary

    def test_constructor(self, symbols):
        ctor = _find(symbols, "constructor", parent="MyClass")
        assert ctor.chunk_type == ChunkType.METHOD
        assert "Constructor docs" in ctor.summary

    def test_methods(self, symbols):
        get_name = _find(symbols, "getName", parent="MyClass")
        assert get_name.metadata.get("visibility") == "public"
        assert get_name.metadata.get("return_type") == "string"

        create = _find(symbols, "create", parent="MyClass")
        assert create.metadata.get("is_static") is True

        fetch = _find(symbols, "fetchData", parent="MyClass")
        assert fetch.metadata.get("is_async") is True

    def test_standalone_function(self, symbols):
        greet = _find(symbols, "greet")
        assert greet.parent is None
        assert greet.metadata.get("is_exported") is True
        assert greet.metadata.get("return_type") == "string"

    def test_arrow_function(self, symbols):
        add = _find(symbols, "add")
        assert add.metadata.get("is_exported") is True
        assert add.metadata.get("return_type") == "number"

    def test_internal_arrow_not_exported(self, symbols):
        internal = _find(symbols, "internal")
        assert internal.metadata.get("is_exported") is None or internal.metadata.get("is_exported") is False

    def test_interface(self, symbols):
        iface = _find(symbols, "MyInterface")
        assert iface.chunk_type == ChunkType.CLASS_OVERVIEW
        assert iface.metadata.get("is_exported") is True
        assert "MyInterface docs" in iface.summary

    def test_enum(self, symbols):
        status = _find(symbols, "Status")
        assert status.chunk_type == ChunkType.ENUM
        assert status.metadata.get("is_exported") is True

    def test_type_alias(self, symbols):
        result = _find(symbols, "Result")
        assert result.chunk_type == ChunkType.CONSTANT
        assert result.metadata.get("is_exported") is True
        assert result.metadata.get("type_parameters")

    def test_jsdoc_tags(self, symbols):
        doc_fn = _find(symbols, "documented")
        assert doc_fn.metadata.get("is_deprecated") is True
        assert doc_fn.metadata.get("jsdoc_returns")

    def test_abstract_class(self, symbols):
        base = _find(symbols, "AbstractBase")
        assert base.chunk_type == ChunkType.CLASS_OVERVIEW
        assert base.metadata.get("is_abstract") is True

        do_work = _find(symbols, "doWork", parent="AbstractBase")
        assert do_work.metadata.get("is_abstract") is True

        helper = _find(symbols, "helper", parent="AbstractBase")
        assert helper.metadata.get("visibility") == "protected"

    def test_include_bodies(self, symbols_with_bodies):
        greet = _find(symbols_with_bodies, "greet")
        assert "return `Hello" in greet.content

    def test_no_bodies_shows_signature(self, symbols):
        greet = _find(symbols, "greet")
        assert "```typescript" in greet.content
        assert "function greet" in greet.content


# ── Rust ───────────────────────────────────────────────────────────────


class TestRustParser:
    @pytest.fixture()
    def symbols(self):
        return _parse_fixture("rust", "sample.rs")

    @pytest.fixture()
    def symbols_with_bodies(self):
        return _parse_fixture("rust", "sample.rs", include_bodies=True)

    def test_struct(self, symbols):
        s = _find(symbols, "MyStruct")
        assert s.chunk_type == ChunkType.CLASS_OVERVIEW
        assert s.metadata.get("visibility") == "pub"
        assert "Debug" in s.metadata.get("derives", [])
        assert "Clone" in s.metadata.get("derives", [])
        assert "A sample struct" in s.summary

    def test_impl_methods(self, symbols):
        new = _find(symbols, "new", parent="MyStruct")
        assert new.chunk_type == ChunkType.METHOD
        assert new.metadata.get("visibility") == "pub"
        assert new.metadata.get("return_type") == "Self"

        name = _find(symbols, "name", parent="MyStruct")
        assert "Gets the name" in name.summary

    def test_async_method(self, symbols):
        fetch = _find(symbols, "fetch", parent="MyStruct")
        assert fetch.metadata.get("is_async") is True

    def test_unsafe_method(self, symbols):
        raw = _find(symbols, "raw_ptr", parent="MyStruct")
        assert raw.metadata.get("is_unsafe") is True

    def test_trait(self, symbols):
        trait = _find(symbols, "Drawable")
        assert trait.chunk_type == ChunkType.CLASS_OVERVIEW
        assert "A trait definition" in trait.summary

    def test_trait_methods(self, symbols):
        draw = _find(symbols, "draw", parent="Drawable")
        assert draw.chunk_type == ChunkType.METHOD

        bounds = _find(symbols, "bounds", parent="Drawable")
        assert bounds.metadata.get("return_type") == "Rect"

    def test_trait_impl(self, symbols):
        draw = _find(symbols, "draw", parent="MyStruct")
        assert draw.metadata.get("trait_impl") == "Drawable"

    def test_standalone_function(self, symbols):
        greet = _find(symbols, "greet")
        assert greet.parent is None
        assert greet.metadata.get("visibility") == "pub"

    def test_generic_function(self, symbols):
        process = _find(symbols, "process")
        assert process.metadata.get("generics")

    def test_enum(self, symbols):
        color = _find(symbols, "Color")
        assert color.chunk_type == ChunkType.ENUM
        assert "Red" in color.metadata.get("variants", [])
        assert "Custom" in color.metadata.get("variants", [])

    def test_constant(self, symbols):
        max_size = _find(symbols, "MAX_SIZE")
        assert max_size.chunk_type == ChunkType.CONSTANT

    def test_static(self, symbols):
        counter = _find(symbols, "COUNTER")
        assert counter.chunk_type == ChunkType.CONSTANT

    def test_type_alias(self, symbols):
        result = _find(symbols, "Result")
        assert result.chunk_type == ChunkType.CONSTANT

    def test_visibility_variants(self, symbols):
        crate_vis = _find(symbols, "crate_visible")
        assert crate_vis.metadata.get("visibility") == "pub(crate)"

        private = _find(symbols, "private_fn")
        assert not private.metadata.get("visibility")

    def test_include_bodies(self, symbols_with_bodies):
        greet = _find(symbols_with_bodies, "greet")
        assert 'format!("Hello' in greet.content


# ── Go ─────────────────────────────────────────────────────────────────


class TestGoParser:
    @pytest.fixture()
    def symbols(self):
        return _parse_fixture("go", "sample.go")

    @pytest.fixture()
    def symbols_with_bodies(self):
        return _parse_fixture("go", "sample.go", include_bodies=True)

    def test_struct(self, symbols):
        s = _find(symbols, "MyStruct")
        assert s.chunk_type == ChunkType.CLASS_OVERVIEW
        assert s.metadata.get("is_exported") is True
        assert "MyStruct is a sample struct." in s.summary

    def test_standalone_function(self, symbols):
        new = _find(symbols, "NewMyStruct")
        assert new.parent is None
        assert new.metadata.get("is_exported") is True

        greet = _find(symbols, "greet")
        assert greet.metadata.get("is_exported") is False

    def test_method_with_pointer_receiver(self, symbols):
        get_name = _find(symbols, "GetName", parent="MyStruct")
        assert get_name.metadata.get("is_pointer_receiver") is True
        assert get_name.metadata.get("receiver_type") == "MyStruct"
        assert get_name.metadata.get("is_exported") is True
        assert get_name.metadata.get("return_type") == "string"

    def test_method_with_value_receiver(self, symbols):
        set_age = _find(symbols, "setAge", parent="MyStruct")
        assert set_age.metadata.get("is_pointer_receiver") is False
        assert set_age.metadata.get("is_exported") is False

    def test_interface(self, symbols):
        stringer = _find(symbols, "Stringer")
        assert stringer.chunk_type == ChunkType.CLASS_OVERVIEW
        assert stringer.metadata.get("is_exported") is True

    def test_multiple_returns(self, symbols):
        multi = _find(symbols, "multiReturn")
        assert multi.metadata.get("return_type") is not None
        assert "int" in multi.metadata["return_type"]
        assert "error" in multi.metadata["return_type"]

    def test_const_single(self, symbols):
        max_size = _find(symbols, "MaxSize")
        assert max_size.chunk_type == ChunkType.CONSTANT
        assert max_size.metadata.get("is_exported") is True

    def test_const_block(self, symbols):
        active = _find(symbols, "StatusActive")
        assert active.chunk_type == ChunkType.CONSTANT

        inactive = _find(symbols, "StatusInactive")
        assert inactive.chunk_type == ChunkType.CONSTANT

    def test_var(self, symbols):
        default_name = _find(symbols, "DefaultName")
        assert default_name.chunk_type == ChunkType.PROPERTY
        assert default_name.metadata.get("is_exported") is True

    def test_var_block(self, symbols):
        counter = _find(symbols, "counter")
        assert counter.chunk_type == ChunkType.PROPERTY
        assert counter.metadata.get("is_exported") is False

    def test_type_alias(self, symbols):
        sl = _find(symbols, "StringList")
        assert sl.chunk_type == ChunkType.CONSTANT

    def test_named_type(self, symbols):
        uid = _find(symbols, "UserID")
        assert uid.chunk_type == ChunkType.CONSTANT

    def test_include_bodies(self, symbols_with_bodies):
        new = _find(symbols_with_bodies, "NewMyStruct")
        assert "return &MyStruct" in new.content

    def test_doc_comments(self, symbols):
        s = _find(symbols, "MyStruct")
        assert "sample struct" in s.summary
