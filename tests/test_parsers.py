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


# ── Python ─────────────────────────────────────────────────────────────


class TestPythonParser:
    @pytest.fixture()
    def symbols(self):
        return _parse_fixture("python", "sample.py")

    @pytest.fixture()
    def symbols_with_bodies(self):
        return _parse_fixture("python", "sample.py", include_bodies=True)

    def test_class_extracted(self, symbols):
        cls = _find(symbols, "DataProcessor")
        assert cls.chunk_type == ChunkType.CLASS_OVERVIEW
        assert "BaseProcessor" in cls.metadata.get("bases", [])
        assert "Processes data records" in cls.summary

    def test_base_class_no_bases(self, symbols):
        cls = _find(symbols, "BaseProcessor")
        assert cls.chunk_type == ChunkType.CLASS_OVERVIEW
        assert cls.metadata.get("bases", []) == []

    def test_method_with_parent(self, symbols):
        process = _find(symbols, "process", parent="DataProcessor")
        assert process.chunk_type == ChunkType.METHOD
        assert process.metadata.get("return_type") == "list[dict]"

    def test_static_method_decorator(self, symbols):
        validate = _find(symbols, "validate", parent="DataProcessor")
        assert "@staticmethod" in validate.metadata.get("decorators", [])

    def test_classmethod_decorator(self, symbols):
        from_config = _find(symbols, "from_config", parent="DataProcessor")
        assert "@classmethod" in from_config.metadata.get("decorators", [])

    def test_property_decorator(self, symbols):
        display_name = _find(symbols, "display_name", parent="DataProcessor")
        assert "@property" in display_name.metadata.get("decorators", [])
        assert display_name.metadata.get("return_type") == "str"

    def test_standalone_function(self, symbols):
        fn = _find(symbols, "run_pipeline")
        assert fn.parent is None
        assert fn.chunk_type == ChunkType.METHOD
        assert "Run data through" in fn.summary

    def test_include_bodies(self, symbols_with_bodies):
        fn = _find(symbols_with_bodies, "run_pipeline")
        assert "for p in processors" in fn.content

    def test_no_bodies_shows_signature(self, symbols):
        fn = _find(symbols, "run_pipeline")
        assert "```python" in fn.content
        assert "def run_pipeline" in fn.content
        assert "for p in processors" not in fn.content


# ── GDScript ───────────────────────────────────────────────────────────


class TestGDScriptParser:
    @pytest.fixture()
    def symbols(self):
        return _parse_fixture("gdscript", "sample.gd")

    @pytest.fixture()
    def symbols_with_bodies(self):
        return _parse_fixture("gdscript", "sample.gd", include_bodies=True)

    def test_class_overview(self, symbols):
        cls = _find(symbols, "PlayerController")
        assert cls.chunk_type == ChunkType.CLASS_OVERVIEW
        assert cls.metadata.get("extends") == "CharacterBody2D"
        assert "sample GDScript" in cls.summary

    def test_signal_no_params(self, symbols):
        sig = _find(symbols, "died", parent="PlayerController")
        assert sig.chunk_type == ChunkType.SIGNAL

    def test_signal_with_params(self, symbols):
        sig = _find(symbols, "health_changed", parent="PlayerController")
        assert sig.chunk_type == ChunkType.SIGNAL
        assert "health_changed" in sig.content

    def test_enum(self, symbols):
        e = _find(symbols, "State", parent="PlayerController")
        assert e.chunk_type == ChunkType.ENUM
        assert "IDLE" in e.metadata.get("values", [])
        assert "DEAD" in e.metadata.get("values", [])

    def test_const_with_type(self, symbols):
        c = _find(symbols, "MAX_SPEED", parent="PlayerController")
        assert c.chunk_type == ChunkType.CONSTANT
        assert "300.0" in c.content

    def test_const_without_type(self, symbols):
        c = _find(symbols, "JUMP_FORCE", parent="PlayerController")
        assert c.chunk_type == ChunkType.CONSTANT

    def test_exported_var(self, symbols):
        v = _find(symbols, "health", parent="PlayerController")
        assert v.chunk_type == ChunkType.PROPERTY
        assert v.metadata.get("type") == "int"
        assert v.metadata.get("default") == "100"

    def test_var_with_type(self, symbols):
        v = _find(symbols, "speed", parent="PlayerController")
        assert v.chunk_type == ChunkType.PROPERTY
        assert v.metadata.get("type") == "float"

    def test_func_with_return_type(self, symbols):
        fn = _find(symbols, "get_speed", parent="PlayerController")
        assert fn.chunk_type == ChunkType.METHOD
        assert fn.metadata.get("return_type") == "float"

    def test_func_doc_comment(self, symbols):
        fn = _find(symbols, "move", parent="PlayerController")
        assert "Move the player" in fn.summary

    def test_func_with_params_metadata(self, symbols):
        fn = _find(symbols, "take_damage", parent="PlayerController")
        params = fn.metadata.get("params", [])
        assert any(p["name"] == "amount" for p in params)

    def test_include_bodies(self, symbols_with_bodies):
        fn = _find(symbols_with_bodies, "move", parent="PlayerController")
        assert "move_and_slide" in fn.content


# ── HLSL ───────────────────────────────────────────────────────────────


class TestHLSLParser:
    @pytest.fixture()
    def symbols(self):
        return _parse_fixture("hlsl", "sample.hlsl")

    @pytest.fixture()
    def symbols_with_bodies(self):
        return _parse_fixture("hlsl", "sample.hlsl", include_bodies=True)

    def test_define_constant(self, symbols):
        sym = _find(symbols, "GI_INTENSITY")
        assert sym.chunk_type == ChunkType.CONSTANT
        assert sym.metadata.get("value") == "1.5"
        assert "Global illumination" in sym.summary

    def test_cbuffer(self, symbols):
        sym = _find(symbols, "FrameConstants")
        assert sym.chunk_type == ChunkType.SHADER_UNIFORM_BLOCK
        assert sym.metadata.get("register") == "b0"

    def test_cbuffer_members(self, symbols):
        vp = _find(symbols, "ViewProjection", parent="FrameConstants")
        assert vp.chunk_type == ChunkType.PROPERTY
        cam = _find(symbols, "CameraPosition", parent="FrameConstants")
        assert cam.chunk_type == ChunkType.PROPERTY

    def test_texture_resource(self, symbols):
        sym = _find(symbols, "AlbedoTex")
        assert sym.chunk_type == ChunkType.SHADER_RESOURCE
        assert "Texture2D" in sym.metadata.get("type", "")
        assert sym.metadata.get("register") == "t0"

    def test_struct(self, symbols):
        sym = _find(symbols, "VSOutput")
        assert sym.chunk_type == ChunkType.CLASS_OVERVIEW
        assert "Vertex shader output" in sym.content or "Vertex shader output" in sym.summary

    def test_struct_members(self, symbols):
        pos = _find(symbols, "Position", parent="VSOutput")
        assert pos.chunk_type == ChunkType.PROPERTY
        assert pos.metadata.get("semantic") == "SV_POSITION"
        tc = _find(symbols, "TexCoord", parent="VSOutput")
        assert tc.chunk_type == ChunkType.PROPERTY
        assert tc.metadata.get("semantic") == "TEXCOORD0"
        nrm = _find(symbols, "Normal", parent="VSOutput")
        assert nrm.chunk_type == ChunkType.PROPERTY
        assert nrm.metadata.get("semantic") == "NORMAL"

    def test_regular_function(self, symbols):
        sym = _find(symbols, "TransformPosition")
        assert sym.chunk_type == ChunkType.METHOD
        assert sym.metadata.get("return_type") == "float4"

    def test_pixel_shader_entry_point(self, symbols):
        sym = _find(symbols, "PSMain")
        assert sym.chunk_type == ChunkType.SHADER_ENTRY_POINT
        assert sym.metadata.get("semantic") == "SV_Target"

    def test_compute_entry_point(self, symbols):
        sym = _find(symbols, "CSMain")
        assert sym.chunk_type == ChunkType.SHADER_ENTRY_POINT
        assert sym.metadata.get("numthreads") == (8, 8, 1)

    def test_doc_comments(self, symbols):
        assert "Pixel shader entry point" in _find(symbols, "PSMain").summary
        assert "Global illumination" in _find(symbols, "GI_INTENSITY").summary

    def test_no_bodies_shows_signature(self, symbols):
        sym = _find(symbols, "PSMain")
        assert "```hlsl" in sym.content
        assert "PSMain" in sym.content
        assert "AlbedoTex.Sample" not in sym.content

    def test_include_bodies(self, symbols_with_bodies):
        sym = _find(symbols_with_bodies, "PSMain")
        assert "AlbedoTex.Sample" in sym.content

    def test_hlsl_unterminated_block_no_crash(self):
        parser = get_parser("hlsl")
        result = parser.parse("cbuffer Broken : register(b0) {\n    float4 val;\n")
        assert isinstance(result, list)


# ── USF ────────────────────────────────────────────────────────────────


class TestUSFParser:
    @pytest.fixture()
    def symbols(self):
        return _parse_fixture("usf", "sample.usf")

    @pytest.fixture()
    def symbols_with_bodies(self):
        return _parse_fixture("usf", "sample.usf", include_bodies=True)

    def test_parameter_struct(self, symbols):
        sym = _find(symbols, "FMyShaderParameters")
        assert sym.chunk_type == ChunkType.SHADER_UNIFORM_BLOCK
        assert sym.metadata.get("ue_parameter_struct") is True

    def test_shader_parameter(self, symbols):
        sym = _find(symbols, "Intensity", parent="FMyShaderParameters")
        assert sym.chunk_type == ChunkType.PROPERTY
        assert sym.metadata.get("ue_parameter_kind") == "SHADER_PARAMETER"

    def test_shader_parameter_texture(self, symbols):
        sym = _find(symbols, "SceneColor", parent="FMyShaderParameters")
        assert sym.chunk_type == ChunkType.PROPERTY
        assert sym.metadata.get("ue_parameter_kind") == "SHADER_PARAMETER_TEXTURE"

    def test_entry_point(self, symbols):
        sym = _find(symbols, "PSMain")
        assert sym.chunk_type == ChunkType.SHADER_ENTRY_POINT
        assert sym.metadata.get("semantic") == "SV_Target"

    def test_no_bodies_shows_signature(self, symbols):
        sym = _find(symbols, "PSMain")
        assert "```hlsl" in sym.content
        assert "PSMain" in sym.content

    def test_include_bodies(self, symbols_with_bodies):
        sym = _find(symbols_with_bodies, "PSMain")
        assert "UV" in sym.content

    def test_hlsl_define_preserved(self, symbols):
        sym = _find(symbols, "CUSTOM_BLEND_MODE")
        assert sym.chunk_type == ChunkType.CONSTANT

    def test_hlsl_cbuffer_preserved(self, symbols):
        sym = _find(symbols, "MaterialConstants")
        assert sym.chunk_type == ChunkType.SHADER_UNIFORM_BLOCK
        assert sym.metadata.get("register") == "b1"

    def test_cbuffer_member(self, symbols):
        sym = _find(symbols, "Opacity", parent="MaterialConstants")
        assert sym.chunk_type == ChunkType.PROPERTY

    def test_no_include_chunk(self, symbols):
        names = [s.name for s in symbols]
        assert "Common.ush" not in names
        assert "__includes__" not in names
        assert not any("include" in s.name.lower() for s in symbols)

    def test_usf_unterminated_block_no_crash(self):
        parser = get_parser("usf")
        result = parser.parse(
            "BEGIN_SHADER_PARAMETER_STRUCT(FBroken,\n    SHADER_PARAMETER(float, X)"
        )
        assert isinstance(result, list)


# ── GLSL ───────────────────────────────────────────────────────────────


class TestGLSLParser:
    @pytest.fixture()
    def symbols(self):
        return _parse_fixture("glsl", "sample.frag")

    @pytest.fixture()
    def symbols_with_bodies(self):
        return _parse_fixture("glsl", "sample.frag", include_bodies=True)

    def test_uniform_block(self, symbols):
        sym = _find(symbols, "CameraBlock")
        assert sym.chunk_type == ChunkType.SHADER_UNIFORM_BLOCK
        layout = str(sym.metadata.get("layout", ""))
        assert "std140" in layout or "binding" in layout

    def test_uniform_block_members(self, symbols):
        vp = _find(symbols, "viewProjection", parent="CameraBlock")
        assert vp.chunk_type == ChunkType.PROPERTY
        cp = _find(symbols, "cameraPos", parent="CameraBlock")
        assert cp.chunk_type == ChunkType.PROPERTY

    def test_sampler_resource(self, symbols):
        sym = _find(symbols, "albedoMap")
        assert sym.chunk_type == ChunkType.SHADER_RESOURCE

    def test_in_qualifier(self, symbols):
        sym = _find(symbols, "fragTexCoord")
        assert sym.chunk_type == ChunkType.PROPERTY
        assert sym.metadata.get("qualifier") == "in"

    def test_out_qualifier(self, symbols):
        sym = _find(symbols, "outColor")
        assert sym.chunk_type == ChunkType.PROPERTY
        assert sym.metadata.get("qualifier") == "out"

    def test_helper_function(self, symbols):
        sym = _find(symbols, "computeLighting")
        assert sym.chunk_type == ChunkType.METHOD
        assert sym.metadata.get("return_type") == "vec3"

    def test_main_entry_point(self, symbols):
        sym = _find(symbols, "main")
        assert sym.chunk_type == ChunkType.SHADER_ENTRY_POINT

    def test_doc_comments(self, symbols):
        sym = _find(symbols, "computeLighting")
        assert "Helper to compute" in sym.summary

    def test_no_bodies_shows_signature(self, symbols):
        sym = _find(symbols, "computeLighting")
        assert "```glsl" in sym.content
        assert "computeLighting" in sym.content

    def test_include_bodies(self, symbols_with_bodies):
        sym = _find(symbols_with_bodies, "main")
        assert "texture(albedoMap" in sym.content

    def test_version_directive_not_a_symbol(self, symbols):
        names = [s.name for s in symbols]
        assert "450" not in names
        assert not any("version" in n.lower() for n in names)

    def test_glsl_unterminated_block_no_crash(self):
        parser = get_parser("glsl")
        result = parser.parse("layout(std140) uniform Broken {\n    vec4 val;\n")
        assert isinstance(result, list)


# ── Godot GLSL ─────────────────────────────────────────────────────────


class TestGodotGLSLParser:
    @pytest.fixture()
    def symbols(self):
        return _parse_fixture("glsl", "sample.gdshader")

    def test_vertex_entry_point(self, symbols):
        sym = _find(symbols, "vertex")
        assert sym.chunk_type == ChunkType.SHADER_ENTRY_POINT

    def test_fragment_entry_point(self, symbols):
        sym = _find(symbols, "fragment")
        assert sym.chunk_type == ChunkType.SHADER_ENTRY_POINT

    def test_sampler_with_godot_hint(self, symbols):
        sym = _find(symbols, "noise_tex")
        assert sym.chunk_type == ChunkType.SHADER_RESOURCE
        assert sym.metadata.get("godot_hint") == "hint_albedo"

    def test_uniform_with_hint_range(self, symbols):
        sym = _find(symbols, "speed")
        assert sym.chunk_type == ChunkType.PROPERTY
        assert "hint_range" in str(sym.metadata.get("godot_hint", ""))
        assert sym.metadata.get("default") == "1.0"

    def test_shader_type_metadata(self, symbols):
        assert symbols[0].metadata.get("file", {}).get("shader_type") == "canvas_item"

    def test_render_mode_metadata(self, symbols):
        assert "blend_mix" in str(symbols[0].metadata.get("file", {}).get("render_mode", ""))

    def test_godot_unterminated_no_crash(self):
        parser = get_parser("glsl")
        result = parser.parse("shader_type canvas_item;\nuniform Block {\n    float x;")
        assert isinstance(result, list)
