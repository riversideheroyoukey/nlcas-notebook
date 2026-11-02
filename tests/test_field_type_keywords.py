""""ベクトル場 / スカラー場" must map to define_vector_field / define_scalar_field
(vector-calculus fields with components), not the generic define_vector — so that
div/curl/grad work afterwards.  Regression for S-f generatability."""
from notebook.runtime.backend_handler import _infer_type_from_label


def test_vector_field_keyword():
    assert _infer_type_from_label("ベクトル場") == "define_vector_field"
    assert _infer_type_from_label("ベクトル フィールド") == "define_vector_field"
    assert _infer_type_from_label("vector field") == "define_vector_field"


def test_scalar_field_keyword():
    assert _infer_type_from_label("スカラー場") == "define_scalar_field"
    assert _infer_type_from_label("scalar field") == "define_scalar_field"


def test_plain_vector_unchanged():
    # "ベクトル" without 場 stays a plain vector quantity
    assert _infer_type_from_label("ベクトル") == "define_vector"
    assert _infer_type_from_label("vector") == "define_vector"


def test_scalar_default_unchanged():
    assert _infer_type_from_label("スカラー") == "define_scalar_parameter"
    assert _infer_type_from_label("運動量") == "define_scalar_parameter"
