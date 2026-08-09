"""Adversarial tests for the inert shared formal-authority foundation."""

from __future__ import annotations

import hashlib
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import pytest

from thermoroute import formal
from thermoroute.formal import authority as authority_module


def _captured_file(tmp_path: Path, name: str = "protocol.json", data: bytes = b"{}"):
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return root, path, formal.capture_binding(root, PurePosixPath(name))


def _pending(tmp_path: Path, *, action: str = "inspect", profile: str = "route-a"):
    root, _, binding = _captured_file(tmp_path)
    pending = formal.issue_pending_authority(
        action=action,
        profile=profile,
        bindings=(binding,),
        payload={"limits": {"allowed": [1, 3, 7]}, "version": 1},
    )
    return root, binding, pending


def test_capture_retains_bytes_digest_size_and_canonical_path(tmp_path: Path) -> None:
    content = b'{"format":"fixture.v1"}'
    root, _, binding = _captured_file(tmp_path, "protocols/frozen.json", content)

    assert binding.path == PurePosixPath("protocols/frozen.json")
    assert binding.content == content
    assert binding.size == len(content)
    assert binding.sha256 == hashlib.sha256(content).hexdigest()
    assert formal.capture_snapshot(
        root,
        [PurePosixPath("protocols/frozen.json")],
    ) == (binding,)


@pytest.mark.parametrize(
    "relative",
    [
        PurePosixPath(),
        PurePosixPath(".."),
        PurePosixPath("../outside"),
        PurePosixPath("inside/../../outside"),
        PurePosixPath("/absolute"),
        PurePosixPath("//absolute"),
        PurePosixPath("bad\x00name"),
    ],
)
def test_capture_rejects_empty_absolute_escape_and_invalid_components(
    tmp_path: Path, relative: PurePosixPath
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(formal.PathPolicyError):
        formal.capture_binding(root, relative)


def test_capture_requires_exact_pure_posix_path_and_absolute_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "file").write_bytes(b"x")

    with pytest.raises(formal.PathPolicyError):
        formal.capture_binding(root, "file")  # type: ignore[arg-type]
    with pytest.raises(formal.PathPolicyError):
        formal.capture_binding(Path("relative-root"), PurePosixPath("file"))
    with pytest.raises(formal.PathPolicyError):
        formal.capture_binding(Path("//tmp"), PurePosixPath("file"))


def test_capture_rejects_root_file_and_root_symlink(tmp_path: Path) -> None:
    root_file = tmp_path / "not-a-root"
    root_file.write_bytes(b"x")
    with pytest.raises(formal.FileTypeError):
        formal.capture_binding(root_file, PurePosixPath("file"))

    real_root = tmp_path / "real"
    real_root.mkdir()
    (real_root / "file").write_bytes(b"x")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(formal.SymlinkRejectedError):
        formal.capture_binding(linked_root, PurePosixPath("file"))


def test_capture_rejects_symlink_in_absolute_root_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    repository = real_parent / "repo"
    repository.mkdir(parents=True)
    (repository / "file").write_bytes(b"content")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(formal.SymlinkRejectedError):
        formal.capture_binding(linked_parent / "repo", PurePosixPath("file"))


def test_capture_rejects_leaf_and_intermediate_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "real.txt").write_bytes(b"content")
    (root / "leaf.txt").symlink_to("real.txt")
    with pytest.raises(formal.SymlinkRejectedError):
        formal.capture_binding(root, PurePosixPath("leaf.txt"))

    real_directory = root / "real-directory"
    real_directory.mkdir()
    (real_directory / "file.txt").write_bytes(b"content")
    (root / "linked-directory").symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(formal.SymlinkRejectedError):
        formal.capture_binding(root, PurePosixPath("linked-directory/file.txt"))


def test_capture_rejects_hardlinks_without_an_override(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    original = root / "original"
    original.write_bytes(b"content")
    os.link(original, root / "alias")

    assert formal.HARDLINK_POLICY == "reject-st_nlink-not-one"
    with pytest.raises(formal.HardlinkRejectedError):
        formal.capture_binding(root, PurePosixPath("original"))
    with pytest.raises(formal.HardlinkRejectedError):
        formal.capture_binding(root, PurePosixPath("alias"))


def test_capture_rejects_fifo_without_opening_it(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    os.mkfifo(root / "pipe")

    with pytest.raises(formal.FileTypeError):
        formal.capture_binding(root, PurePosixPath("pipe"))


def test_capture_rejects_device_file() -> None:
    if not Path("/dev/null").exists():
        pytest.skip("POSIX null device is unavailable")
    with pytest.raises(formal.FileTypeError):
        formal.capture_binding(Path("/dev"), PurePosixPath("null"))


def test_snapshot_is_sorted_unique_and_revalidated_by_recapture(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "z").write_bytes(b"last")
    (root / "a").write_bytes(b"first")
    snapshot = formal.capture_snapshot(root, [PurePosixPath("z"), PurePosixPath("a")])

    assert [binding.path.as_posix() for binding in snapshot] == ["a", "z"]
    assert formal.recapture_snapshot(root, snapshot) == snapshot
    formal.revalidate_snapshot(root, snapshot)

    with pytest.raises(formal.PathPolicyError):
        formal.capture_snapshot(root, [PurePosixPath("a"), PurePosixPath("a")])


def test_snapshot_detects_same_size_same_mtime_content_mutation(tmp_path: Path) -> None:
    root, path, binding = _captured_file(tmp_path, data=b"before")
    original_stat = path.stat()
    path.write_bytes(b"after!")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    changed_stat = path.stat()
    assert changed_stat.st_size == original_stat.st_size
    assert changed_stat.st_mtime_ns == original_stat.st_mtime_ns
    with pytest.raises(formal.SnapshotMismatchError):
        formal.revalidate_snapshot(root, (binding,))


def test_binding_is_frozen_and_tampering_is_detected(tmp_path: Path) -> None:
    _, _, binding = _captured_file(tmp_path, data=b"bound")
    with pytest.raises(FrozenInstanceError):
        binding.size = 0  # type: ignore[misc]
    with pytest.raises(formal.BindingValidationError):
        replace(binding, sha256="0" * 64)

    pending = formal.issue_pending_authority(
        action="inspect",
        profile="route-a",
        bindings=(binding,),
        payload={},
    )
    object.__setattr__(binding, "content", b"other")
    with pytest.raises(formal.AuthorityValidationError):
        formal.require_pending_authority(pending, action="inspect", profile="route-a")


def test_canonical_json_round_trip_is_sorted_finite_utf8_and_immutable() -> None:
    value = {"z": [1, True, None], "é": {"b": 2, "a": "water"}}
    payload = formal.canonical_json_bytes(value)

    assert payload == b'{"z":[1,true,null],"\\u00e9":{"a":"water","b":2}}\n'
    parsed = formal.parse_canonical_json_object(payload)
    assert type(parsed) is MappingProxyType
    assert type(parsed["z"]) is tuple
    assert type(parsed["é"]) is MappingProxyType
    assert formal.canonical_json_bytes(parsed) == payload
    with pytest.raises(TypeError):
        parsed["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        parsed["é"]["a"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "payload,exception",
    [
        (b'{"a":1,"a":2}\n', formal.DuplicateJSONKeyError),
        (b'{"outer":{"a":1,"a":2}}\n', formal.DuplicateJSONKeyError),
        (b"\xef\xbb\xbf{}\n", formal.CanonicalJSONError),
        (b'{"value":"\xff"}\n', formal.CanonicalJSONError),
        (b'{"value":NaN}\n', formal.CanonicalJSONError),
        (b'{"value":Infinity}\n', formal.CanonicalJSONError),
        (b'{"value":-Infinity}\n', formal.CanonicalJSONError),
        (b'{"value":1e400}\n', formal.CanonicalJSONError),
        (b"[]\n", formal.CanonicalJSONError),
        (b"null\n", formal.CanonicalJSONError),
        (b"1\n", formal.CanonicalJSONError),
    ],
)
def test_canonical_object_parser_rejects_ambiguous_or_non_object_json(
    payload: bytes, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        formal.parse_canonical_json_object(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"b":2,"a":1}\n',
        b'{"a": 1}\n',
        b'{"a":1}',
        b'{"a":1}\n\n',
        b'{"a":"\xc3\xa9"}\n',
        b'{"a":"\\u00E9"}\n',
        b'{"a":1e0}\n',
        b'{"a":-0}\n',
    ],
)
def test_canonical_object_parser_rejects_noncanonical_bytes(payload: bytes) -> None:
    with pytest.raises(formal.CanonicalJSONError, match="not canonical"):
        formal.parse_canonical_json_object(payload)


def test_canonical_object_parser_accepts_ascii_escaped_unicode_and_one_newline() -> None:
    payload = b'{"a":"\\u00e9","emoji":"\\ud83c\\udf0a"}\n'
    parsed = formal.parse_canonical_json_object(payload)

    assert parsed == {"a": "é", "emoji": "🌊"}
    assert formal.canonical_json_bytes(parsed) == payload


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_serializer_rejects_nonfinite_numbers(value: float) -> None:
    with pytest.raises(formal.CanonicalJSONError):
        formal.canonical_json_bytes({"value": value})


@pytest.mark.parametrize(
    "value",
    [
        {"value": "\ud83c"},
        {"value": "\ud83c\udf0a"},
        {"\ud83c": "value"},
    ],
)
def test_canonical_serializer_rejects_non_scalar_surrogate_code_points(
    value: dict[str, str],
) -> None:
    with pytest.raises(formal.CanonicalJSONError, match="Unicode scalar"):
        formal.canonical_json_bytes(value)


def test_canonical_parser_rejects_lone_surrogate_but_accepts_valid_pair() -> None:
    with pytest.raises(formal.CanonicalJSONError, match="canonical JSON"):
        formal.parse_canonical_json_object(b'{"value":"\\ud83c"}\n')

    payload = b'{"value":"\\ud83c\\udf0a"}\n'
    parsed = formal.parse_canonical_json_object(payload)
    assert parsed == {"value": "\U0001f30a"}
    assert formal.canonical_json_bytes(parsed) == payload


def test_exact_closed_field_helper_rejects_missing_extra_and_duplicate_contract() -> None:
    document = formal.parse_canonical_json_object(b'{"action":"inspect","profile":"route-a"}\n')
    formal.require_exact_fields(document, ("action", "profile"), label="request")

    with pytest.raises(formal.JSONFieldError, match="missing"):
        formal.require_exact_fields(document, ("action", "profile", "bindings"))
    with pytest.raises(formal.JSONFieldError, match="unexpected"):
        formal.require_exact_fields(document, ("action",))
    with pytest.raises(formal.JSONFieldError, match="unique"):
        formal.require_exact_fields(document, ("action", "action"))


def test_freeze_json_defensively_copies_every_nested_container() -> None:
    source = {"nested": {"values": [1, 2]}}
    frozen = formal.freeze_json(source)
    source["nested"]["values"].append(3)
    source["new"] = True

    assert frozen == {"nested": {"values": (1, 2)}}
    assert type(frozen) is MappingProxyType
    assert type(frozen["nested"]) is MappingProxyType
    assert type(frozen["nested"]["values"]) is tuple


def test_pending_factory_freezes_slots_payload_and_seals_content(tmp_path: Path) -> None:
    root, _, binding = _captured_file(tmp_path)
    payload = {"nested": {"values": [1, 2]}}
    pending = formal.issue_pending_authority(
        action="inspect",
        profile="route-a",
        bindings=(binding,),
        payload=payload,
    )
    payload["nested"]["values"].append(3)

    assert not hasattr(pending, "__dict__")
    assert pending.payload == {"nested": {"values": (1, 2)}}
    assert len(pending.content_seal) == 64
    formal.require_pending_authority(pending, action="inspect", profile="route-a")
    formal.revalidate_authority_snapshot(
        root,
        pending,
        action="inspect",
        profile="route-a",
    )
    with pytest.raises(FrozenInstanceError):
        pending.action = "publish"  # type: ignore[misc]
    with pytest.raises(TypeError):
        pending.payload["new"] = True  # type: ignore[index]


def test_pending_rejects_direct_subclass_object_new_replace_and_bad_issuer(
    tmp_path: Path,
) -> None:
    _, _, pending = _pending(tmp_path)

    with pytest.raises(formal.AuthorityConstructionError):
        formal.PendingAuthority()
    with pytest.raises(formal.AuthorityConstructionError):

        class DerivedPending(formal.PendingAuthority):
            pass

    uninitialized = object.__new__(formal.PendingAuthority)
    with pytest.raises(formal.AuthorityValidationError):
        formal.require_pending_authority(uninitialized, action="inspect", profile="route-a")
    with pytest.raises((TypeError, ValueError)):
        replace(pending, action="publish")

    object.__setattr__(pending, "_issuer", object())
    with pytest.raises(formal.AuthorityValidationError, match="issuer"):
        formal.require_pending_authority(pending, action="inspect", profile="route-a")


def test_pending_content_seal_detects_field_tampering(tmp_path: Path) -> None:
    _, _, pending = _pending(tmp_path)
    object.__setattr__(pending, "action", "publish")

    with pytest.raises(formal.AuthorityValidationError, match="seal"):
        formal.require_pending_authority(pending, action="publish", profile="route-a")


def test_pending_checks_action_and_profile_without_mixing_authorities(tmp_path: Path) -> None:
    _, _, pending = _pending(tmp_path, action="inspect", profile="route-a")

    with pytest.raises(formal.AuthorityMismatchError):
        formal.require_pending_authority(pending, action="publish", profile="route-a")
    with pytest.raises(formal.AuthorityMismatchError):
        formal.require_pending_authority(pending, action="inspect", profile="base")


def test_execution_authority_has_no_public_factory_and_rejects_bypasses(tmp_path: Path) -> None:
    _, _, pending = _pending(tmp_path)

    assert "_issue_execution_authority" not in formal.__all__
    assert not hasattr(formal, "issue_execution_authority")
    with pytest.raises(formal.AuthorityConstructionError):
        formal.ExecutionAuthority()
    with pytest.raises(formal.AuthorityConstructionError):
        authority_module._issue_execution_authority(
            pending,
            action="inspect",
            profile="route-a",
        )
    with pytest.raises(formal.AuthorityConstructionError):

        class DerivedExecution(formal.ExecutionAuthority):
            pass

    uninitialized = object.__new__(formal.ExecutionAuthority)
    with pytest.raises(formal.AuthorityValidationError):
        formal.require_execution_authority(uninitialized, action="inspect", profile="route-a")


def test_internal_execution_issuance_validates_pending_action_profile_and_seal(
    tmp_path: Path,
) -> None:
    _, _, pending = _pending(tmp_path)
    token = authority_module._INTERNAL_EXECUTION_FACTORY_TOKEN

    with pytest.raises(formal.AuthorityMismatchError):
        authority_module._issue_execution_authority(
            pending,
            action="inspect",
            profile="base",
            _factory_token=token,
        )
    execution = authority_module._issue_execution_authority(
        pending,
        action="inspect",
        profile="route-a",
        _factory_token=token,
    )
    assert not hasattr(execution, "__dict__")
    assert execution.bindings is pending.bindings
    assert execution.payload is pending.payload
    formal.require_execution_authority(execution, action="inspect", profile="route-a")

    with pytest.raises((TypeError, ValueError)):
        replace(execution, profile="base")
    object.__setattr__(execution, "profile", "base")
    with pytest.raises(formal.AuthorityValidationError):
        formal.require_execution_authority(execution, action="inspect", profile="base")
