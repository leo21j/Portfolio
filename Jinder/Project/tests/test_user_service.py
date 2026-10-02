import hashlib

from services.user_service.user_service import UserService


def service() -> UserService:
    """A UserService without __init__, so no AWS connection is attempted.

    Password hashing does not touch DynamoDB; only the surrounding CRUD does.
    """
    return UserService.__new__(UserService)


def test_hash_is_not_a_bare_sha256_digest():
    svc = service()
    stored = svc._hash_password("hunter2")
    assert stored != hashlib.sha256(b"hunter2").hexdigest()
    assert stored.startswith("pbkdf2_sha256$")


def test_same_password_hashes_differently_each_time():
    # A random per-user salt is what stops one rainbow table from covering
    # every account at once.
    svc = service()
    assert svc._hash_password("hunter2") != svc._hash_password("hunter2")


def test_correct_password_verifies():
    svc = service()
    valid, needs_upgrade = svc._verify_password("hunter2", svc._hash_password("hunter2"))
    assert valid
    assert not needs_upgrade


def test_wrong_password_rejected():
    svc = service()
    valid, _ = svc._verify_password("wrong", svc._hash_password("hunter2"))
    assert not valid


def test_legacy_sha256_hash_still_verifies_and_flags_upgrade():
    # Accounts created before salted hashing must keep working, or every
    # existing user is locked out on deploy.
    svc = service()
    legacy = hashlib.sha256(b"hunter2").hexdigest()
    valid, needs_upgrade = svc._verify_password("hunter2", legacy)
    assert valid
    assert needs_upgrade


def test_wrong_password_against_legacy_hash_is_rejected():
    svc = service()
    legacy = hashlib.sha256(b"hunter2").hexdigest()
    valid, _ = svc._verify_password("wrong", legacy)
    assert not valid


def test_malformed_hash_is_rejected_without_raising():
    svc = service()
    valid, needs_upgrade = svc._verify_password("hunter2", "pbkdf2_sha256$not$valid")
    assert not valid
    assert not needs_upgrade


def test_email_normalization():
    svc = service()
    assert svc._normalize_email("  Leo@Example.COM ") == "leo@example.com"
