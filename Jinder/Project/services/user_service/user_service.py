import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

# PBKDF2-HMAC-SHA256 via the standard library, so password storage needs no
# extra dependency. 600k iterations matches current OWASP guidance for SHA-256.
_HASH_SCHEME = "pbkdf2_sha256"
_HASH_ITERATIONS = 600_000
_SALT_BYTES = 16


class UserService:
    """User accounts and preferences, backed by a DynamoDB table."""

    def __init__(self, table_name: str | None = None, region: str | None = None):
        self.table_name = table_name or os.environ.get("JINDER_USERS_TABLE", "JinderUsers")
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")

        dynamodb = boto3.resource("dynamodb", region_name=self.region)
        self.table = dynamodb.Table(self.table_name)

    # --- helpers ---

    def _normalize_email(self, email: str) -> str:
        return email.strip().lower()

    def _hash_password(self, password: str, salt: bytes | None = None) -> str:
        """Hash a password as ``pbkdf2_sha256$<iterations>$<salt>$<hash>``.

        A random per-user salt means two users with the same password get
        different stored values, and the iteration count makes brute-forcing a
        leaked table expensive. Both are missing from a bare SHA-256 digest.
        """
        salt = salt if salt is not None else secrets.token_bytes(_SALT_BYTES)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _HASH_ITERATIONS)
        return f"{_HASH_SCHEME}${_HASH_ITERATIONS}${salt.hex()}${digest.hex()}"

    def _verify_password(self, password: str, stored: str) -> tuple[bool, bool]:
        """Check a password against a stored hash.

        Returns ``(is_valid, needs_upgrade)``. Accounts created before salted
        hashing hold a bare SHA-256 hex digest; those still verify, and the
        caller re-hashes them on successful login so the old format drains away
        without locking anyone out.
        """
        if stored.startswith(f"{_HASH_SCHEME}$"):
            try:
                _, iterations, salt_hex, expected = stored.split("$", 3)
                candidate = hashlib.pbkdf2_hmac(
                    "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
                ).hex()
            except (ValueError, TypeError):
                logger.warning("Malformed password hash encountered")
                return False, False
            return hmac.compare_digest(candidate, expected), False

        # Legacy unsalted SHA-256.
        legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(legacy, stored), True

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # public method

    def get_user(self, email: str) -> dict[str, Any] | None:
        email = self._normalize_email(email)
        try:
            resp = self.table.get_item(Key={"email": email})
            return resp.get("Item")
        except (BotoCoreError, ClientError) as e:
            logger.error("get_user failed for %s: %s", email, e)
            return None

    def create_user(
        self,
        email: str,
        password: str,
        default_title: str | None = None,
        default_location: str | None = None,
        remote_pref: str | None = None,
    ) -> dict[str, Any]:
        """
        creates new user. Raises ValueError if user already exists
        """
        email = self._normalize_email(email)
        existing = self.get_user(email)
        if existing:
            raise ValueError("User already exists")

        password_hash = self._hash_password(password)
        now = self._now_iso()

        item: dict[str, Any] = {
            "email": email,
            "password_hash": password_hash,
            "created_at": now,
            "updated_at": now,
        }

        if default_title:
            item["default_title"] = default_title
        if default_location:
            item["default_location"] = default_location
        if remote_pref:
            item["remote_pref"] = remote_pref

        try:
            self.table.put_item(Item=item)
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"Failed to create user: {e}") from e

        # Never hand the hash back to the caller.
        item_copy = dict(item)
        item_copy.pop("password_hash", None)
        return item_copy

    def check_credentials(self, email: str, password: str) -> dict[str, Any] | None:
        """Return the user when the password is correct, otherwise None."""
        email = self._normalize_email(email)
        user = self.get_user(email)
        if not user:
            return None

        stored_hash = user.get("password_hash")
        if not stored_hash:
            return None

        is_valid, needs_upgrade = self._verify_password(password, stored_hash)
        if not is_valid:
            return None

        if needs_upgrade:
            self._upgrade_password_hash(email, password)

        user_copy = dict(user)
        user_copy.pop("password_hash", None)
        return user_copy

    def _upgrade_password_hash(self, email: str, password: str) -> None:
        """Re-store a legacy hash in the salted format after a valid login."""
        try:
            self.table.update_item(
                Key={"email": email},
                UpdateExpression="SET password_hash = :h, updated_at = :u",
                ExpressionAttributeValues={
                    ":h": self._hash_password(password),
                    ":u": self._now_iso(),
                },
            )
            logger.info("Upgraded password hash for %s", email)
        except (BotoCoreError, ClientError) as e:
            # A failed upgrade must not fail the login that triggered it.
            logger.error("Could not upgrade password hash for %s: %s", email, e)

    def update_preferences(
        self,
        email: str,
        default_title: str | None = None,
        default_location: str | None = None,
        remote_pref: str | None = None,
    ) -> dict[str, Any] | None:
        """
        update user preferences. returns updated item / None if user not found
        """
        email = self._normalize_email(email)
        user = self.get_user(email)
        if not user:
            return None

        update_expr_parts = []
        expr_values: dict[str, Any] = {":updated_at": self._now_iso()}
        expr_names: dict[str, str] = {}

        # update updated_at
        update_expr_parts.append("#u = :updated_at")
        expr_names["#u"] = "updated_at"

        if default_title is not None:
            update_expr_parts.append("#t = :title")
            expr_names["#t"] = "default_title"
            expr_values[":title"] = default_title

        if default_location is not None:
            update_expr_parts.append("#l = :loc")
            expr_names["#l"] = "default_location"
            expr_values[":loc"] = default_location

        if remote_pref is not None:
            update_expr_parts.append("#r = :rem")
            expr_names["#r"] = "remote_pref"
            expr_values[":rem"] = remote_pref

        update_expr = "SET " + ", ".join(update_expr_parts)

        try:
            resp = self.table.update_item(
                Key={"email": email},
                UpdateExpression=update_expr,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                ReturnValues="ALL_NEW",
            )
        except (BotoCoreError, ClientError) as e:
            logger.error("update_preferences failed for %s: %s", email, e)
            return None

        item = resp.get("Attributes", {})
        item.pop("password_hash", None)
        return item
