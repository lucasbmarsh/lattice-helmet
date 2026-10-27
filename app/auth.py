"""Onshape OAuth for the panel.

Onshape's sign-in page can't be shown inside the panel's iframe, so the
panel opens /oauth/login in a popup with a random `state` it made, then
polls /oauth/poll with that state until the callback has swapped the code
for tokens. Polling avoids relying on window.opener, which Onshape's pages
may cut off.

The panel keeps the tokens and sends the access token as a Bearer header;
the service holds only the client secret, so a restart (or a machine that
slept) doesn't sign anyone out. /oauth/refresh renews an expired token.

Set ONSHAPE_CLIENT_ID and ONSHAPE_CLIENT_SECRET to turn this on, and
LATTICE_ALLOWED_USERS (comma-separated Onshape emails) to say who may use
the service. PUBLIC_URL fixes the redirect URI when the request's own
address is wrong (e.g. behind a proxy that hides https).
"""

import os
import re
import threading
import time

import httpx

OAUTH_URL = os.environ.get("ONSHAPE_OAUTH_URL", "https://oauth.onshape.com")
STATE_RE = re.compile(r"^[A-Za-z0-9_-]{22,128}$")
PENDING_TTL = 600
SESSION_TTL = 600


class OAuthError(RuntimeError):
    pass


class OAuth:
    def __init__(self, client_id, client_secret, allowed_users=(), public_url=None,
                 oauth_url=OAUTH_URL, http=None):
        self.client_id, self.client_secret = client_id, client_secret
        self.allowed = {u.strip().lower() for u in allowed_users if u.strip()}
        self.public_url = public_url.rstrip("/") if public_url else None
        self.oauth_url = oauth_url
        self.http = http or httpx.Client(timeout=30)
        self.pending = {}  # state -> (created, tokens or None)
        self.sessions = {}  # access token -> (checked at, user)
        self.lock = threading.Lock()

    @classmethod
    def from_env(cls):
        cid, secret = os.environ.get("ONSHAPE_CLIENT_ID"), os.environ.get("ONSHAPE_CLIENT_SECRET")
        if not (cid and secret):
            return None
        return cls(cid, secret, os.environ.get("LATTICE_ALLOWED_USERS", "").split(","),
                   os.environ.get("PUBLIC_URL"))

    def redirect_uri(self, base_url):
        return (self.public_url or str(base_url).rstrip("/")) + "/oauth/callback"

    def start(self, state, redirect_uri):
        if not STATE_RE.match(state or ""):
            raise OAuthError("bad state")
        now = time.time()
        with self.lock:
            for s, (t, _) in list(self.pending.items()):
                if now - t > PENDING_TTL:
                    del self.pending[s]
            self.pending[state] = (now, None)
        q = httpx.QueryParams(response_type="code", client_id=self.client_id,
                              redirect_uri=redirect_uri, state=state)
        return f"{self.oauth_url}/oauth/authorize?{q}"

    def finish(self, state, code, redirect_uri):
        with self.lock:
            if state not in self.pending:
                raise OAuthError("sign-in expired, try again from the panel")
        tokens = self._token(grant_type="authorization_code", code=code, redirect_uri=redirect_uri)
        with self.lock:
            self.pending[state] = (time.time(), tokens)

    def poll(self, state):
        """Tokens once the callback has run, None while waiting; KeyError if unknown."""
        with self.lock:
            created, tokens = self.pending[state]
            if tokens:
                del self.pending[state]
            return tokens

    def refresh(self, refresh_token):
        return self._token(grant_type="refresh_token", refresh_token=refresh_token)

    def _token(self, **form):
        r = self.http.post(f"{self.oauth_url}/oauth/token",
                           data={**form, "client_id": self.client_id, "client_secret": self.client_secret})
        if r.status_code >= 400:
            raise OAuthError(f"token request -> {r.status_code}: {r.text[:300]}")
        t = r.json()
        return {"access_token": t["access_token"], "refresh_token": t.get("refresh_token", form.get("refresh_token")),
                "expires_at": int(time.time()) + int(t.get("expires_in", 3600))}

    def check(self, token, session_info):
        """The Onshape user behind `token`, cached for a while.

        session_info() is called on a cache miss and should return Onshape's
        /api/users/sessioninfo. Raises PermissionError for users not allowed.
        """
        now = time.time()
        with self.lock:
            hit = self.sessions.get(token)
        if hit and now - hit[0] < SESSION_TTL:
            user = hit[1]
        else:
            info = session_info()
            user = {"id": info.get("id"), "email": (info.get("email") or "").lower(), "name": info.get("name")}
            with self.lock:
                self.sessions = {k: v for k, v in self.sessions.items() if now - v[0] < SESSION_TTL}
                self.sessions[token] = (now, user)
        if self.allowed and user["email"] not in self.allowed and (user["id"] or "") not in self.allowed:
            raise PermissionError(f"{user['email'] or user['id']} is not allowed to use this service")
        return user
