"""Closed-loop per-turn router: an OpenAI-compatible proxy that picks an arm per call.

mini-swe-agent (under Pier, in a Modal sandbox) is pointed at this endpoint via
OPENAI_API_BASE. Pier automatically adds our hostname to the sandbox's network
allowlist (pier/agents/installed/mini_swe_agent.py: network_allowlist()), so this is a
supported integration rather than a hack.

Per request the proxy:
  1. reconstructs episode state from the incoming messages (turn index, struggle
     signals from the last tool outputs, cost so far, previous arm),
  2. scores arms with a small linear policy (weights in a Modal Volume),
  3. forwards the call to the real provider with that arm's model + reasoning effort,
  4. appends (state, action, usage, cost) to a per-episode JSONL in the Volume.

The training loop (rl/perturn_live.py) reads those logs, joins them to Pier's
verifier reward, and updates the policy — REINFORCE with a group baseline over M
rollouts of the same task. Nothing in the loop touches the offline matrix.

Deploy:  modal deploy rl/router_proxy.py
Episode: pier run ... --model openai/router \
           --ae OPENAI_API_BASE=https://<app>--router.modal.run/v1 \
           --ae OPENAI_API_KEY=$ROUTER_TOKEN --ae ROUTER_EPISODE=<task>__<rollout>
"""
import json
import os
import time

import modal

app = modal.App("coding-router-proxy")
vol = modal.Volume.from_name("router-proxy-state", create_if_missing=True)
STATE = "/state"

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("fastapi[standard]", "httpx", "numpy"))

# Arm pool = the live-benchmarked frontier (EXP-015). price is $/task observed live,
# used only for the reward; the policy sees it as a feature.
ARMS = [
    # Phase 1: OpenAI-only so every arm speaks the Responses API natively. gpt-5.6 models
    # REJECT function tools + reasoning_effort on /v1/chat/completions (they require
    # /v1/responses), and mini-swe-agent does send tools. Anthropic arms need a
    # Responses<->Messages translation and land in phase 2.
    # price/f2p are the live EXP-015 measurements.
    {"id": "luna_medium", "provider": "openai", "model": "gpt-5.6-luna", "effort": "medium",
     "price": 0.031},
    {"id": "luna_high", "provider": "openai", "model": "gpt-5.6-luna", "effort": "high",
     "price": 0.141},
    {"id": "terra_high", "provider": "openai", "model": "gpt-5.6-terra", "effort": "high",
     "price": 0.770},
    {"id": "terra_max", "provider": "openai", "model": "gpt-5.6-terra", "effort": "max",
     "price": 0.421},
    {"id": "sol_xhigh", "provider": "openai", "model": "gpt-5.6-sol", "effort": "xhigh",
     "price": 4.681},
]
N_ARMS = len(ARMS)
FEATS = 8  # see features()


def features(turn: int, cost_so_far: float, last_ok: bool, err_hits: int,
             test_pass: bool, prev_arm: int, n_msgs: int) -> list[float]:
    """Episode state visible to the router at this turn. Deliberately cheap: no model
    inference inside the proxy, so per-turn latency stays negligible."""
    import math
    return [
        1.0,
        min(turn, 60) / 60.0,
        math.log10(cost_so_far + 1e-3) / 3.0 + 1.0,
        1.0 if last_ok else 0.0,
        min(err_hits, 5) / 5.0,
        1.0 if test_pass else 0.0,
        (prev_arm + 1) / N_ARMS if prev_arm >= 0 else 0.0,
        min(n_msgs, 120) / 120.0,
    ]


def load_policy():
    p = os.path.join(STATE, "policy.json")
    if os.path.exists(p):
        with open(p) as fh:
            d = json.load(fh)
        return d["W"], float(d.get("temp", 1.0)), float(d.get("explore", 0.05))
    # cold start: prefer the cheap-but-strong arm, mild preference gradient by price
    W = [[0.0] * FEATS for _ in range(N_ARMS)]
    for a in range(N_ARMS):
        W[a][0] = 1.0 if ARMS[a]["id"] == "luna_high" else 0.0  # cheap-strong default
    return W, 1.0, 0.15


def parse_state(messages):
    """Reconstruct turn index and struggle signals from the conversation so far."""
    turn = sum(1 for m in messages if m.get("role") == "assistant")
    text = ""
    for m in messages[-3:]:
        c = m.get("content")
        if isinstance(c, str):
            text += c
        elif isinstance(c, list):
            text += " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    low = text.lower()
    err_hits = sum(low.count(k) for k in ("traceback", "error:", "failed", "fatal"))
    last_ok = ("exit=0" in low) or ("exit code: 0" in low)
    test_pass = ("tests passed" in low) or ("all tests pass" in low)
    return turn, last_ok, err_hits, test_pass, len(messages)


def _decide(messages, ep):
    """Shared policy step: rebuild state, sample an arm, return (arm_index, x, p)."""
    import numpy as np
    turn, last_ok, err_hits, test_pass, n_msgs = parse_state(messages)
    ep_path = os.path.join(STATE, "episodes", f"{ep}.jsonl")
    os.makedirs(os.path.dirname(ep_path), exist_ok=True)
    cost_so_far, prev_arm = 0.0, -1
    if os.path.exists(ep_path):
        with open(ep_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    cost_so_far += r.get("cost", 0.0)
                    prev_arm = r.get("arm", -1)
                except json.JSONDecodeError:
                    pass
    W, temp, explore = load_policy()
    x = np.array(features(turn, cost_so_far, last_ok, err_hits, test_pass, prev_arm, n_msgs))
    z = np.array(W) @ x
    p = np.exp((z - z.max()) / max(temp, 1e-3))
    p = (1 - explore) * p / p.sum() + explore / N_ARMS
    a = int(np.random.choice(N_ARMS, p=p))
    return a, x, p, turn, ep_path


def _log(ep_path, turn, a, x, p, usage, t0, err=None):
    cost = (usage.get("prompt_tokens", usage.get("input_tokens", 0)) * 1e-6 * 2.0
            + usage.get("completion_tokens", usage.get("output_tokens", 0)) * 1e-6 * 10.0)
    with open(ep_path, "a") as fh:
        fh.write(json.dumps({"turn": turn, "arm": a, "arm_id": ARMS[a]["id"],
                             "x": list(x), "p": list(p), "cost": cost,
                             "usage": usage, "err": err,
                             "wall": round(time.time() - t0, 2)}) + "\n")
    vol.commit()


@app.function(image=image, volumes={STATE: vol}, timeout=3600, max_containers=40,
              secrets=[modal.Secret.from_name("router-provider-keys")])
@modal.asgi_app()
def router():
    """OpenAI-compatible surface. litellm appends /chat/completions or /responses to
    OPENAI_API_BASE, so both paths are served."""
    import httpx
    from fastapi import FastAPI, Request

    web = FastAPI()

    @web.get("/health")
    async def health():
        W, temp, explore = load_policy()
        return {"ok": True, "arms": [a["id"] for a in ARMS], "explore": explore}

    def _normalize(body, arm):
        """Same request from the agent must be valid for every arm we might pick."""
        body["model"] = arm["model"]
        body.pop("model_class", None)
        if "max_tokens" in body:                       # gpt-5.6 family rejects it
            body["max_completion_tokens"] = body.pop("max_tokens")
        for k in ("temperature", "top_p"):             # reasoning models reject these
            body.pop(k, None)
        if arm["provider"] == "openai":
            body["reasoning_effort"] = arm["effort"]
        else:
            body["reasoning_effort"] = arm["effort"]
            body.setdefault("max_completion_tokens", 64000)
        return body

    async def _forward(url, headers, body):
        async with httpx.AsyncClient(timeout=1800.0) as cl:
            r = await cl.post(url, json=body, headers=headers)
            try:
                j = r.json()
            except Exception:
                j = {"error": {"message": r.text[:800], "status": r.status_code}}
            if r.status_code >= 400 or "error" in j:
                # surface upstream failures instead of returning an unparseable body
                j.setdefault("_status", r.status_code)
                j["_sent_keys"] = sorted(body.keys())
            return j

    @web.post("/{full_path:path}")
    async def any_post(request: Request, full_path: str):
        """Single catch-all: tolerates any base-url shape litellm builds, and carries the
        episode id in the path (/ep/<id>/...) since the agent cannot send custom headers."""
        t0 = time.time()
        payload = await request.json()
        parts = [x for x in full_path.split("/") if x]
        ep = "unknown"
        if len(parts) >= 2 and parts[0] == "ep":
            ep = parts[1]
        is_responses = full_path.rstrip("/").endswith("responses")

        if is_responses:
            inp = payload.get("input", [])
            msgs = inp if isinstance(inp, list) else [{"role": "user", "content": str(inp)}]
        else:
            msgs = payload.get("messages", [])
        a, x, p_, turn, ep_path = _decide(msgs, ep)
        arm = ARMS[a]
        body = _normalize(dict(payload), arm)

        if is_responses:
            body.pop("reasoning_effort", None)
            body["reasoning"] = {"effort": arm["effort"]}
            url = "https://api.openai.com/v1/responses"
            headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        elif arm["provider"] == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
        else:
            url = "https://api.anthropic.com/v1/chat/completions"
            headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                       "anthropic-version": "2023-06-01"}
        out = await _forward(url, headers, body)
        err = None
        if not out.get("usage"):
            err = json.dumps(out.get("error", out))[:400]
        _log(ep_path, turn, a, x, p_, out.get("usage", {}) or {}, t0, err)
        return out

    return web


@app.function(image=image, volumes={STATE: vol})
def put_policy(W: list, temp: float = 1.0, explore: float = 0.05):
    """Training loop calls this to push updated weights."""
    with open(os.path.join(STATE, "policy.json"), "w") as fh:
        json.dump({"W": W, "temp": temp, "explore": explore}, fh)
    vol.commit()
    return "ok"


@app.function(image=image, volumes={STATE: vol})
def get_episodes(prefix: str = "") -> dict:
    """Training loop pulls per-turn decision logs."""
    out = {}
    d = os.path.join(STATE, "episodes")
    if not os.path.isdir(d):
        return out
    for f in os.listdir(d):
        if f.startswith(prefix) and f.endswith(".jsonl"):
            with open(os.path.join(d, f)) as fh:
                out[f[:-6]] = [json.loads(x) for x in fh if x.strip()]
    return out
