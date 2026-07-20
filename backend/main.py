from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agents

app = FastAPI(title="Kavach AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class TextIn(BaseModel):
    text: str


class SenderIn(BaseModel):
    identifier: str


class CombinedIn(BaseModel):
    text: str = ""
    identifier: str = ""


class ComplaintIn(BaseModel):
    location: str


class LinkIn(BaseModel):
    a: str
    a_type: str
    b: str
    b_type: str
    relation: str


class EvidenceIn(BaseModel):
    payload: dict


@app.post("/api/currency")
async def currency(file: UploadFile = File(...)):
    return agents.check_currency(await file.read())


@app.post("/api/voice")
async def voice(file: UploadFile = File(...)):
    return agents.check_voice(await file.read())


@app.post("/api/scam-text")
def scam_text(body: TextIn):
    return agents.check_scam_text(body.text)


@app.post("/api/sender")
def sender(body: SenderIn):
    return agents.check_sender(body.identifier)


@app.post("/api/check")
def full_check(body: CombinedIn):
    text_result = agents.check_scam_text(body.text) if body.text.strip() else None
    sender_result = agents.check_sender(body.identifier) if body.identifier.strip() else None

    flags = []
    if text_result and text_result.get("verdict") not in ("safe", None):
        pct = round((text_result.get("confidence") or 0) * 100)
        flags.append(f"the message text matches known scam patterns ({pct}% confidence)")
    if sender_result and sender_result.get("verdict") not in ("safe", None):
        flags.append(f"the sender looks suspicious \u2014 {sender_result.get('reason')}")

    if not text_result and not sender_result:
        overall = "unknown"
        summary = "enter a message, a sender, or both to check"
    elif flags:
        overall = "suspicious"
        summary = " and ".join(flags) + "."
    else:
        overall = "safe"
        summary = "no red flags found in the message text or the sender."

    return {
        "overall_verdict": overall,
        "summary": summary,
        "text_result": text_result,
        "sender_result": sender_result,
    }


@app.post("/api/graph/link")
def graph_link(body: LinkIn):
    return agents.add_link(body.a, body.a_type, body.b, body.b_type, body.relation)


@app.get("/api/graph/cluster/{node}")
def graph_cluster(node: str):
    return agents.get_cluster(node)


@app.get("/api/graph")
def graph_all():
    return agents.full_graph()


@app.get("/api/graph/rings")
def graph_rings():
    return agents.detect_rings()


@app.get("/api/geo")
def geo():
    return agents.geo_heatmap()


@app.post("/api/complaint")
def complaint(body: ComplaintIn):
    return agents.log_complaint(body.location)


@app.post("/api/evidence")
def evidence(body: EvidenceIn):
    return agents.generate_evidence(body.payload)