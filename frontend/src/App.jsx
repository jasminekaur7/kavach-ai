import { useState, useEffect, useRef } from 'react'

const API = 'http://localhost:8000'

// Distinguishes different auto-detected fraud rings from each other on the
// graph and in the ring-card list. Was referenced but never defined before —
// fixed here, and colors updated to match the new palette.
const RING_PALETTE = ['#c79a4a', '#c1432b', '#4c9a6a', '#a8927c', '#8a5a7a', '#d4915f']

// Scroll-triggered reveal: wraps any section, fades/slides it in once it
// enters the viewport. Used on the taller panels (Fraud Intel) where content
// actually scrolls; panels themselves already animate on mount via CSS.
function Reveal({ children, className = '', ...rest }) {
  const ref = useRef(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { setVisible(true); obs.disconnect() }
    }, { threshold: 0.15 })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])
  return (
    <div ref={ref} className={`reveal ${visible ? 'reveal-in' : ''} ${className}`} {...rest}>
      {children}
    </div>
  )
}

// Hand-drawn fraud/security line-art (shield, magnifying-glass-over-document,
// fingerprint, padlock, alert-triangle) — real SVG line art, not emoji.
// Purely decorative: fixed, low-opacity, non-interactive, sits behind content.
function BackgroundMotifs() {
  const icons = [
    // shield
    <path key="shield" d="M12 2 20 5.5V11c0 5-3.4 8.7-8 9.9C7.4 19.7 4 16 4 11V5.5L12 2Z" />,
    // magnifying glass over a document
    <g key="glass">
      <rect x="3" y="2" width="12" height="16" rx="1" />
      <line x1="6" y1="6" x2="12" y2="6" /><line x1="6" y1="9" x2="12" y2="9" /><line x1="6" y1="12" x2="10" y2="12" />
      <circle cx="16" cy="16" r="4.5" /><line x1="19.2" y1="19.2" x2="23" y2="23" />
    </g>,
    // fingerprint
    <g key="finger">
      <path d="M12 3a9 9 0 0 0-9 9c0 2 .5 3.5 1 5" /><path d="M12 3a9 9 0 0 1 9 9c0 3-1 5-1 5" />
      <path d="M8 20c-1.5-2-2-4-2-7a6 6 0 0 1 12 0c0 1 0 2-.3 3" />
      <path d="M12 21c-1-1.5-1.5-3-1.5-5a1.5 1.5 0 0 1 3 0c0 1.2-.2 2-.6 3" />
    </g>,
    // padlock
    <g key="lock">
      <rect x="4" y="10" width="16" height="11" rx="1.5" /><path d="M7 10V7a5 5 0 0 1 10 0v3" />
      <circle cx="12" cy="15" r="1.6" /><line x1="12" y1="16.6" x2="12" y2="18.5" />
    </g>,
    // alert triangle
    <g key="alert">
      <path d="M12 3 22 20H2L12 3Z" /><line x1="12" y1="9" x2="12" y2="14" /><circle cx="12" cy="17" r="0.8" />
    </g>,
  ]
  // fixed scatter positions (percent-based), kept off the centered content column
  const placements = [
    { top: '8%', left: '4%', size: 70, rot: -8 },
    { top: '22%', left: '90%', size: 56, rot: 12 },
    { top: '48%', left: '3%', size: 60, rot: 6 },
    { top: '68%', left: '92%', size: 64, rot: -10 },
    { top: '85%', left: '6%', size: 52, rot: 14 },
  ]
  return (
    <div className="bg-motifs" aria-hidden="true">
      {placements.map((p, i) => (
        <svg key={i} className="motif" style={{ top: p.top, left: p.left, width: p.size, height: p.size, transform: `rotate(${p.rot}deg)` }}
          viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="0.9" strokeLinecap="round" strokeLinejoin="round">
          {icons[i % icons.length]}
        </svg>
      ))}
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState('check')
  return (
    <div className="shell">
      <BackgroundMotifs />
      <div className="scan-beam" aria-hidden="true" />
      <header className="topbar">
        <div className="brand">
          <span className="brand-seal">
            <span className="brand-mark">कवच</span>
          </span>
          <span className="brand-text">
            <span className="brand-name">KAVACH<span className="accent"> AI</span></span>
            <span className="brand-tagline">Fraud Defense Console</span>
          </span>
        </div>
        <nav className="tabs">
          {[
            ['check', 'Full Check'],
            ['currency', 'Currency'],
            ['scam', 'Scam Text'],
            ['sender', 'Sender'],
            ['voice', 'Voice'],
            ['intel', 'Fraud Intel'],
          ].map(([id, label]) => (
            <button key={id} className={tab === id ? 'tab active' : 'tab'} onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </nav>
      </header>

      <main className="stage">
        {tab === 'check' && <FullCheckPanel />}
        {tab === 'currency' && <CurrencyPanel />}
        {tab === 'scam' && <ScamTextPanel />}
        {tab === 'sender' && <SenderPanel />}
        {tab === 'voice' && <VoicePanel />}
        {tab === 'intel' && <FraudIntelPanel />}
      </main>
    </div>
  )
}

// ---------------------------------------------------------------------------
function FullCheckPanel() {
  const [text, setText] = useState('')
  const [identifier, setIdentifier] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function check() {
    if (!text.trim() && !identifier.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const r = await fetch(`${API}/api/check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, identifier }),
      })
      if (!r.ok) throw new Error(`Server returned ${r.status}`)
      setResult(await r.json())
    } catch (err) {
      setError('Could not reach the server. Make sure the backend is running, then try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <h1>Full Fraud Check</h1>
      <p className="sub">Paste the message you received, and where it came from (phone number, email, or domain/link). We check both together and give you one verdict.</p>

      <label className="field-label">1. What did the message say?</label>
      <textarea
        className="textarea"
        rows={5}
        placeholder="e.g. This is the CBI, you are under digital arrest, do not disconnect the call..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <label className="field-label">2. Where did it come from?</label>
      <input
        className="textarea"
        style={{ height: 'auto' }}
        placeholder="e.g. ajioin.in, +91 98765xxxxx, or an email address"
        value={identifier}
        onChange={(e) => setIdentifier(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && check()}
      />

      <button className="primary" onClick={check} disabled={loading}>
        {loading ? 'checking…' : 'Check message'}
      </button>

      {error && <div className="verdict bad"><div className="verdict-reason">{error}</div></div>}

      {result && (
        <div className={`verdict ${result.overall_verdict === 'safe' ? 'ok' : 'bad'}`}>
          <div className="verdict-label">{result.overall_verdict?.toUpperCase()}</div>
          <div className="verdict-reason">{result.summary}</div>

          {(result.text_result || result.sender_result) && (
            <div className="verdict-breakdown">
              {result.text_result && (
                <div>
                  Text: {result.text_result.verdict?.toUpperCase()}
                  {typeof result.text_result.confidence === 'number' &&
                    ` (${Math.round(result.text_result.confidence * 100)}% scam probability)`}
                </div>
              )}
              {result.sender_result && (
                <div>
                  Sender: {result.sender_result.verdict?.toUpperCase()}
                  {typeof result.sender_result.confidence === 'number' &&
                    ` (${Math.round(result.sender_result.confidence * 100)}% confidence)`}
                  {' \u2014 '}{result.sender_result.reason}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
function CurrencyPanel() {
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function onFile(e) {
    const file = e.target.files[0]
    if (!file) return
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const r = await fetch(`${API}/api/currency`, { method: 'POST', body: form })
      if (!r.ok) throw new Error(`Server returned ${r.status}`)
      setResult(await r.json())
    } catch (err) {
      setError('Could not reach the server. Make sure the backend is running, then try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <h1>Currency Authenticity Check</h1>
      <p className="sub">Upload a note photo. An OCR pass first scans for counterfeit-indicator text (specimen/novelty/prop-money markings), then a trained visual classifier analyses print quality to flag likely counterfeits.</p>

      <label className="dropzone">
        <input type="file" accept="image/*" onChange={onFile} hidden />
        {preview ? <img src={preview} alt="note preview" className="preview" /> : <span>Click to upload a note image</span>}
      </label>

      {loading && <p className="status">analyzing…</p>}
      {error && <div className="verdict bad"><div className="verdict-reason">{error}</div></div>}

      {result && (
        <div className={`verdict ${result.verdict === 'genuine' ? 'ok' : 'bad'}`}>
          <div className="verdict-label">{result.verdict?.toUpperCase()}</div>
          <div className="verdict-conf">confidence {Math.round(result.confidence * 100)}%</div>
          <div className="verdict-reason">{result.reason}</div>
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
function ScamTextPanel() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function check() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const r = await fetch(`${API}/api/scam-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!r.ok) throw new Error(`Server returned ${r.status}`)
      setResult(await r.json())
    } catch (err) {
      setError('Could not reach the server. Make sure the backend is running, then try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <h1>Scam Message / Call-Transcript Check</h1>
      <p className="sub">Paste a suspicious message or call transcript. The NLP agent flags known digital-arrest / fraud script patterns.</p>

      <textarea
        className="textarea"
        rows={6}
        placeholder="e.g. This is the CBI, you are under digital arrest, do not disconnect the call..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button className="primary" onClick={check} disabled={loading}>
        {loading ? 'checking…' : 'Check for scam'}
      </button>

      {error && <div className="verdict bad"><div className="verdict-reason">{error}</div></div>}

      {result && (
        <div className={`verdict ${result.verdict === 'safe' ? 'ok' : 'bad'}`}>
          <div className="verdict-label">{result.verdict?.toUpperCase()}</div>
          <div className="verdict-conf">scam probability {Math.round(result.confidence * 100)}%</div>
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
function SenderPanel() {
  const [identifier, setIdentifier] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function check() {
    if (!identifier.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const r = await fetch(`${API}/api/sender`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ identifier }),
      })
      if (!r.ok) throw new Error(`Server returned ${r.status}`)
      setResult(await r.json())
    } catch (err) {
      setError('Could not reach the server. Make sure the backend is running, then try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <h1>Sender Reputation Check</h1>
      <p className="sub">Enter a domain, email, or phone number. The reputation agent flags brand impersonation, lookalike domains, disposable emails, and invalid numbers.</p>

      <input
        className="textarea"
        style={{ height: 'auto' }}
        placeholder="e.g. ajioin.in or +91 98765xxxxx"
        value={identifier}
        onChange={(e) => setIdentifier(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && check()}
      />
      <button className="primary" onClick={check} disabled={loading}>
        {loading ? 'checking…' : 'Check sender'}
      </button>

      {error && <div className="verdict bad"><div className="verdict-reason">{error}</div></div>}

      {result && (
        <div className={`verdict ${result.verdict === 'safe' ? 'ok' : 'bad'}`}>
          <div className="verdict-label">{result.verdict?.toUpperCase()}</div>
          <div className="verdict-reason">{result.reason}</div>
        </div>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------
function VoicePanel() {
  const [fileName, setFileName] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function onFile(e) {
    const file = e.target.files[0]
    if (!file) return
    setFileName(file.name)
    setResult(null)
    setError(null)
    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const r = await fetch(`${API}/api/voice`, { method: 'POST', body: form })
      if (!r.ok) throw new Error(`Server returned ${r.status}`)
      setResult(await r.json())
    } catch (err) {
      setError('Could not reach the server. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="panel">
      <h1>Voice Spoof / AI-Clone Check</h1>
      <p className="sub">Upload a call recording (WAV or FLAC). The audio agent analyses pitch jitter, spectral flatness, and MFCC features via a trained classifier to flag likely synthetic or AI-cloned voices.</p>

      <label className="dropzone" style={{ height: 120 }}>
        <input type="file" accept="audio/wav,audio/x-wav,audio/flac,.wav,.flac" onChange={onFile} hidden />
        <span>{fileName || 'Click to upload a call recording (.wav / .flac)'}</span>
      </label>

      {loading && <p className="status">analyzing…</p>}
      {error && <div className="verdict bad"><div className="verdict-reason">{error}</div></div>}

      {result && (
        <div className={`verdict ${result.verdict === 'likely_genuine' ? 'ok' : 'bad'}`}>
          <div className="verdict-label">{result.verdict?.replace('_', ' ').toUpperCase()}</div>
          <div className="verdict-conf">confidence {Math.round((result.confidence || 0) * 100)}%</div>
          <div className="verdict-reason">{result.reason}</div>
        </div>
      )}
    </section>
  )
}

function GraphSVG({ graph, rings = [] }) {
  const w = 720, h = 420, r = Math.min(w, h) / 2 - 60
  const cx = w / 2, cy = h / 2
  const n = graph.nodes.length || 1
  const pos = Object.fromEntries(
    graph.nodes.map((node, i) => {
      const angle = (2 * Math.PI * i) / n
      return [node.id, { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) }]
    })
  )
  const colors = {
    PhoneNumber: '#c96a4a', BankAccount: '#c79a4a', CounterfeitNote: '#c1432b',
    Location: '#8a5a7a', FraudRing: '#c1432b', Person: '#a8927c',
  }
  // map each node id -> ring color, so nodes in the same detected ring share an outline
  const ringColor = {}
  rings.forEach((ring, i) => {
    ring.members.forEach((m) => { ringColor[m.id] = RING_PALETTE[i % RING_PALETTE.length] })
  })
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="graph-svg">
      {graph.edges.map((e, i) => (
        <line key={i} x1={pos[e.source]?.x} y1={pos[e.source]?.y} x2={pos[e.target]?.x} y2={pos[e.target]?.y} className="edge" />
      ))}
      {graph.nodes.map((node) => (
        <g key={node.id} transform={`translate(${pos[node.id].x}, ${pos[node.id].y})`}>
          <circle
            r="10"
            fill={colors[node.type] || '#999'}
            stroke={ringColor[node.id] || 'none'}
            strokeWidth={ringColor[node.id] ? 3 : 0}
          />
          <text x="14" y="4" className="node-label">{node.id}</text>
        </g>
      ))}
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Threshold-based risk color, matching the legend shown on the map:
// grey = no complaints yet, green = 1-5, yellow = 6-10, red = 11+.
function riskColor(complaintCount) {
  if (complaintCount === 0) return '#6b5f52'
  if (complaintCount <= 5) return '#4c9a6a'
  if (complaintCount <= 10) return '#c79a4a'
  return '#c1432b'
}

function FraudIntelPanel() {
  const mapRef = useRef(null)
  const leafletMap = useRef(null)
  const [error, setError] = useState(null)
  const [graph, setGraph] = useState({ nodes: [], edges: [] })
  const [rings, setRings] = useState([])
  const [points, setPoints] = useState([])
  const [form, setForm] = useState({ a: '', a_type: 'PhoneNumber', b: '', b_type: 'BankAccount', relation: 'used_in' })
  const [complaintLoc, setComplaintLoc] = useState('')
  const [evidenceByRing, setEvidenceByRing] = useState({})

  async function exportEvidence(ring, i) {
    try {
      const r = await fetch(`${API}/api/evidence`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ payload: { case_type: 'fraud_ring', ring_id: ring.ring_id, kingpin: ring.kingpin, members: ring.members } }),
      })
      if (!r.ok) throw new Error(`Server returned ${r.status}`)
      const { hash } = await r.json()
      setEvidenceByRing((prev) => ({ ...prev, [i]: hash }))
    } catch (err) {
      setError('Could not generate the evidence packet. Make sure the backend is running, then try again.')
    }
  }

  function downloadEvidence(hash) {
    fetch(`${API}/api/evidence/${hash}`)
      .then((r) => r.json())
      .then((record) => {
        const blob = new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `kavach-evidence-${hash.slice(0, 16)}.json`
        a.click()
        URL.revokeObjectURL(url)
      })
      .catch(() => setError('Could not download the evidence file. Make sure the backend is running.'))
  }

  async function refresh() {
    try {
      const [graphRes, ringsRes, geoRes] = await Promise.all([
        fetch(`${API}/api/graph`), fetch(`${API}/api/graph/rings`), fetch(`${API}/api/geo`),
      ])
      if (!graphRes.ok || !ringsRes.ok || !geoRes.ok) throw new Error('server error')
      setGraph(await graphRes.json())
      setRings((await ringsRes.json()).rings || [])
      setPoints((await geoRes.json()).points || [])
      setError(null)
    } catch (err) {
      setError('Could not reach the server. Make sure the backend is running.')
    }
  }

  useEffect(() => { refresh() }, [])

  async function addLink() {
    if (!form.a || !form.b) return
    try {
      const r = await fetch(`${API}/api/graph/link`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!r.ok) throw new Error(`Server returned ${r.status}`)
      setForm({ ...form, a: '', b: '' })
      refresh()
    } catch (err) {
      setError('Could not save the link. Make sure the backend is running, then try again.')
    }
  }

  async function logComplaint() {
    if (!complaintLoc.trim()) return
    try {
      const r = await fetch(`${API}/api/complaint`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ location: complaintLoc }),
      })
      const data = await r.json()
      if (data.error) { setError(data.error); return }
      setComplaintLoc('')
      refresh()
    } catch (err) {
      setError('Could not reach the server. Make sure the backend is running.')
    }
  }

  useEffect(() => {
    if (!window.L || !mapRef.current) return
    if (!leafletMap.current) {
      leafletMap.current = window.L.map(mapRef.current, { zoomControl: true }).setView([22.5, 80], 5)
      window.L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
      }).addTo(leafletMap.current)
    }
    const map = leafletMap.current
    map.eachLayer((layer) => { if (layer instanceof window.L.CircleMarker) map.removeLayer(layer) })
    points.forEach((p) => {
      const color = riskColor(p.complaint_count)
      window.L.circleMarker([p.lat, p.lon], {
        radius: 12,
        color,
        fillColor: color,
        fillOpacity: 0.75,
      })
        .bindPopup(
          `<b>${p.name}</b><br/>${p.complaint_count} complaint${p.complaint_count === 1 ? '' : 's'} filed<br/>${p.linked_entities} linked entit${p.linked_entities === 1 ? 'y' : 'ies'}`
        )
        .addTo(map)
    })
  }, [points])

  const highAlert = points.filter((p) => p.complaint_count > 10)

  return (
    <section className="panel wide">
      <h1>Fraud Intelligence — Graph &amp; Hotspot Map</h1>
      <p className="sub">Link physical evidence to digital entities, and log citizen complaints by location — both views below update together from the same fraud graph.</p>

      <label className="field-label">Link two entities</label>
      <div className="link-form">
        <input placeholder="Entity A (e.g. 98765xxxxx)" value={form.a} onChange={(e) => setForm({ ...form, a: e.target.value })} />
        <select value={form.a_type} onChange={(e) => setForm({ ...form, a_type: e.target.value })}>
          {['PhoneNumber', 'BankAccount', 'CounterfeitNote', 'Location', 'FraudRing', 'Person'].map((t) => <option key={t}>{t}</option>)}
        </select>
        <span className="arrow">→</span>
        <input placeholder="Entity B (e.g. ACC123)" value={form.b} onChange={(e) => setForm({ ...form, b: e.target.value })} />
        <select value={form.b_type} onChange={(e) => setForm({ ...form, b_type: e.target.value })}>
          {['BankAccount', 'PhoneNumber', 'CounterfeitNote', 'Location', 'FraudRing', 'Person'].map((t) => <option key={t}>{t}</option>)}
        </select>
        <input placeholder="relation (e.g. linked_to)" value={form.relation} onChange={(e) => setForm({ ...form, relation: e.target.value })} />
        <button className="primary" onClick={addLink}>Add link</button>
      </div>

      <label className="field-label">Log a citizen complaint by location</label>
      <div className="link-form">
        <input
          placeholder="e.g. Jamtara"
          value={complaintLoc}
          onChange={(e) => setComplaintLoc(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && logComplaint()}
        />
        <button className="primary" onClick={logComplaint}>Log complaint</button>
      </div>

      {error && <div className="verdict bad"><div className="verdict-reason">{error}</div></div>}

      {highAlert.length > 0 && (
        <div className="verdict bad">
          <div className="verdict-label">⚠ HIGH ALERT</div>
          <div className="verdict-reason">
            {highAlert.map((p) => p.name).join(', ')} {highAlert.length === 1 ? 'has' : 'have'} more than 10 complaints filed — recommend immediate patrol/investigation priority.
          </div>
        </div>
      )}

      <Reveal className="intel-grid">
        <div>
          <h2 className="section-label">Fraud Fusion Graph</h2>
          <GraphSVG graph={graph} rings={rings} />
        </div>
        <div>
          <h2 className="section-label">Hotspot Map</h2>
          <div ref={mapRef} style={{ height: 420, borderRadius: 12, overflow: 'hidden', border: '1px solid var(--line)' }} />
          <div className="legend">
            <span><i style={{ background: '#6b5f52' }} /> 0 complaints</span>
            <span><i style={{ background: '#4c9a6a' }} /> 1–5</span>
            <span><i style={{ background: '#c79a4a' }} /> 6–10</span>
            <span><i style={{ background: '#c1432b' }} /> 11+ (high alert)</span>
          </div>
        </div>
      </Reveal>

      {rings.length > 0 && (
        <Reveal className="ring-list-section" style={{ marginTop: 24 }}>
          <h2 className="section-label">Auto-Detected Fraud Rings ({rings.length})</h2>
          <div className="ring-list">
            {rings.map((ring, i) => (
              <div key={ring.ring_id} className="ring-card" style={{ borderColor: RING_PALETTE[i % RING_PALETTE.length], animationDelay: `${i * 0.08}s` }}>
                <div className="ring-title" style={{ color: RING_PALETTE[i % RING_PALETTE.length] }}>
                  Ring {i + 1} — {ring.size} linked entities
                </div>
                <div className="ring-detail">Kingpin: <b>{ring.kingpin}</b></div>
                <div className="ring-detail">Members: {ring.members.map((m) => m.id).join(', ')}</div>
                {evidenceByRing[i] ? (
                  <div className="ring-detail" style={{ marginTop: 8 }}>
                    <div style={{ fontFamily: 'monospace', fontSize: '0.8em', opacity: 0.8 }}>
                      hash: {evidenceByRing[i].slice(0, 16)}…
                    </div>
                    <button className="primary" style={{ marginTop: 4, padding: '4px 10px', fontSize: '0.85em' }} onClick={() => downloadEvidence(evidenceByRing[i])}>
                      Download evidence packet
                    </button>
                  </div>
                ) : (
                  <button className="primary" style={{ marginTop: 8, padding: '4px 10px', fontSize: '0.85em' }} onClick={() => exportEvidence(ring, i)}>
                    Export as evidence
                  </button>
                )}
              </div>
            ))}
          </div>
        </Reveal>
      )}


      {points.length === 0 && graph.nodes.length === 0 && (
        <p className="status">nothing linked yet — add an entity link or log a complaint above to populate both views.</p>
      )}
    </section>
  )
}