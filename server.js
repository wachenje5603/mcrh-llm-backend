/**
 * MCRH LLM Backend v2.0
 * Full-featured proxy for Groq — powers chat, analysis, creative summaries,
 * search, insights, and record explanations across the MCRH system.
 */

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const crypto = require("crypto");

const app = express();
// Render (and most hosts) sit behind a proxy. Without this, every visitor looks like the same IP
// and the 30-requests-per-minute limit is shared by ALL users.
app.set("trust proxy", 1);
const PORT = process.env.PORT || 3000;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions";

const MODEL_CHAIN = [
  process.env.GROQ_MODEL_PRIMARY    || "openai/gpt-oss-120b",
  process.env.GROQ_MODEL_FALLBACK_1 || "openai/gpt-oss-20b",
  process.env.GROQ_MODEL_FALLBACK_2 || "qwen/qwen3.6-27b",
];

// ---------- Response cache (5 min) ----------
const cache = new Map();
const CACHE_TTL_MS = 5 * 60 * 1000;
const cacheKey = (prefix, data) =>
  prefix + ":" + crypto.createHash("sha256").update(JSON.stringify(data)).digest("hex").slice(0, 16);

// ---------- Rate limiter (per IP) ----------
const rateBuckets = new Map();
const RATE_WINDOW_MS = 60 * 1000;
const RATE_MAX = 30;        // signed-in users: per minute, counted per user
const RATE_MAX_ANON = 10;   // not signed in (login-page assistant): per minute, counted per IP

function rateLimit(req, res, next) {
  const ip = req.ip || req.headers["x-forwarded-for"] || "unknown";
  const id = req.user ? "u:" + req.user.uid : "ip:" + ip;
  const max = req.user ? RATE_MAX : RATE_MAX_ANON;
  const now = Date.now();
  const entry = rateBuckets.get(id) || { count: 0, reset: now + RATE_WINDOW_MS };
  if (now > entry.reset) { entry.count = 0; entry.reset = now + RATE_WINDOW_MS; }
  entry.count++;
  rateBuckets.set(id, entry);
  if (rateBuckets.size > 1000) {
    for (const [k, v] of rateBuckets) if (Date.now() > v.reset + 60000) rateBuckets.delete(k);
  }
  if (entry.count > max) {
    return res.status(429).json({ error: "Rate limit reached. Wait a minute.", retryAfter: Math.ceil((entry.reset - now) / 1000) });
  }
  next();
}

// ---- FIREBASE AUTH START ----
// Pages send the signed-in user's Firebase ID token (Authorization: Bearer <token>).
// It is verified here against Google's public signing keys, so no service account or extra
// package is needed. Set REQUIRE_AUTH=false to switch enforcement off (e.g. while rolling out new pages).
const FIREBASE_PROJECT_ID = process.env.FIREBASE_PROJECT_ID || "moivoibiomed";
const REQUIRE_AUTH = process.env.REQUIRE_AUTH !== "false";
const GOOGLE_CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com";
let certCache = { certs: null, expires: 0 };

async function getGoogleCerts() {
  if (certCache.certs && Date.now() < certCache.expires) return certCache.certs;
  const r = await fetch(GOOGLE_CERTS_URL);
  if (!r.ok) throw new Error("Could not fetch Google signing keys (" + r.status + ")");
  const m = /max-age=(\d+)/.exec(r.headers.get("cache-control") || "");
  certCache = { certs: await r.json(), expires: Date.now() + (m ? parseInt(m[1], 10) : 3600) * 1000 };
  return certCache.certs;
}

const b64urlDecode = (s) => Buffer.from(s.replace(/-/g, "+").replace(/_/g, "/"), "base64");

async function verifyFirebaseToken(idToken, getCerts = getGoogleCerts, projectId = FIREBASE_PROJECT_ID) {
  const parts = String(idToken).split(".");
  if (parts.length !== 3) throw new Error("malformed token");
  let header, payload;
  try {
    header = JSON.parse(b64urlDecode(parts[0]).toString("utf8"));
    payload = JSON.parse(b64urlDecode(parts[1]).toString("utf8"));
  } catch { throw new Error("malformed token"); }
  if (header.alg !== "RS256" || !header.kid) throw new Error("bad algorithm");

  let certs;
  try {
    certs = await getCerts();
    if (!certs[header.kid]) { certCache.expires = 0; certs = await getCerts(); }   // keys rotate: refetch once
  } catch (e) { throw Object.assign(new Error("auth service unavailable"), { status: 503 }); }
  if (!certs[header.kid]) throw new Error("unknown signing key");

  const verifier = crypto.createVerify("RSA-SHA256");
  verifier.update(parts[0] + "." + parts[1]);
  if (!verifier.verify(crypto.createPublicKey(certs[header.kid]), b64urlDecode(parts[2]))) throw new Error("bad signature");

  const now = Math.floor(Date.now() / 1000);
  if (payload.aud !== projectId) throw new Error("wrong project");
  if (payload.iss !== "https://securetoken.google.com/" + projectId) throw new Error("wrong issuer");
  if (!payload.sub || typeof payload.sub !== "string") throw new Error("no subject");
  if (typeof payload.exp !== "number" || payload.exp <= now) throw new Error("expired");
  if (typeof payload.iat !== "number" || payload.iat > now + 300) throw new Error("issued in the future");
  return { uid: payload.sub, email: payload.email || null };
}

// Reads the token if one is sent. A bad token is rejected; no token simply means "anonymous".
async function authenticate(req, res, next) {
  req.user = null;
  const m = /^Bearer\s+(.+)$/i.exec(req.headers.authorization || "");
  if (!m) return next();
  try { req.user = await verifyFirebaseToken(m[1].trim()); }
  catch (err) {
    console.warn("[auth] token rejected:", err.message);
    if (err.status === 503) return res.status(503).json({ error: "Could not verify your login right now. Try again shortly." });
    return res.status(401).json({ error: "Your session has expired or is invalid. Please sign in again." });
  }
  next();
}
function requireAuth(req, res, next) {
  if (req.user || !REQUIRE_AUTH) return next();
  res.status(401).json({ error: "Please sign in to use this feature." });
}
// ---- FIREBASE AUTH END ----

// Optional: set ALLOWED_ORIGINS="https://your-site.web.app,https://your-site.firebaseapp.com" to accept
// requests only from your own pages. Left unset, any site can call this API (previous behaviour).
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "").split(",").map(s => s.trim()).filter(Boolean);
app.use(cors({ origin: ALLOWED_ORIGINS.length ? ALLOWED_ORIGINS : "*", methods: ["GET", "POST", "OPTIONS"], allowedHeaders: ["Content-Type", "Authorization"] }));
app.use(express.json({ limit: "4mb" }));
app.use(authenticate);
app.use(rateLimit);

// ---------- Groq core with retry + fallback ----------
async function callGroq({ messages, systemPrompt, temperature = 0.7, maxTokens = 2000, jsonMode = false }) {
  let lastError = null;
  for (const model of MODEL_CHAIN) {
    const body = {
      model,
      messages: [
        ...(systemPrompt ? [{ role: "system", content: systemPrompt }] : []),
        ...messages,
      ],
      temperature,
      max_tokens: maxTokens,
    };
    if (jsonMode) body.response_format = { type: "json_object" };

    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        console.log(`[AI] ${model} (attempt ${attempt + 1})`);
        const response = await fetch(GROQ_ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${GROQ_API_KEY}` },
          body: JSON.stringify(body),
        });

        if (response.ok) {
          const result = await response.json();
          console.log(`[AI] ✓ ${model} (${result.usage?.total_tokens || "?"} tokens)`);
          return { reply: result.choices?.[0]?.message?.content || "", model: result.model, usage: result.usage };
        }

        if (response.status === 404) { lastError = `Model ${model} unavailable`; break; }
        if (response.status === 401) throw Object.assign(new Error("Invalid API key"), { status: 401 });

        if (response.status === 429) {
          const retryAfter = parseInt(response.headers.get("retry-after") || "5", 10);
          lastError = "Rate limited";
          if (attempt === 0) { await new Promise(r => setTimeout(r, retryAfter * 1000)); continue; }
          break;
        }

        const errText = await response.text();
        lastError = `Groq ${response.status}: ${errText.slice(0, 100)}`;
        break;
      } catch (err) {
        lastError = err.message;
        if (err.status === 401) throw err;
        break;
      }
    }
  }
  throw Object.assign(new Error("All models unavailable"), { detail: lastError, status: 503 });
}

// ---------- Info endpoints ----------
app.get("/", (req, res) => res.json({
  service: "MCRH LLM Backend",
  version: "2.0.0",
  status: "ok",
  authRequired: REQUIRE_AUTH,
  models: MODEL_CHAIN,
  endpoints: ["/health", "/api/chat", "/api/analyze", "/api/creative-summary", "/api/search", "/api/insights", "/api/explain"],
}));

app.get("/health", (req, res) => res.json({ status: "healthy", uptime: process.uptime(), models: MODEL_CHAIN }));

// ---------- 1. General chat (widget, onboarding, help) ----------
app.post("/api/chat", async (req, res) => {
  try {
    const { messages, context } = req.body || {};
    if (!Array.isArray(messages) || !messages.length) return res.status(400).json({ error: "messages required" });
    const anon = !req.user;
    // Only plain user/assistant turns are accepted; nobody can inject a "system" message.
    const clean = messages.slice(anon ? -6 : -10).map(m => ({
      role: m && m.role === "assistant" ? "assistant" : "user",
      content: String((m && m.content) || "").slice(0, 4000),
    }));
    const systemPrompt = buildChatSystemPrompt(anon ? { currentPage: "Login Screen", userRole: "anonymous" } : context);
    const result = await callGroq({ messages: clean, systemPrompt, temperature: 0.7, maxTokens: anon ? 400 : 700 });
    res.json(result);
  } catch (err) { handleError(res, err); }
});

// ---------- 2. Monthly report analysis (structured) ----------
app.post("/api/analyze", requireAuth, async (req, res) => {
  try {
    const { messages, systemPrompt, temperature, maxTokens } = req.body || {};
    if (!Array.isArray(messages) || !messages.length) return res.status(400).json({ error: "messages required" });
    const t = Number(temperature);
    const result = await callGroq({
      messages: messages.slice(-10),
      systemPrompt: typeof systemPrompt === "string" ? systemPrompt.slice(0, 8000) : undefined,
      temperature: Number.isFinite(t) ? Math.min(Math.max(t, 0), 1) : 0.4,
      maxTokens: Math.min(Number(maxTokens) || 1800, 4000),
    });
    res.json(result);
  } catch (err) { handleError(res, err); }
});

// ---------- 3. Creative summary (narrative for PDF handout) ----------
app.post("/api/creative-summary", requireAuth, async (req, res) => {
  try {
    const { monthData, month, hospitalName } = req.body || {};
    if (!monthData) return res.status(400).json({ error: "monthData required" });

    const ck = cacheKey("creative", { monthData, month });
    if (cache.has(ck) && cache.get(ck).expires > Date.now()) {
      return res.json({ ...cache.get(ck).data, cached: true });
    }

    const systemPrompt = buildCreativePrompt(hospitalName);
    const userMessage = `Month: ${month}\n\nData:\n${JSON.stringify(monthData, null, 2)}`;
    const result = await callGroq({
      messages: [{ role: "user", content: userMessage }],
      systemPrompt,
      temperature: 0.75,
      maxTokens: 2500,
    });

    cache.set(ck, { data: result, expires: Date.now() + CACHE_TTL_MS });
    res.json(result);
  } catch (err) { handleError(res, err); }
});

// ---------- 4. Natural-language search ----------
app.post("/api/search", requireAuth, async (req, res) => {
  try {
    const { query, items, itemType } = req.body || {};
    if (!query || !Array.isArray(items)) return res.status(400).json({ error: "query and items required" });

    const systemPrompt = `You are a search assistant for a hospital equipment maintenance system.
Given a natural language query and a list of ${itemType || "items"}, return ONLY a JSON object:
{"matches": ["id1", "id2", ...], "reasoning": "short explanation"}
- Return at most 20 IDs, ordered by relevance
- Match on any field: numbers, names, equipment, departments, statuses, dates, keywords
- Handle fuzzy matches (typos, partial words, synonyms)
- If nothing matches, return {"matches": [], "reasoning": "No relevant matches"}`;

    const userMessage = `Query: "${query}"\n\nItems:\n${JSON.stringify(items.slice(0, 200))}`;
    const result = await callGroq({
      messages: [{ role: "user", content: userMessage }],
      systemPrompt,
      temperature: 0.2,
      maxTokens: 600,
      jsonMode: true,
    });

    let parsed;
    try { parsed = JSON.parse(result.reply); } catch { parsed = { matches: [], reasoning: "AI returned invalid JSON" }; }
    res.json({ ...parsed, model: result.model });
  } catch (err) { handleError(res, err); }
});

// ---------- 5. Analytics insights ----------
app.post("/api/insights", requireAuth, async (req, res) => {
  try {
    const { data, focus } = req.body || {};
    if (!data) return res.status(400).json({ error: "data required" });

    const systemPrompt = `You are a data analyst for a hospital equipment maintenance system.
Analyse the provided data and return 3-5 short, actionable insights.
Format each as: {"insights": [{"title": "...", "detail": "one sentence", "severity": "info|warning|critical", "action": "suggested action"}]}
Return ONLY valid JSON.`;

    const userMessage = `Focus: ${focus || "general overview"}\n\nData:\n${JSON.stringify(data, null, 2)}`;
    const result = await callGroq({
      messages: [{ role: "user", content: userMessage }],
      systemPrompt,
      temperature: 0.5,
      maxTokens: 1200,
      jsonMode: true,
    });

    let parsed;
    try { parsed = JSON.parse(result.reply); } catch { parsed = { insights: [] }; }
    res.json({ ...parsed, model: result.model });
  } catch (err) { handleError(res, err); }
});

// ---------- 6. Explain a record in plain language ----------
app.post("/api/explain", requireAuth, async (req, res) => {
  try {
    const { record, type, audience } = req.body || {};
    if (!record) return res.status(400).json({ error: "record required" });

    const systemPrompt = `You explain maintenance records to ${audience || "a non-technical manager"}.
Use plain language, avoid jargon. 2-3 short paragraphs. Be factual and helpful.`;
    const userMessage = `Explain this ${type || "record"}:\n${JSON.stringify(record, null, 2)}`;
    const result = await callGroq({
      messages: [{ role: "user", content: userMessage }],
      systemPrompt,
      temperature: 0.6,
      maxTokens: 500,
    });
    res.json(result);
  } catch (err) { handleError(res, err); }
});

// ---------- System prompts ----------
function buildChatSystemPrompt(context) {
  let base = `You are a helpful AI assistant embedded in the MCRH (Moi Voi County Referral Hospital) Equipment Maintenance System.

About the system:
- Users submit equipment maintenance requests
- Maintenance staff create job cards to track repairs
- Job statuses: Pending, Completed, Written Off
- Request statuses: No Job, Pending, Completed, Written Off, Received
- Roles: admin, maintenance, requester
- Admins manage users, approvals, and reports
- Maintenance staff see requests for their assigned departments
- Requesters only see their own requests
- Analytics page shows KPIs and charts
- Monthly report generates AI summaries

Your role:
- Help users navigate ("where do I find X?")
- Explain statuses, workflows, and features in plain language
- Answer questions about requests, job cards, analytics
- Never invent data. If asked about specific records you don't have, point to where to look.
- Keep answers short (2-4 sentences). Use **bold** for UI labels.
- If asked for anything outside the system, politely redirect.`;

  if (context) {
    base += `\n\nCurrent context:`;
    if (context.userName) base += `\n- User: ${context.userName}`;
    if (context.userRole) base += `\n- Role: ${context.userRole}`;
    if (context.currentPage) base += `\n- Page: ${context.currentPage}`;
    if (context.totalRequests !== undefined) base += `\n- Total requests: ${context.totalRequests}`;
    if (context.pendingRequests !== undefined) base += `\n- Pending requests: ${context.pendingRequests}`;
    if (context.totalJobcards !== undefined) base += `\n- Total job cards: ${context.totalJobcards}`;
    if (context.departments?.length) base += `\n- Departments: ${context.departments.join(", ")}`;
  }
  return base;
}

function buildCreativePrompt(hospitalName) {
  return `You are a professional writer preparing a monthly maintenance narrative for ${hospitalName || "the hospital"}.

Produce a warm, engaging narrative summary suitable for hospital leadership and staff. Not a dry report — a story of the month.

Structure with these sections using markdown ### headings:

### A Month in Maintenance
2-3 short paragraphs weaving together the work done this month. Mention numbers naturally in flowing prose.

### The Work Behind the Scenes
Highlight the technicians and departments. Call out 2-3 specific jobs that stood out.

### Equipment That Needed Attention
Note any equipment that failed repeatedly or was written off. Be factual, not alarming.

### What's Still in Progress
Honest summary of pending work — what's waiting on parts, approval, or people.

### Looking Ahead
1-2 short paragraphs of forward-looking recommendations in an encouraging tone.

### A Word of Thanks
1 short paragraph acknowledging the team.

Rules:
- Warm, professional, human tone — never corporate or robotic
- Use **bold** sparingly for emphasis
- Keep total length 400-600 words
- Weave numbers into sentences instead of listing them
- Use actual data — never invent figures`;
}

function handleError(res, err) {
  console.error("[API]", err.message, err.detail || "");
  res.status(err.status || 500).json({ error: err.message, detail: err.detail });
}

app.use((req, res) => res.status(404).json({ error: `Unknown route ${req.method} ${req.path}` }));
app.use((err, req, res, next) => { console.error(err); res.status(500).json({ error: "Internal server error" }); });

app.listen(PORT, () => {
  console.log(`\n🧠 MCRH LLM Backend v2.0 on port ${PORT}`);
  console.log(`   Models: ${MODEL_CHAIN.join(" → ")}`);
  console.log(`   Auth: ${REQUIRE_AUTH ? "Firebase ID token required (project " + FIREBASE_PROJECT_ID + ")" : "NOT enforced (REQUIRE_AUTH=false)"}`);
  console.log(`   Endpoints:`);
  console.log(`     GET  /health`);
  console.log(`     POST /api/chat             (widget, help)`);
  console.log(`     POST /api/analyze          (structured analysis)`);
  console.log(`     POST /api/creative-summary (narrative report)`);
  console.log(`     POST /api/search           (natural language search)`);
  console.log(`     POST /api/insights         (analytics insights)`);
  console.log(`     POST /api/explain          (plain-language records)`);
  if (!GROQ_API_KEY) console.warn("\n⚠️  GROQ_API_KEY not set\n");
});