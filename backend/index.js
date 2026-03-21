import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { OpenAI } from 'openai';
import dotenv from 'dotenv';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const pdf = require('pdf-parse');
import { v4 as uuidv4 } from 'uuid';
import morgan from 'morgan';
import axios from 'axios';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

// ---------------------------
// Environment & OpenAI config
// ---------------------------
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.join(__dirname, '.env') });

const api_key = process.env.OPENAI_API_KEY;
if (!api_key) {
  console.error("OPENAI_API_KEY not found in environment variables. Please check your .env file.");
  process.exit(1);
}

const client = new OpenAI({ apiKey: api_key });

const GEN_MODEL = "gpt-4o";
const EMBED_MODEL = "text-embedding-3-small";

// ---------------------------
// Utility: robust JSON parsing
// ---------------------------
function parseStrictJson(raw) {
  let s = raw.trim();

  // Strip markdown code fences if present
  if (s.startsWith("```")) {
    if (s.toLowerCase().startsWith("```json")) {
      s = s.substring(7);
    } else {
      s = s.substring(3);
    }
    if (s.endsWith("```")) {
      s = s.substring(0, s.length - 3);
    }
  }

  s = s.trim();

  // Take content between first { and last }
  const first = s.indexOf("{");
  const last = s.lastIndexOf("}");
  if (first !== -1 && last !== -1 && last > first) {
    s = s.substring(first, last + 1);
  }

  try {
    return JSON.parse(s);
  } catch (e) {
    return null;
  }
}

async function callOpenAIChat(prompt, jsonMode = false) {
  try {
    const response = await client.chat.completions.create({
      model: GEN_MODEL,
      messages: [
        { role: "system", content: "You are a helpful ESG analyst AI." },
        { role: "user", content: prompt }
      ],
      response_format: jsonMode ? { type: "json_object" } : undefined
    });
    return response.choices[0].message.content;
  } catch (e) {
    console.error("OpenAI API error:", e);
    throw new Error(`OpenAI API error: ${e.message}`);
  }
}

// ---------------------------
// Vector Math Utilities
// ---------------------------
function l2Normalize(vec) {
  const norm = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0)) || 1.0;
  return vec.map(val => val / norm);
}

function dotProduct(vec1, vec2) {
  return vec1.reduce((sum, val, idx) => sum + val * vec2[idx], 0);
}

// ---------------------------
// ESG Index (Replacement for FAISS)
// ---------------------------
class ESGIndex {
  constructor() {
    this.chunks = [];
    this.reports = {};
  }

  async embedText(text) {
    try {
      const cleanText = text.replace(/\n/g, " ");
      const response = await client.embeddings.create({
        input: [cleanText],
        model: EMBED_MODEL
      });
      const emb = response.data[0].embedding;
      return l2Normalize(emb);
    } catch (e) {
      console.error("Embedding error:", e);
      return new Array(1536).fill(0.0);
    }
  }

  async embedTexts(texts) {
    try {
      const cleanTexts = texts.map(t => t.replace(/\n/g, " "));
      const response = await client.embeddings.create({
        input: cleanTexts,
        model: EMBED_MODEL
      });
      return response.data.map(d => l2Normalize(d.embedding));
    } catch (e) {
      console.error("Batch embedding error:", e);
      return texts.map(() => new Array(1536).fill(0.0));
    }
  }

  chunkPages(pages, chunkChars = 1600, overlap = 150, maxChunks = 800) {
    const chunks = [];
    for (let i = 0; i < pages.length; i++) {
      const text = (pages[i] || "").trim();
      if (!text) continue;
      const pageIdx = i + 1;
      let start = 0;
      const n = text.length;
      while (start < n) {
        if (chunks.length >= maxChunks) return chunks;
        const end = Math.min(start + chunkChars, n);
        const chunk = text.substring(start, end).trim();
        if (chunk.length > 50) {
          chunks.push({ text: chunk, page: pageIdx });
        }
        if (end === n) break;
        start = end - overlap;
        if (start < 0) start = 0;
      }
    }
    return chunks;
  }

  async addReport(name, content, pagesText = null) {
    const reportId = `rep_${Object.keys(this.reports).length + 1}_${Date.now()}`;
    const pages = pagesText !== null ? pagesText : [content];

    const chunksWithPages = this.chunkPages(pages);
    if (!chunksWithPages.length) {
      throw new Error("No text chunks extracted from report");
    }

    const texts = chunksWithPages.map(c => c.text);
    const embeddings = await this.embedTexts(texts);

    const startIndex = this.chunks.length;
    chunksWithPages.forEach((item, idx) => {
      this.chunks.push({
        id: startIndex + idx,
        text: item.text,
        reportId: reportId,
        reportName: name,
        page: item.page,
        embedding: embeddings[idx]
      });
    });

    const reportMeta = {
      id: reportId,
      name: name,
      pages: pages.length,
      uploadedAt: new Date().toISOString()
    };
    this.reports[reportId] = reportMeta;
    return reportMeta;
  }

  async search(query, topK = 8, reportIds = null) {
    if (!this.chunks.length) return [];
    
    const queryEmbedding = await this.embedText(query);
    const allowedReports = reportIds ? new Set(reportIds) : null;

    const scored = this.chunks
      .filter(chunk => !allowedReports || allowedReports.has(chunk.reportId))
      .map(chunk => ({
        score: dotProduct(queryEmbedding, chunk.embedding),
        text: chunk.text,
        reportId: chunk.reportId,
        reportName: chunk.reportName,
        page: chunk.page,
        id: chunk.id
      }))
      .sort((a, b) => b.score - a.score);

    const results = [];
    const seenIds = new Set();
    for (const item of scored) {
      if (seenIds.has(item.id)) continue;
      seenIds.add(item.id);
      results.push(item);
      if (results.length >= topK) break;
    }
    return results;
  }

  listReports() {
    return Object.values(this.reports);
  }

  previewText(reportId, maxChars = 1000) {
    const reportChunks = this.chunks.filter(c => c.reportId === reportId);
    if (!reportChunks.length) return "";
    const text = reportChunks.map(c => c.text).join(" ");
    return text.substring(0, maxChars);
  }
}

// ---------------------------
// Evidence Index
// ---------------------------
class EvidenceIndex {
  constructor() {
    this.docs = {};
    this.docChunks = [];
  }

  async embedTexts(texts) {
    try {
      const cleanTexts = texts.map(t => t.replace(/\n/g, " "));
      const response = await client.embeddings.create({
        input: cleanTexts,
        model: EMBED_MODEL
      });
      return response.data.map(d => l2Normalize(d.embedding));
    } catch (e) {
      return texts.map(() => new Array(1536).fill(0.0));
    }
  }

  chunkPages(pages, chunkSize = 900, overlap = 120) {
    const chunks = [];
    for (let i = 0; i < pages.length; i++) {
       const text = (pages[i] || "").trim();
       if (!text) continue;
       const pageNo = i + 1;
       let start = 0;
       const n = text.length;
       while (start < n) {
          const end = Math.min(n, start + chunkSize);
          const chunk = text.substring(start, end).trim();
          if (chunk) {
            chunks.push({ text: chunk, page: pageNo });
          }
          if (end === n) break;
          start = end - overlap;
       }
    }
    return chunks;
  }

  async addDoc(title, sourceType, content, pagesText = null, url = null) {
     const docId = `ev_${Object.keys(this.docs).length + 1}_${Date.now()}`;
     const pages = pagesText !== null ? pagesText : [content];
     const chunksWithPages = this.chunkPages(pages);
     if (!chunksWithPages.length) {
       throw new Error("No text chunks extracted from evidence");
     }

     const texts = chunksWithPages.map(c => c.text);
     const embeddings = await this.embedTexts(texts);
     
     const baseIdx = this.docChunks.length;
     chunksWithPages.forEach((item, j) => {
        this.docChunks.push({
          docId,
          chunkIndex: baseIdx + j,
          page: item.page,
          text: item.text,
          sourceType,
          title,
          url,
          embedding: embeddings[j]
        });
     });

     this.docs[docId] = {
       docId,
       title,
       sourceType,
       url,
       chunks: chunksWithPages.length,
       createdAt: new Date().toISOString()
     };
     return this.docs[docId];
  }

  async search(query, topK = 6) {
    if (!this.docChunks.length) return [];
    
    try {
      const response = await client.embeddings.create({
        input: [query.replace(/\n/g, " ")],
        model: EMBED_MODEL
      });
      const queryEmbedding = l2Normalize(response.data[0].embedding);
      
      const scored = this.docChunks.map(chunk => ({
        ...chunk,
        score: dotProduct(queryEmbedding, chunk.embedding)
      })).sort((a, b) => b.score - a.score);

      return scored.slice(0, topK).map(item => {
        const { embedding, ...rest } = item;
        return rest;
      });
    } catch (e) {
      console.error("Evidence search error:", e);
      return [];
    }
  }
}

// ---------------------------
// Helpers
// ---------------------------
function claimCandidateSnippets(report, esgIndex, maxSnippets = 24) {
  const keywords = [
    "net zero", "carbon neutral", "reduced", "reduction", "target", "goal",
    "scope 1", "scope 2", "scope 3", "emissions", "renewable", "energy",
    "diversity", "inclusion", "gender", "pay gap", "living wage",
    "human rights", "child labor", "supplier", "audit",
    "compliance", "certified", "iso", "tcfd", "sasb", "gri",
    "%", "ton", "tco2", "tCO2e", "kWh"
  ];
  const reportId = report.id;
  const chunks = esgIndex.chunks.filter(c => c.reportId === reportId);
  
  const scored = chunks.map(c => {
    const t = (c.text || "").toLowerCase();
    let score = 0;
    keywords.forEach(kw => {
      if (t.includes(kw)) score += 1;
    });
    // Boost numeric claims
    if (/\b\d{1,3}(?:\.\d+)?\s?%\b/.test(t)) score += 2;
    if (/\b\d{2,4}\b/.test(t)) score += 1;
    return { score, chunk: c };
  }).filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score);

  let picked = scored.slice(0, maxSnippets).map(item => item.chunk);
  if (!picked.length) picked = chunks.slice(0, maxSnippets);

  return picked.map(c => `[page ${c.page}] ${c.text}`).join("\n\n");
}

function calculateFluffRatio(context) {
  const fluffWords = ["striving", "conscious", "journey", "committed", "commitment", "revolutionary", "vision", "philosophy", "belief", "beliefs", "hope", "aim", "aiming", "dedicated", "passion", "passionate", "pioneer", "pioneering", "innovative", "leading", "world-class", "transformative", "believe"];
  const text = context.toLowerCase();
  
  let fluffCount = 0;
  fluffWords.forEach(word => {
    const matches = text.match(new RegExp(word, 'g'));
    if (matches) fluffCount += matches.length;
  });

  const factMatches = text.match(/\b\d{1,3}(?:\.\d+)?\s?%|\b\$\d+|\b\d{2,4}\b|tco2e|mwh|kwh|tonnes/g);
  const factCount = factMatches ? factMatches.length : 0;
  
  const total = fluffCount + factCount;
  if (total === 0) return "0% Fluff / 0% Fact";
  
  const fluffPct = Math.round((fluffCount / total) * 100);
  const factPct = 100 - fluffPct;
  return `${fluffPct}% Fluff / ${factPct}% Fact`;
}

// ---------------------------
// Global Instances
// ---------------------------
const esgIndex = new ESGIndex();
const evidenceIndex = new EvidenceIndex();

// ---------------------------
// App Setup
// ---------------------------
const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(morgan('dev'));

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 } // 50MB
});

// ---------------------------
// Endpoints
// ---------------------------
app.get("/", (req, res) => {
  res.json({ message: "TrueScope API running (Node.js/OpenAI)", docs: "/api/health" });
});

app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    model: GEN_MODEL,
    chunks: esgIndex.chunks.length,
    reports: Object.keys(esgIndex.reports).length
  });
});

async function fetchLiveNews(companyName) {
  // Simple replacement: Use some news API or a dummy search result if needed.
  // Since we don't have a direct Node port of DDGS, we'll try a simplified web search if possible
  // or return an empty string if it's too complex for this demo.
  try {
     // You could use NewsAPI.org or similar here. 
     // For this exact functionality, we'll try a public search endpoint if available.
     return ""; 
  } catch (e) {
    return "";
  }
}

app.post("/api/reports", upload.array('files'), async (req, res) => {
  const isJson = req.is('application/json');
  let reports_to_process = [];

  if (isJson) {
      if (req.body.text) {
          reports_to_process = [{ 
              name: req.body.name || "Extracted Report", 
              text: req.body.text, 
              pagesText: [req.body.text] 
          }];
      }
  } else {
      if (!req.files || req.files.length === 0) {
          return res.status(400).json({ detail: "No files uploaded" });
      }
      // Handle files (as it was)
      for (const file of req.files) {
          const name = file.originalname;
          const contentBuffer = file.buffer;
          let pages = [];
          let fullText = "";

          if (name.toLowerCase().endsWith(".pdf")) {
              const data = await pdf(contentBuffer);
              fullText = data.text;
              pages = [fullText];
          } else {
              fullText = contentBuffer.toString('utf-8');
              pages = [fullText];
          }
          reports_to_process.push({ name, text: fullText, pagesText: pages });
      }
  }

  if (reports_to_process.length === 0) {
      return res.status(400).json({ detail: "No reports or text content found" });
  }

  const processedReports = [];
  for (const item of reports_to_process) {
      try {
          const reportMeta = await esgIndex.addReport(item.name, item.text, item.pagesText);
          processedReports.push(reportMeta);
      } catch (e) {
          console.error(`Failed to process ${item.name}:`, e);
      }
  }

  res.json({ reports: esgIndex.listReports() });
});

app.post("/api/sample-report", (req, res) => {
  const sampleText = "ESG SUSTAINABILITY AUDIT 2024 - TESTCORP INTERNATIONAL. TestCorp is a global manufacturing leader. Scope 1: 12500 tCO2e. Scope 2: 8200 tCO2e. Scope 3: 455000 tCO2e. Total Energy: 156000 MWh. Water: 98500 m3. Waste: 4300 tonnes. Employees: 12450. Board Diversity: 33% female representation. The report follows GRI Standards, SASB Industrial Machinery, and IFRS S1/S2 requirements. We support SDG 7, 12, and 13. Claim 1: 'Net Zero by 2040' supported by $50M solar investment. Claim 2: '100% eco-friendly packaging'. Contradiction: Inner plastic lining is non-recyclable.";
  esgIndex.addReport("Sample_ESG_Report_Comprehensive.txt", sampleText, [sampleText]).then(meta => {
    res.json({ reports: esgIndex.listReports() });
  });
});

app.get("/api/reports", (req, res) => {
  res.json({ reports: esgIndex.listReports() });
});

app.get("/api/reports/:reportId/preview", (req, res) => {
  const { reportId } = req.params;
  if (!esgIndex.reports[reportId]) {
    return res.status(404).json({ detail: "Report not found" });
  }
  const preview = esgIndex.previewText(reportId, 1000);
  res.json({ reportId, previewText: preview });
});

app.post("/api/query", async (req, res) => {
  const { question, report_ids, top_k = 8 } = req.body;
  if (!question || !question.trim()) {
    return res.status(400).json({ detail: "Question is empty" });
  }

  const searchResults = await esgIndex.search(question, top_k, report_ids);
  if (!searchResults.length) {
    return res.json({
      answer: "I couldn’t find relevant ESG context for that question in the uploaded reports.",
      citations: []
    });
  }

  const contextLines = searchResults.map((item, i) => 
    `[${i+1}] (Report: ${item.reportName}, page ${item.page})\n${item.text}`
  );
  const citations = searchResults.map((item, i) => ({
    id: `c${i+1}`,
    report_id: item.reportId,
    report_name: item.reportName,
    page: item.page,
    snippet: item.text.substring(0, 400)
  }));

  const contextBlock = contextLines.join("\n\n");
  const prompt = `You are an ESG analyst assistant. Answer the user's question using ONLY the provided context.
Be concise and specific. If the context does not contain the answer, say so.

CONTEXT:
${contextBlock}

QUESTION:
${question}
`;

  const answer = await callOpenAIChat(prompt);
  res.json({ answer, citations });
});

app.post("/api/summary", async (req, res) => {
  const { report_id } = req.body;
  const report = esgIndex.reports[report_id];
  if (!report) return res.status(404).json({ detail: "Report not found" });

  const context = claimCandidateSnippets(report, esgIndex, 40);
  const prompt = `You are an ESG analyst. Based only on the context below, write a professional, 1-page ESG executive summary of this company's sustainability performance.

Structure the summary as markdown with the following sections:
1. Overview
2. Key environmental metrics (CO₂, energy, water, waste)
3. Social initiatives
4. Governance & risk management
5. Strengths
6. Gaps and risks

Be factual and avoid making up data. If a metric is not disclosed, say so.

Context:
${context}

Now write the markdown summary.`;

  const summary_md = await callOpenAIChat(prompt);
  res.json({ report_id, summary_md });
});

app.post("/api/metrics", async (req, res) => {
  const { report_id } = req.body;
  const report = esgIndex.reports[report_id];
  if (!report) return res.status(404).json({ detail: "Report not found" });

  const context = claimCandidateSnippets(report, esgIndex, 40);
  const prompt = `You are an ESG data extraction assistant. Based ONLY on the context below, extract key ESG metrics and return them as strict JSON. Do not include commentary.

Use this exact JSON structure:
{
  "emissions": {
    "scope1_tco2e": number | null,
    "scope2_tco2e": number | null,
    "scope3_tco2e": number | null
  },
  "energy": {
    "total_mwh": number | null
  },
  "water": {
    "withdrawals_m3": number | null
  },
  "waste": {
    "total_tonnes": number | null
  },
  "social": {
    "employees_total": number | null
  },
  "governance": {
    "board_female_pct": number | null
  }
}

If a value is not clearly stated, use null. Do NOT add any extra top-level fields. Respond with JSON only, no extra text.

Context:
${context}`;

  const raw = await callOpenAIChat(prompt, true);
  const metrics = parseStrictJson(raw) || {};
  res.json({ report_id, metrics });
});

app.post("/api/compliance", async (req, res) => {
  const { report_id } = req.body;
  const report = esgIndex.reports[report_id];
  if (!report) return res.status(404).json({ detail: "Report not found" });

  const context = claimCandidateSnippets(report, esgIndex, 40);
  const prompt = `You are an ESG reporting compliance assistant. Based ONLY on the context below, assess whether the report discusses or references each of the following:
- SDGs (Sustainable Development Goals)
- GRI (Global Reporting Initiative)
- SASB (Sustainability Accounting Standards Board)
- IFRS S1
- IFRS S2
- TCFD (Task Force on Climate-related Financial Disclosures)

Return a strict JSON object with this structure:
{
  "sdgs": {"covered": boolean, "notes": string},
  "gri": {"covered": boolean, "notes": string},
  "sasb": {"covered": boolean, "notes": string},
  "tcfd": {"covered": boolean, "notes": string},
  "ifrs_s1": {"covered": boolean, "notes": string},
  "ifrs_s2": {"covered": boolean, "notes": string},
  "compliance_score": number (0-100)
}

If you are not sure, set covered to false and explain briefly in notes. Respond with JSON only, no extra text.

Context:
${context}`;

  const raw = await callOpenAIChat(prompt, true);
  const compliance = parseStrictJson(raw) || {};
  res.json({ report_id, compliance });
});

// --- NEW ENTERPRISE ENDPOINTS ---

app.post("/api/frameworks", async (req, res) => {
  const { report_id } = req.body;
  const report = esgIndex.reports[report_id];
  if (!report) return res.status(404).json({ detail: "Report not found" });

  try {
    console.log(`Auditing Frameworks for report: ${report_id}`);
    const context = claimCandidateSnippets(report, esgIndex, 50);
    const prompt = `You are a Senior ESG Auditor. Perform a deep alignment check against GRI, TCFD, and SASB. Return a strict JSON:
    { "gri_alignment": { "score": number, "findings": string, "missing": string[] }, "tcfd_alignment": { "score": number, "findings": string, "missing": string[] }, "sasb_alignment": { "score": number, "findings": string, "missing": string[] }, "overall_audit_summary": string }
    Context: ${context}`;

    const raw = await callOpenAIChat(prompt, true);
    const data = parseStrictJson(raw) || {};
    res.json({ report_id, ...data });
  } catch (err) {
    console.error("Framework Audit Error:", err);
    res.status(500).json({ detail: `Framework Audit Error: ${err.message}` });
  }
});

app.post("/api/risk/predict", async (req, res) => {
  const { report_id } = req.body;
  const report = esgIndex.reports[report_id];
  if (!report) return res.status(404).json({ detail: "Report not found" });

  try {
    console.log(`Predicting Risks for report: ${report_id}`);
    const context = claimCandidateSnippets(report, esgIndex, 50);
    const prompt = `You are an ESG Risk Forecaster. Analyze the provided context to predict potential future controversies or regulatory issues.
    Return a strict JSON: { "predicted_risks": [ { "risk_type": string, "probability": "Low"|"Medium"|"High", "justification": string, "regulatory_impact": string } ], "controversy_likelihood": number (0-100), "early_warning_signals": string[] }
    Context: ${context}`;

    const raw = await callOpenAIChat(prompt, true);
    const data = parseStrictJson(raw) || {};
    res.json({ report_id, ...data });
  } catch (err) {
    console.error("Risk Prediction Error:", err);
    res.status(500).json({ detail: `Risk Prediction Error: ${err.message}` });
  }
});

app.post("/api/carbon/analysis", async (req, res) => {
  const { report_id } = req.body;
  const report = esgIndex.reports[report_id];
  if (!report) return res.status(404).json({ detail: "Report not found" });

  try {
    console.log(`Analyzing Carbon for report: ${report_id}`);
    const context = claimCandidateSnippets(report, esgIndex, 50);
    const prompt = `You are a Carbon Accounting Expert. Perform a deep analysis of the carbon footprint. Extract Scope 1, 2, and 3 data.
    Return a strict JSON: { "breakdown": { "scope1": { "value": number|null, "unit": string, "trend": string }, "scope2": { "value": number|null, "unit": string, "trend": string }, "scope3": { "value": number|null, "unit": string, "trend": string } }, "insights": { "intensity_check": string, "net_zero_viability": string, "data_gaps": string[] }, "paris_alignment_score": number (0-100) }
    Context: ${context}`;

    const raw = await callOpenAIChat(prompt, true);
    const data = parseStrictJson(raw) || {};
    res.json({ report_id, ...data });
  } catch (err) {
    console.error("Carbon Analysis Error:", err);
    res.status(500).json({ detail: `Carbon Analysis Error: ${err.message}` });
  }
});

app.post("/api/risk", async (req, res) => {
  const { report_id } = req.body;
  const report = esgIndex.reports[report_id];
  if (!report) return res.status(404).json({ detail: "Report not found" });

  const context = claimCandidateSnippets(report, esgIndex, 40);
  const prompt = `You are an ESG analyst assessing potential greenwashing. Based ONLY on the context below, assign a simple greenwashing risk label and explanation. Choose one of these labels: Low, Medium, High.

Return a strict JSON object:
{
  "score": "Low" | "Medium" | "High",
  "explanation": string
}

Do not add other fields. Respond with JSON only.

Context:
${context}`;

  const raw = await callOpenAIChat(prompt, true);
  const data = parseStrictJson(raw) || {};
  const score = data.score || "Medium";
  const explanation = data.explanation || "";
  const fluff_ratio = calculateFluffRatio(context);

  res.json({ report_id, score, explanation, fluff_ratio });
});

app.post("/api/claims/extract", async (req, res) => {
  const { report_id, max_claims = 20 } = req.body;
  const report = esgIndex.reports[report_id];
  if (!report) return res.status(404).json({ detail: "Report not found" });

  const snippets = claimCandidateSnippets(report, esgIndex, 26);
  const prompt = `You are an ESG claim extractor. Task: extract specific ESG claims. Return ONLY valid JSON.
Output schema:
{
  "claims":[
    {
      "claim_id":"c1",
      "text":"...",
      "category":"E|S|G",
      "claim_type":"metric|target|policy|certification|compliance|other",
      "timeframe":"...",
      "page_refs":[1,2],
      "specificity":"high|medium|low"
    }
  ]
}
Rules: max ${max_claims} claims. Prefer claims with numbers, dates, standards.
Snippets:
${snippets}`;

  const raw = await callOpenAIChat(prompt, true);
  const data = parseStrictJson(raw);
  const claims = (data?.claims || []).slice(0, max_claims);
  claims.forEach((c, i) => { if (!c.claim_id) c.claim_id = `c${i+1}`; });
  res.json({ report_id, claims });
});

app.post("/api/claims/verify", async (req, res) => {
  const { report_id, claims, include_external_evidence = true, top_k_evidence = 5 } = req.body;
  if (!esgIndex.reports[report_id]) return res.status(404).json({ detail: "Report not found" });

  // Simple sequential processing for now (you can add concurrency limit with p-limit if needed)
  const results = [];
  for (const claim of (claims || [])) {
    const claimText = (claim.text || "").trim();
    if (!claimText) {
      results.push({ claim, verdict: "unsupported", confidence: 0.0, rationale: "Empty claim text", evidence: [] });
      continue;
    }

    const repHits = await esgIndex.search(claimText, top_k_evidence, [report_id]);
    const extHits = include_external_evidence ? await evidenceIndex.search(claimText, top_k_evidence) : [];

    const evidenceBlocks = [
      ...repHits.map(h => ({ source: "report", page: h.page, title: esgIndex.reports[report_id].name, snippet: h.text.substring(0, 900) })),
      ...extHits.map(h => ({ source: h.sourceType, page: h.page, title: h.title, url: h.url, snippet: h.text.substring(0, 900) }))
    ];

    const evidenceText = evidenceBlocks.map((b, i) => 
      `- [${i+1}] (${b.source}) ${b.page ? 'page ' + b.page : ''} ${b.title}: ${b.snippet}`
    ).join("\n\n");

    const prompt = `You are an ESG claim verifier. Given a CLAIM and EVIDENCE SNIPPETS, decide whether the claim is supported. Return ONLY valid JSON.
Verdict options: supported, weak, unsupported, contradictory.
Output schema: {"verdict": "...", "confidence": 0.0, "why": "...", "missing": "...", "evidence_refs": [1,2]}
CLAIM: ${claimText}
EVIDENCE:
${evidenceText}`;

    const raw = await callOpenAIChat(prompt, true);
    const data = parseStrictJson(raw) || {};
    const verdict = (data.verdict || "unsupported").toLowerCase();
    const confidence = Math.max(0, Math.min(1, parseFloat(data.confidence || 0)));
    const cited = (data.evidence_refs || []).map(r => evidenceBlocks[parseInt(r) - 1]).filter(Boolean);

    results.push({ claim, verdict, confidence, rationale: data.why || "", missing: data.missing || "", evidence: cited });
  }

  const counts = { supported: 0, weak: 0, unsupported: 0, contradictory: 0 };
  results.forEach(r => { if (counts[r.verdict] !== undefined) counts[r.verdict]++; });
  
  const total = Math.max(1, results.length);
  const score = Math.max(0, Math.min(1, (counts.weak * 0.5 + counts.unsupported * 1.0 + counts.contradictory * 1.2) / total));

  res.json({
    report_id,
    results,
    greenwashing_score: score,
    summary: { total_claims: total, ...counts }
  });
});

app.post("/api/evidence", upload.single('file'), async (req, res) => {
  const { title = "Evidence document", source_type = "external", url } = req.body;
  if (!req.file) return res.status(400).json({ detail: "No file uploaded" });

  try {
    const name = req.file.originalname;
    const contentBuffer = req.file.buffer;
    let content = "";
    let pages = [];

    if (name.toLowerCase().endsWith(".pdf")) {
      const data = await pdf(contentBuffer);
      content = data.text;
      pages = [content];
    } else {
      content = contentBuffer.toString('utf-8');
      pages = [content];
    }

    const doc = await evidenceIndex.addDoc(title, source_type, content, pages, url);
    res.json({ ok: true, evidence: doc });
  } catch (e) {
    res.status(500).json({ detail: e.message });
  }
});

app.post("/api/evidence/text", async (req, res) => {
  const { title, source_type = "external", text, url } = req.body;
  try {
    const doc = await evidenceIndex.addDoc(title, source_type, text, [text], url);
    res.json({ ok: true, evidence: doc });
  } catch (e) {
    res.status(500).json({ detail: e.message });
  }
});

app.get("/api/evidence", (req, res) => {
  res.json({ evidence: Object.values(evidenceIndex.docs) });
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`TrueScope API (Node.js) running on port ${PORT}`);
});
