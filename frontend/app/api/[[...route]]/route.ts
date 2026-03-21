import { NextRequest, NextResponse } from 'next/server';
import { OpenAI } from 'openai';
import pdf from 'pdf-parse';
import { v4 as uuidv4 } from 'uuid';

// ---------------------------
// Config
// ---------------------------
const api_key = process.env.OPENAI_API_KEY;
const client = new OpenAI({ apiKey: api_key });
const GEN_MODEL = "gpt-4o";
const EMBED_MODEL = "text-embedding-3-small";

// ---------------------------
// In-Memory Global State (Note: Resets on cold start!)
// ---------------------------
const globalState: any = global as any;
if (!globalState.esgIndex) {
  globalState.esgIndex = { chunks: [], reports: {} };
}
const esgIndex = globalState.esgIndex;

// ---------------------------
// Backend Utilities (Consolidated for Serverless)
// ---------------------------
function l2Normalize(vec: number[]) {
  const norm = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0)) || 1.0;
  return vec.map(val => val / norm);
}

function dotProduct(vec1: number[], vec2: number[]) {
  return vec1.reduce((sum, val, idx) => sum + val * vec2[idx], 0);
}

async function embedText(text: string) {
  const response = await client.embeddings.create({
    input: [text.replace(/\n/g, " ")],
    model: EMBED_MODEL
  });
  return l2Normalize(response.data[0].embedding);
}

async function embedTexts(texts: string[]) {
  const response = await client.embeddings.create({
    input: texts.map(t => t.replace(/\n/g, " ")),
    model: EMBED_MODEL
  });
  return response.data.map(d => l2Normalize(d.embedding));
}

function chunkPages(pages: string[], chunkChars = 1600, overlap = 150) {
  const chunks = [];
  for (let i = 0; i < pages.length; i++) {
    const text = (pages[i] || "").trim();
    if (!text) continue;
    let start = 0;
    while (start < text.length) {
      const end = Math.min(start + chunkChars, text.length);
      const chunk = text.substring(start, end).trim();
      if (chunk.length > 50) chunks.push({ text: chunk, page: i + 1 });
      if (end === text.length) break;
      start = end - overlap;
    }
  }
  return chunks;
}

function claimCandidateSnippets(report_id: string, max_snippets = 30) {
  const chunks = esgIndex.chunks.filter((c: any) => c.reportId === report_id);
  const keywords = ["net zero", "carbon", "reduction", "scope", "target", "compliance", "verification", "verified", "diversity", "waste", "water"];
  
  const scored = chunks.map((c: any) => {
    let score = 0;
    const t = c.text.toLowerCase();
    keywords.forEach(kw => { if (t.includes(kw)) score++; });
    if (/\d+%/.test(t)) score += 2;
    return { score, chunk: c };
  }).sort((a: any, b: any) => b.score - a.score);

  return scored.slice(0, max_snippets).map((s: any) => `[Page ${s.chunk.page}] ${s.chunk.text}`).join("\n\n");
}

async function callOpenAIChat(prompt: string, jsonMode = false) {
  const response = await client.chat.completions.create({
    model: GEN_MODEL,
    messages: [{ role: "user", content: prompt }],
    response_format: jsonMode ? { type: "json_object" } : undefined
  });
  return response.choices[0].message.content;
}

// ---------------------------
// Route Handler
// ---------------------------
export async function POST(req: NextRequest, { params }: any) {
  const p = await params;
  const path = p.route?.join('/') || '';

  try {
    // ---------------------------
    // /api/reports (Upload)
    // ---------------------------
    if (path === 'reports') {
      const data = await req.formData();
      const files: File[] = data.getAll('files') as any;
      
      for (const file of files) {
        const buffer = Buffer.from(await file.arrayBuffer());
        let fullText = "";
        let pagesText: string[] = [];

        if (file.name.endsWith('.pdf')) {
          const doc = await pdf(buffer);
          fullText = doc.text;
          pagesText = [fullText]; // pdf-parse basic
        } else {
          fullText = buffer.toString('utf-8');
          pagesText = [fullText];
        }

        const reportId = `rep_${uuidv4()}`;
        const chunks = chunkPages(pagesText);
        const embeddings = await embedTexts(chunks.map(c => c.text));

        chunks.forEach((chunk, idx) => {
          esgIndex.chunks.push({
            reportId,
            reportName: file.name,
            text: chunk.text,
            page: chunk.page,
            embedding: embeddings[idx]
          });
        });

        esgIndex.reports[reportId] = { id: reportId, name: file.name, uploadedAt: new Date().toISOString() };
      }
      return NextResponse.json({ reports: Object.values(esgIndex.reports) });
    }

    // ---------------------------
    // /api/compliance, /api/risk, /api/metrics, /api/frameworks, /api/risk/predict, /api/carbon/analysis
    // ---------------------------
    const body = await req.json();
    const report_id = body.report_id;
    const report = esgIndex.reports[report_id];
    if (!report && path !== 'query') return NextResponse.json({ error: "Report not found" }, { status: 404 });

    const context = report_id ? claimCandidateSnippets(report_id) : "";

    if (path === 'risk') {
      const prompt = `Assess potential greenwashing risk (Low|Medium|High) for this ESG report based only on: ${context}. Return JSON: { "score": "...", "explanation": "..." }`;
      const res = await callOpenAIChat(prompt, true);
      return NextResponse.json(JSON.parse(res || "{}"));
    }

    if (path === 'metrics') {
      const prompt = `Extract Scope 1, 2, 3 emissions and key ESG stats as JSON { "metrics": { ... } } from: ${context}`;
      const res = await callOpenAIChat(prompt, true);
      return NextResponse.json(JSON.parse(res || "{}"));
    }

    if (path === 'compliance') {
      const prompt = `Check GRI/SASB/TCFD alignment from: ${context}. Return JSON { "compliance": { ... } }`;
      const res = await callOpenAIChat(prompt, true);
      return NextResponse.json(JSON.parse(res || "{}"));
    }

    if (path === 'frameworks') {
      const prompt = `Detailed GRI/TCFD/SASB audit alignment scores from: ${context}. Return JSON { "gri_alignment": { "score": 85... }, ... }`;
      const res = await callOpenAIChat(prompt, true);
      return NextResponse.json(JSON.parse(res || "{}"));
    }

    if (path === 'risk/predict') {
      const prompt = `Predict future controversies for this company. Return JSON { "predicted_risks": [...], "controversy_likelihood": 75 } using: ${context}`;
      const res = await callOpenAIChat(prompt, true);
      return NextResponse.json(JSON.parse(res || "{}"));
    }

    if (path === 'carbon/analysis') {
      const prompt = `Deep carbon accounting and Scope 1/2/3 insights. Return JSON { "breakdown": { ... }, "paris_alignment_score": 80 } from: ${context}`;
      const res = await callOpenAIChat(prompt, true);
      return NextResponse.json(JSON.parse(res || "{}"));
    }

    if (path === 'summary') {
      const prompt = `Write a markdown ESG summary based on: ${context}`;
      const summary = await callOpenAIChat(prompt);
      return NextResponse.json({ summary_md: summary });
    }

    if (path === 'claims/extract') {
      const prompt = `Extract specific ESG claims as JSON { "claims": [...] } from: ${context}`;
      const res = await callOpenAIChat(prompt, true);
      return NextResponse.json(JSON.parse(res || "{}"));
    }

    if (path === 'claims/verify') {
      const claims = body.claims || [];
      const results = [];
      for (const claim of claims) {
        const prompt = `Verify this ESG claim: "${claim.text}" using evidence from the report: ${context}. Return JSON { "verdict": "supported|weak|unsupported", "confidence": 0.9, "why": "..." }`;
        const res = await callOpenAIChat(prompt, true);
        const data = JSON.parse(res || "{}");
        results.push({ claim, ...data });
      }
      return NextResponse.json({ results, summary: { total_claims: results.length } }); // Simplified summary
    }

    if (path === 'query') {
      const query = body.question;
      const emb = await embedText(query);
      const searchResults = esgIndex.chunks
        .map((c: any) => ({ ...c, score: dotProduct(emb, c.embedding) }))
        .sort((a: any, b: any) => b.score - a.score)
        .slice(0, 5);
      
      const ctx = searchResults.map((s: any) => s.text).join("\n\n");
      const answer = await callOpenAIChat(`Answer the question based on this ESG context: ${ctx}. Question: ${query}`);
      return NextResponse.json({ answer, citations: searchResults });
    }

    return NextResponse.json({ detail: "Route not found" }, { status: 404 });
  } catch (error: any) {
    console.error("API Route Error:", error);
    return NextResponse.json({ detail: error.message }, { status: 500 });
  }
}

export async function GET(req: NextRequest, { params }: any) {
  const p = await params;
  const path = p.route?.join('/') || '';

  if (path === 'reports') {
    return NextResponse.json({ reports: Object.values(esgIndex.reports) });
  }

  if (path === 'health') {
    return NextResponse.json({ status: "ok", reports: Object.keys(esgIndex.reports).length });
  }

  return NextResponse.json({ detail: "Route not found" }, { status: 404 });
}
