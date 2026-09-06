import { google } from "@ai-sdk/google";
import { embed, embedMany, generateText, Output } from "ai";
import { readFileSync } from "fs";
import z from "zod";

const c = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  magenta: "\x1b[35m",
};

const step = (n, title) =>
  console.log(`\n${c.cyan}${c.bold}---STEP ${n}: ${title} ---${c.reset}`);
const info = (label, value) =>
  console.log(`  ${c.dim} ${label}: ${c.reset} ${value}`);

const bar = (score) => "[*]".repeat(Math.round(score * 10)).padEnd(10, "[]");


// Keeping overlap so that each chunk has some meaning of it's sorroundings
function chunkText(text, size = 60, overlap = 15) {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  for (let i = 0; i < words.length; i += size - overlap) {
    chunks.push(words.slice(i, i + size).join(" "));
    if (i + size > words.length) break;
  }
  return chunks;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];

  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

// Load + Chunk the content Doc

const text = readFileSync("./content.txt", "utf-8");
const chunks = chunkText(text, 100, 20);

step(1, "Chunking the Document");
console.log(`  ${c.dim}total words::${c.reset} ${text.split(/\s+/).length}`);
console.log(`  ${c.dim}Chunks created::${c.reset} ${chunks.length}`);

chunks.forEach((ch, i) =>
  console.log(
    `  ${c.magenta}chunk[${i}]${c.reset} ${c.dim}${ch.slice(0, 70)}...${c.reset}`,
  ),
);

// tokenize: lowercase + split into words
function tokenize(text) {
  return text.toLowerCase().match(/\w+/g) || [];
}

function buildBM25Index(chunks) {
  const docs = chunks.map(tokenize);
  const N = docs.length;
  //Average document length
  const avgdl = docs.reduce((sum, doc) => sum + doc.length, 0) / Math.max(N, 1);

  //Document frequence - how many chunks contain word
  // BM-25 is a lexical search which excels at exact word matching
  const df = {};
  for (const doc of docs) {
    const seen = new Set();
    for (const term of doc) {
      if (seen.has(term)) continue;
      seen.add(term);
      df[term] = (df[term] || 0) + 1;
    }
  }

  return { docs, N, avgdl, df };
}

const bm25Index = buildBM25Index(chunks);

function scoreBM25Query(query, docTokens, index, options = {}) {
  const { k1 = 1.5, b = 0.75 } = options;
  const queryTerms = tokenize(query);

  if (!queryTerms.length || !docTokens.length) return 0;

  const seen = new Set();
  let score = 0;

  for (const term of queryTerms) {
    if (seen.has(term)) continue;
    seen.add(term);

    const tf = docTokens.filter((token) => token === term).length;
    const df = index.df[term] || 0;
    if (df === 0) continue;

    const idf = Math.log(((index.N - df + 0.5) / (df + 0.5)) + 1);
    const docLen = docTokens.length;
    const denom = tf + k1 * (1 - b + b * (docLen / index.avgdl));
    score += idf * ((tf * (k1 + 1)) / denom);
  }

  return score;
}

function reciprocalRankFusion(lists, k = 60) {
  const merged = new Map();

  for (const list of lists) {
    list.forEach((item, rank) => {
      const key = item.index;
      const existing = merged.get(key) || {
        index: item.index,
        text: item.text,
        fusedScore: 0,
      };

      existing.fusedScore += 1 / (k + rank);
      existing.text = item.text;
      merged.set(key, existing);
    });
  }

  return Array.from(merged.values())
    .map((item) => ({ ...item, score: item.fusedScore }))
    .sort((a, b) => b.score - a.score);
}

function showCandidateList(label, list) {
  console.log(`\n${c.yellow}${label}${c.reset}`);
  list.forEach((item, i) => {
    const score = item.score ?? 0;
    console.log(
      `  [${i}] ${c.magenta}${bar(Math.min(score, 1))}${c.reset} ${c.bold}${score.toFixed(3)}${c.reset} ${c.dim}${item.text.slice(0, 75)}...${c.reset}`,
    );
  });
}

// 2. Embed Each chunk (write path)

step(2, "Embed the chunks");
const { embeddings } = await embedMany({
  model: google.embeddingModel("gemini-embedding-001"),
  values: chunks,
});

console.log(
  `  ${c.dim}Embedded ${embeddings.length} chunks ${embeddings[0].length} dims each${c.reset}`,
);

// 3. Asking the question + Retrieve
const question = "What are the Ten Essentials that are recommended to all hikers?";
// const question = "when was Bhabha committee found?"

step(3, "Retrieval: semantic + BM25 + RRF");
console.log(`  ${c.dim}Question ${c.reset} ${c.yellow}${question}${c.reset}`);

const { embedding: qVec } = await embed({
  model: google.embeddingModel("gemini-embedding-001"),
  value: question,
});

const semanticCandidates = chunks
  .map((chunkTextValue, i) => ({
    index: i,
    text: chunkTextValue,
    score: cosine(qVec, embeddings[i]),
  }))
  .sort((a, b) => b.score - a.score)
  .slice(0, 10);

const bm25Candidates = chunks
  .map((chunkTextValue, i) => ({
    index: i,
    text: chunkTextValue,
    score: scoreBM25Query(question, bm25Index.docs[i], bm25Index),
  }))
  .filter((candidate) => candidate.score > 0)
  .sort((a, b) => b.score - a.score)
  .slice(0, 10);

showCandidateList("Vector search (cosine similarity)", semanticCandidates);
showCandidateList("BM25 lexical search", bm25Candidates);

console.log(`\n${c.yellow}RRF fusion formula:${c.reset} sum(1 / (k + rank))`);
console.log(`  ${c.dim}k = 60, rank starts at 0 for the top result${c.reset}`);

const fusedCandidates = reciprocalRankFusion([
  semanticCandidates.map((candidate, rank) => ({ ...candidate, rank })),
  bm25Candidates.map((candidate, rank) => ({ ...candidate, rank })),
], 60).slice(0, 8);

showCandidateList("Reciprocal Rank Fusion (semantic + BM25)", fusedCandidates);

const retrieved = fusedCandidates.slice(0, 5);

console.log(`\n${c.yellow}CONTEXT BEING SENT:${c.reset}`);
retrieved.forEach((r, i) => console.log(`[${i}] ${r.text}`));

step(4, "LLM Re-ranking (Cross-Encoder)");

const candidatePool = fusedCandidates;
const { output } = await generateText({
  model: google("gemini-2.5-flash"),
  output: Output.object({
    schema: z.object({
      rankings: z.array(
        z.object({
          index: z.number().describe("the candidate index in the list below"),
          relevance: z.number().describe("0 (irrelevant) to 10 (directly answers the question)"),
        }),
      ),
    }),
  }),
  prompt: `
    Score how well each candidate passage answers the question (0-10).

    Question: ${question}

    Candidates:
    ${candidatePool.map((cand, i) => `[${i}] ${cand.text}`).join("\n\n")}
  `,
});

const reRanked = output?.rankings
  ?.sort((a, b) => b.relevance - a.relevance)
  .slice(0, 3)
  .map((r) => ({ ...candidatePool[r.index], reRank: r.relevance }));

console.log(`\n${c.yellow}Reranked CONTEXT BEING SENT:${c.reset}`);
reRanked.forEach((r, i) => console.log(`[${i}] ${r.reRank}/10 (RRF=${r.score.toFixed(3)}) ${r.text.slice(0, 55)}...`));

step(5, "Generation from reranked context");
const { text: answer } = await generateText({
  model: google("gemini-2.5-flash"),
  temperature: 0,
  prompt: `Answer completely using ONLY the context below. If the answer isn't there, say "I don't know."

    context:
    ${reRanked.map((r) => "- " + r.text).join("\n")}

    Question: ${question}
  `,
});

console.log(`\n${c.green}${c.bold} ✅ ANSWER: ${c.reset} ${c.green} ${answer} ${c.reset}\n`)