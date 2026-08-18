import { google } from "@ai-sdk/google";
import { embed, embedMany, generateText } from "ai";
import { readFileSync } from "fs";

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

function cosine(a,b){
    let dot = 0, na = 0, nb = 0;
    for(let i = 0; i < a.length; i++){
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];

    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

// Load + Chunk the content Doc

const text = readFileSync("./content.txt", "utf-8");
const chunks = chunkText(text);

step(1, "Chunking the Document");
console.log(`  ${c.dim}total words::${c.reset} ${text.split(/\s+/).length}`);
console.log(`  ${c.dim}Chunks created::${c.reset} ${chunks.length}`);

chunks.forEach((ch, i) =>
  console.log(
    `  ${c.magenta}chunk[${i}]${c.reset} ${c.dim}${ch.slice(0, 70)}...${c.reset}`,
  ),
);

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
// const question = "What are the Ten Essentials that are recommended to all hikers?";
const question = "when was Bhabha committee found?"

step(3, "Retrieval");
console.log(`  ${c.dim}Question ${c.reset} ${c.yellow}${question}${c.reset}`);

const { embedding: qVec } = await embed({
  model: google.embeddingModel("gemini-embedding-001"),
  value: question,
});

const scored = chunks
  .map((text, i) => ({ text, score: cosine(qVec, embeddings[i]) }))
  .sort((a, b) => b.score - a.score);


  scored.slice(0,5).forEach((r)=>
console.log(` ${c.magenta}${bar(r.score)}${c.reset} ${c.bold}${r.score.toFixed(3)}${c.reset} ${c.dim} ${r.text.slice(0,60)}... ${c.reset}`)
)

const retrieved = scored.slice(0,5); // Top-3 chunks

console.log(`\n${c.yellow}CONTEXT BEING SENT:${c.reset}`);
retrieved.forEach((r, i) => console.log(`[${i}] ${r.text}`));


step(4, "Generation (grounded, temprature 0)");
const { text: answer }  = await generateText({
    model: google("gemini-2.5-flash"),
    temperature: 0,
    prompt: `Answer using ONLY the context below. If the answer isn't there, say "I don't know."
    
    context: 
    ${retrieved.map((r) => "- " + r.text).join("\n")}


    Question: ${question}
    `
})

 console.log(`\n${c.green}${c.bold} ✅ ANSWER: ${c.reset} ${c.green} ${answer} ${c.reset}\n`)