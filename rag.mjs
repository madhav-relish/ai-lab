import { google } from "@ai-sdk/google";
import { embed, embedMany, generateText } from "ai";


const c = {
    reset: "\x1b[0m", bold: "\x1b[1m", dim:"\x1b[2m",
    cyan: "\x1b[36m", green: "\x1b[32m", yellow: "\x1b[33m", magenta: "\x1b[35m"
};

const step = (n, title) => console.log(`\n${c.cyan}${c.bold}---STEP ${n}: ${title} ---${c.reset}`);
const info = (label, value) => console.log(`  ${c.dim} ${label}: ${c.reset} ${value}`)

const bar = (score) => "[*]".repeat(Math.round(score * 10)).padEnd(10, "[]");
 
// 1 - Knowledge Base
const docs = [
    "The Eiffel Tower is in Paris and was completed in 1889.",
    "Python is a programming language that was created by Guido van Rossum in 1991.",
    "The Great Wall of China is over 13,000 miles long.",
    "React is a Javascript library for building user interfades, made by Meta."
]

step(1, "Knowledge Base");
docs.forEach((d,i) => info(`doc[${i}]`, d));

// 2 - Write Path -- Embed Every doc

step(2, "Embedding the document (write path)");

const { embeddings } = await embedMany({
    model: google.embeddingModel("gemini-embedding-001"),
    values: docs
});

info("document embedded", `${embeddings.length}`)
info("Vector dimensions each", `${embeddings[0].length}`);
console.log(`  ${c.dim} {each doc is now a list of ${embeddings[0].length} numbers capturing its meaning} ${c.reset}`);


// 3 - Cosine Similiarity = closeness in meaning
function cosine(a,b){
    let dot = 0, na = 0, nb = 0;
    for(let i = 0; i < a.length; i++){
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];

    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

// 4 - Read Path: Embed question + retrieve
const question = "Which javascript library was made by Meta?";

step(4, "The question");
info("question", `${c.yellow} ${question} ${c.reset}` )


step(4.1, "Retrieval - scoring every doc against the question")

const { embedding: qVec } = await embed({
    model: google.embeddingModel("gemini-embedding-001"),
    value: question
})

const scored = docs.map((text,i) => ({ text , score: cosine(qVec, embeddings[i])})).sort((a,b) => b.score - a.score)
 
 scored.forEach((r)=> {
    console.log(` ${c.magenta}${bar(r.score)}${c.reset} ${c.bold}${r.score.toFixed(3)}${c.reset} ${c.dim} ${r.text} ${c.reset}`)
 })

 const TOP_K = 2;
 const retrieved = scored.slice(0, TOP_K);
 console.log(`  ${c.dim} -> keeping top ${TOP_K} as context ${c.reset}`)

 // 5 - Generate

 step(5, "Generation -- answering from the retrieved context only");
 info("Context sent to model", "");
 retrieved.forEach((r) => console.log(`   ${c.dim} - ${r.text}${c.reset}`))

 const { text } = await generateText({
    model: google("gemini-2.5-flash"),
    temperature: 0,
    prompt: `Answer using only the context below.
    If the context contains the answer, give it directly.
    If the answer is NOT there in the context, say "I don't know."
    
    Context : ${retrieved.map((r) => "- " + r.text).join("\n")};

    Question: ${question}
    `
 });

 console.log(`\n${c.green}${c.bold} ✅ ANSWER: ${c.reset} ${c.green} ${text} ${c.reset}\n`)

