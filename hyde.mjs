import { google } from "@ai-sdk/google";
import { embed, generateText } from "ai";

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

// ---- HyDE: generate a hypothetical answer, then search with it
const { text: hypothetical } = await generateText({
    model: google('gemini-2.5-flash'),
    temperature: 0,
    prompt: `Write a short, factual paragraph that would answer the question.
    If unsure, write what a correct answer would typically look like.
    
    Question: ${question}
    `
})


//embed the hypothetical answer (Not the RAW question)
const { embedding: hydeVec } = await embed({
    model: google.embeddingModel("gemini-embedding-001"),
    value: hypothetical,
})

//retrieve with each , side by side
const rank = (vec) => chunks.map((text, i) => ({
    i, text, score: cosine(vec, embeddings[i])
})).sort((a, b) => b.score - a.score);

const rawRanked = rank(qVec);
const hydeRanked = rank(hydeVec)