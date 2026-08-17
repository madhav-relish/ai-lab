import { google } from "@ai-sdk/google";
import { generateText } from "ai";


const { text } = await generateText({
    model: google('gemini-2.5-flash'),
    prompt: "In two sentences, explain what 'Applied AI means for a frontend developer'"
})

console.log("AI Said::", text)