import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Initialize Gemini Client
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// Chat API endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { messages, currentContext, model = 'gemini-3.8-flash' } = req.body;

    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        error: 'GEMINI_API_KEY is not configured in the Secrets panel.',
      });
    }

    const systemInstruction = `You are the DotML AI Mentor, an expert machine learning researcher and visual educator.
Your mission is to help users intuitively understand machine learning algorithms (K-Means, K-Nearest Neighbors, Linear Regression & Gradient Descent, Support Vector Machines, Neural Networks, PCA, Decision Trees) through the lens of data dots, geometry, loss surfaces, and decision boundaries.

Context about the user's current simulation:
${currentContext ? JSON.stringify(currentContext, null, 2) : 'General ML Cosmos exploration'}

Guidelines:
- Explain concepts using geometric intuition (dots, margins, vectors, valleys, clusters).
- Connect mathematical formulas to what they visually do to the data dots on screen.
- Keep responses friendly, engaging, concise, and pedagogical.
- Format formulas cleanly with clear notation.
- If asked about specific parameter tweaks (e.g. learning rate, K, C), explain the visual consequence (e.g., overshoot, overfitting, margin width).`;

    // Convert history to contents format
    const contents = (messages || []).map((m: { role: string; content: string }) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }],
    }));

    const response = await ai.models.generateContent({
      model: model || 'gemini-3.8-flash',
      contents,
      config: {
        systemInstruction,
      },
    });

    const replyText = response.text || 'I could not generate an explanation at this moment.';
    return res.json({ reply: replyText });
  } catch (err: any) {
    console.error('Gemini API Error:', err);
    return res.status(500).json({
      error: err?.message || 'Failed to communicate with Gemini API.',
    });
  }
});

// AI Data Ingestion API endpoint (Any format -> 2D ML Data Dots)
app.post('/api/ingest-data', async (req, res) => {
  try {
    const { rawText, instructions = '' } = req.body;

    if (!process.env.GEMINI_API_KEY) {
      return res.status(500).json({
        error: 'GEMINI_API_KEY is not configured in the Secrets panel.',
      });
    }

    if (!rawText || typeof rawText !== 'string' || !rawText.trim()) {
      return res.status(400).json({ error: 'Please provide raw data text to ingest.' });
    }

    const systemInstruction = `You are an expert Data Engineer & ML Preprocessor for DotML, an interactive 2D visual machine learning cosmos.
Your job is to ingest raw user data in ANY format (natural language descriptions, medical/patient vitals, server logs, messy tables, JSON, key-values, sensor telemetry, conversational text, unformatted lists) and extract clean 2D data dots ready for machine learning algorithms.

Requirements:
- Extract between 20 and 70 distinct data points.
- Extract or synthesize two informative continuous features and normalize them strictly to [0.05, 0.95] for x and y coordinates.
- If classification is implied, assign integer label 0 or 1 based on the class outcome.
- If regression is implied, set continuousTarget as a float between 0.05 and 0.95.
- Preserve realistic correlation, clusters, or boundaries from the source data.
- Return ONLY valid JSON with this exact structure:
{
  "datasetName": string,
  "description": string,
  "feature1Name": string,
  "feature2Name": string,
  "targetName": string,
  "dots": [
    {
      "x": number,
      "y": number,
      "label": 0 or 1,
      "continuousTarget": number,
      "rawX": string | number,
      "rawY": string | number,
      "rawLabel": string
    }
  ]
}`;

    const prompt = `Here is the user's raw input data:
---
${rawText.slice(0, 10000)}
---
${instructions ? `User guidance: ${instructions}` : ''}

Extract and normalize these into 2D machine learning data dots JSON.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      config: {
        systemInstruction,
        responseMimeType: 'application/json',
      },
    });

    const parsedJson = JSON.parse(response.text || '{}');
    return res.json(parsedJson);
  } catch (err: any) {
    console.error('Ingest Data API Error:', err);
    return res.status(500).json({
      error: err?.message || 'Failed to ingest data with Gemini AI.',
    });
  }
});

// Dev server or Production static serving
async function startServer() {
  const isDev = process.env.NODE_ENV !== 'production';

  if (isDev) {
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
