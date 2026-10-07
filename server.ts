import express from "express";
import { exec, spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PYTHON_CMD = process.platform === "win32" ? "python" : "python3";
const pythonEnv = {
  ...process.env,
  PYTHONIOENCODING: "utf-8",
  PYTHONUTF8: "1",
};

async function startServer() {
  const app = express();
  const PORT = process.env.PORT || 3000;

  app.use(express.json());

  // 1. Search Endpoint
  app.post("/api/search", (req, res) => {
    const {
      query,
      mode = "indicbridge",
      top_k = 8,
      alpha = 0.7,
      beta = 0.15,
      gamma = 0.1,
      delta = 0.05,
    } = req.body;

    if (!query || !query.trim()) {
      return res.status(400).json({ error: "Query is required" });
    }

    const args = [
      "indicbridge.py",
      "search",
      query,
      "--mode",
      mode,
      "--top-k",
      String(top_k),
      "--alpha",
      String(alpha),
      "--beta",
      String(beta),
      "--gamma",
      String(gamma),
      "--delta",
      String(delta),
      "--explain",
      "--json",
    ];

    const child = spawn(PYTHON_CMD, args, { cwd: __dirname, env: pythonEnv });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data) => (stdout += data.toString("utf-8")));
    child.stderr.on("data", (data) => (stderr += data.toString("utf-8")));

    child.on("close", (code) => {
      if (code !== 0) {
        console.error("Search child process error:", stderr);
        return res.status(500).json({ error: stderr || "Search failed" });
      }
      try {
        res.json(JSON.parse(stdout));
      } catch (err) {
        console.error("JSON parse failure. Raw output:", stdout);
        res.status(500).json({ error: "Failed to parse Python response" });
      }
    });
  });

  // 2. Diagnostics Endpoint
  app.get("/api/diagnostics", (req, res) => {
    const term = req.query.term as string;
    if (!term) return res.status(400).json({ error: "Term parameter is required" });

    const child = spawn(PYTHON_CMD, ["indicbridge.py", "diag", term, "--json"], {
      cwd: __dirname,
      env: pythonEnv,
    });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data) => (stdout += data.toString("utf-8")));
    child.stderr.on("data", (data) => (stderr += data.toString("utf-8")));

    child.on("close", (code) => {
      if (code !== 0) return res.status(500).json({ error: stderr });
      try {
        res.json(JSON.parse(stdout));
      } catch {
        res.status(500).json({ error: "Failed to parse diagnostics" });
      }
    });
  });

  // 3. Evaluation Endpoint (Direct cache fallback for instant load)
  app.get("/api/evaluation", (req, res) => {
    const evalResultsPath = path.join(__dirname, "evaluation", "results", "evaluation_results.json");
    if (fs.existsSync(evalResultsPath)) {
      try {
        const cached = JSON.parse(fs.readFileSync(evalResultsPath, "utf-8"));
        return res.json(cached);
      } catch (e) {
        // Fall back to script execution
      }
    }

    const child = spawn(PYTHON_CMD, ["indicbridge.py", "eval", "--json"], {
      cwd: __dirname,
      env: pythonEnv,
    });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data) => (stdout += data.toString("utf-8")));
    child.stderr.on("data", (data) => (stderr += data.toString("utf-8")));

    child.on("close", (code) => {
      if (code !== 0) return res.status(500).json({ error: stderr });
      try {
        res.json(JSON.parse(stdout));
      } catch {
        res.status(500).json({ error: "Failed to parse evaluation metrics" });
      }
    });
  });

  // 4. Ablation Endpoint
  app.get("/api/ablation", (req, res) => {
    const ablationCsv = path.join(__dirname, "evaluation", "results", "ablation_results.csv");

    const serveCsv = () => {
      if (fs.existsSync(ablationCsv)) {
        const content = fs.readFileSync(ablationCsv, "utf-8").trim();
        const lines = content.split("\n");
        const headers = lines[0].split(",").map((h) => h.trim());
        const rows = lines.slice(1).map((line) => {
          const vals = line.split(",").map((v) => v.trim());
          const item: Record<string, any> = {};
          headers.forEach((h, i) => {
            const num = parseFloat(vals[i]);
            item[h] = isNaN(num) ? vals[i] : num;
          });
          return item;
        });
        return res.json({ rows });
      }
      res.json({ rows: [] });
    };

    if (fs.existsSync(ablationCsv)) {
      serveCsv();
    } else {
      const child = spawn(PYTHON_CMD, ["evaluation/ablation.py"], {
        cwd: __dirname,
        env: pythonEnv,
      });
      child.on("close", () => serveCsv());
    }
  });

  // 5. In-Browser CLI Sandbox with ANSI Strip
  app.post("/api/cli", (req, res) => {
    const { command } = req.body;
    if (!command) return res.status(400).json({ error: "Command required" });

    const normalizedCmd = command.replace(/^python3\s+/, "python ");
    const start = Date.now();

    exec(
      normalizedCmd,
      { cwd: __dirname, env: pythonEnv },
      (error, stdout, stderr) => {
        const elapsedMs = Date.now() - start;
        // Strip ANSI color codes
        const cleanStdout = (stdout || "").replace(/\x1b\[[0-9;]*[a-zA-Z]/g, "");
        const cleanStderr = (stderr || (error ? error.message : "")).replace(/\x1b\[[0-9;]*[a-zA-Z]/g, "");
        res.json({
          stdout: cleanStdout,
          stderr: cleanStderr,
          exitCode: error ? error.code || 1 : 0,
          elapsedMs,
        });
      }
    );
  });

  // Vite development middleware
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: "spa",
  });
  app.use(vite.middlewares);

  app.listen(PORT, () => {
    console.log(`\n======================================================`);
    console.log(`[+] IndicBridge Server running at http://localhost:${PORT}`);
    console.log(`[+] Python executable: ${PYTHON_CMD}`);
    console.log(`======================================================\n`);
  });
}

startServer();