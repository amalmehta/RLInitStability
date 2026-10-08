// Static server for site/. Starts on $PORT or 8765 and moves to the next
// free port if that one is taken. Sends no-cache headers so edited modules
// are never served stale. Run: npm start
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../site/', import.meta.url));
const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const server = createServer(async (req, res) => {
  let path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (path.endsWith('/')) path += 'index.html';
  const file = normalize(join(root, path));
  if (!file.startsWith(root.endsWith(sep) ? root : root + sep)) {
    res.writeHead(403).end('Forbidden');
    return;
  }
  try {
    const body = await readFile(file);
    res.writeHead(200, { 'Content-Type': TYPES[extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(body);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain' }).end('Not found');
  }
});

const first = Number(process.env.PORT) || 8765;
let port = first;
server.on('error', (err) => {
  if (err.code === 'EADDRINUSE' && port < first + 20) {
    port += 1;
    server.listen(port);
  } else {
    console.error(err.message);
    process.exit(1);
  }
});
server.on('listening', () => {
  if (port !== first) console.log(`Port ${first} is busy, using ${port} instead.`);
  console.log(`RL Init and Stability running at http://localhost:${port}  (Ctrl+C to stop)`);
});
server.listen(port);
