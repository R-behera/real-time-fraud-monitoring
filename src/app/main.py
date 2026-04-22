from pathlib import Path
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((BASE / 'config' / 'manifest.json').read_text())
WEB = Path(__file__).resolve().parent / 'web'

app = FastAPI(title=MANIFEST['title'], version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.mount('/assets', StaticFiles(directory=str(WEB)), name='assets')

@app.get('/')
def root():
    return FileResponse(WEB / 'index.html')

@app.get('/health')
def health():
    return JSONResponse({'status':'ok','project':MANIFEST['slug']})

@app.get('/api/manifest')
def manifest():
    return JSONResponse(MANIFEST)

@app.get('/api/readiness')
def readiness():
    return JSONResponse(MANIFEST['api_examples']['readiness'])

@app.get('/api/signals')
def signals():
    return JSONResponse(MANIFEST['api_examples']['signals'])
