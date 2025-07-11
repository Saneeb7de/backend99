# backend/main.py

import time
from collections import defaultdict
import os
import shutil
import uuid
import asyncio
import json
import base64
from pathlib import Path
from contextlib import asynccontextmanager


from dotenv import load_dotenv
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import google.generativeai as genai
import aiofiles
from google.api_core import exceptions

# NEW: Import Google Cloud Speech client
from google.cloud import speech

# Import from our other files
from backend.database import engine, async_session_factory
from backend.models import Base, Transcript
from backend.audio_processor import process_and_split_audio
from backend.utils import (
    VERBATIM_TRANSCRIPTION_PROMPT,
    process_medical_data_with_gemini_async,
    combine_transcription_data,
    create_gemini_model,
    cleanup_uploaded_files
)

# --- Setup ---
chunk_upload_times = defaultdict(dict)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

# Initialize Google Cloud Speech client for production
def initialize_speech_client():
    """Initialize Google Cloud Speech client with proper credential handling"""
    try:
        # Check if running in production (Render)
        if os.getenv("RENDER"):
            # In production, use the base64-encoded credentials from environment
            encoded_creds = os.getenv("GOOGLE_CLOUD_CREDENTIALS_BASE64")
            if encoded_creds:
                # Decode and write credentials to file
                creds_json = base64.b64decode(encoded_creds).decode('utf-8')
                creds_path = "/tmp/gcloud.json"
                with open(creds_path, "w") as f:
                    f.write(creds_json)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
            else:
                print("⚠️ WARNING: GOOGLE_CLOUD_CREDENTIALS_BASE64 not found in environment")
                return None
        else:
            # In development, use local gcloud.json file
            if not os.path.exists("gcloud.json"):
                print("⚠️ WARNING: gcloud.json not found. Live transcription will be disabled.")
                return None
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcloud.json"
        
        speech_client = speech.SpeechAsyncClient()
        print("✅ Google Cloud Speech client initialized successfully.")
        return speech_client
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize Google Cloud Speech client: {e}")
        return None
    
speech_client = initialize_speech_client()

genai.configure(api_key=GOOGLE_API_KEY)
model = create_gemini_model(temperature=0.3)

TEMP_DIR = Path("temp_audio")
tasks = {}
active_websockets = set()

# --- Lifespan and App Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global active_websockets
    active_websockets = set()
    TEMP_DIR.mkdir(exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Server startup complete.")
    yield
    print("--- Server shutting down ---")
    # Clean up any lingering connections
    for websocket in list(active_websockets):
        try:
            await websocket.close(code=1012)
        except:
            pass
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    await engine.dispose()

app = FastAPI(lifespan=lifespan)

# Configure CORS for production
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Database & Pydantic Models ---
async def get_session() -> AsyncSession:
    async with async_session_factory() as session:
        yield session

class FinalizeRequest(BaseModel):
    session_id: str

class TranscriptCreate(BaseModel):
    content: str

class TranscriptUpdate(BaseModel):
    content: str

# --- Live Transcription WebSocket Endpoint ---
@app.websocket("/ws/live-transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)

    if not speech_client:
        await websocket.send_text(json.dumps({
            "error": "Live transcription is not configured on the server."
        }))
        await websocket.close(code=1011)
        active_websockets.discard(websocket)
        return

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        sample_rate_hertz=48000,
        language_code="ml-IN",
        alternative_language_codes=["en-IN"],
        enable_automatic_punctuation=True,
        model="latest_long",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False
    )

    audio_queue = asyncio.Queue()

    async def audio_generator():
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    async def receive_audio_task():
        try:
            while True:
                audio_chunk = await websocket.receive_bytes()
                await audio_queue.put(audio_chunk)
        except WebSocketDisconnect:
            print("WebSocket disconnected by client.")
        except Exception as e:
            print(f"Error in receive_audio_task: {e}")
        finally:
            await audio_queue.put(None)

    async def transcribe_stream_task():
        try:
            responses = await speech_client.streaming_recognize(requests=audio_generator())
            
            async for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript

                message = {
                    "transcript": transcript,
                    "is_final": result.is_final
                }

                # --- THE FIX ---
                # The client might disconnect at any moment. If the websocket is closed
                # when we try to send, just catch the exception and stop the task.
                try:
                    await websocket.send_text(json.dumps(message))
                except WebSocketDisconnect:
                    print("Could not send message, client disconnected.")
                    break # Exit the 'async for' loop
                except Exception as e:
                    # Catching other potential send errors, e.g., ConnectionClosedError
                    print(f"Error sending message to websocket: {e}")
                    break # Exit the 'async for' loop
                # --- END FIX ---

        except Exception as e:
            # This part handles errors from the Google Speech API itself.
            error_message = f"Error during transcription: {e}"
            print(f"❌ {error_message}")
            try:
                # Attempt to send an error message, but don't crash if this also fails.
                await websocket.send_text(json.dumps({"error": error_message}))
            except Exception:
                # The websocket is likely already closed, so we just ignore the error.
                pass

    try:
        receive_task = asyncio.create_task(receive_audio_task())
        transcribe_task = asyncio.create_task(transcribe_stream_task())
        await asyncio.gather(receive_task, transcribe_task)
    except Exception as e:
        print(f"Error in websocket handler: {e}")
    finally:
        print("Live transcription session finished.")
        active_websockets.discard(websocket)

# --- Background Transcription Task ---
async def transcribe_in_background(task_id: str, session_id: str, db_session_factory):
    print(f"BACKGROUND TASK [{task_id}]: Started for session {session_id}")
    files_to_clean_up = []
    uploaded_files_list = []

    try:
        browser_chunks = sorted(TEMP_DIR.glob(f"{session_id}_chunk_*.webm"))
        if not browser_chunks:
            raise FileNotFoundError("No audio chunks found to process.")

        combined_audio_path = TEMP_DIR / f"{session_id}_combined_raw.webm"
        files_to_clean_up.extend(browser_chunks)
        files_to_clean_up.append(combined_audio_path)

        async with aiofiles.open(combined_audio_path, 'wb') as final_file:
            for chunk_path in browser_chunks:
                async with aiofiles.open(chunk_path, 'rb') as chunk_file:
                    await final_file.write(await chunk_file.read())
        print(f"BACKGROUND TASK [{task_id}]: Combined browser chunks into {combined_audio_path.name}.")

        loop = asyncio.get_event_loop()
        upload_chunk_paths = await loop.run_in_executor(
            None, process_and_split_audio, combined_audio_path, session_id, TEMP_DIR
        )
        files_to_clean_up.extend(upload_chunk_paths)

        uploadable_chunks = [p for p in upload_chunk_paths if 'upload_chunk' in p.name]
        total_chunks = len(uploadable_chunks)

        for i, chunk_path in enumerate(uploadable_chunks):
            print(f"BACKGROUND TASK [{task_id}]: Preparing to upload chunk {i+1}/{total_chunks} ({chunk_path.name})...")

            max_retries = 5
            base_delay = 1.0

            for attempt in range(max_retries):
                try:
                    uploaded_file = await loop.run_in_executor(
                        None, genai.upload_file, str(chunk_path)
                    )
                    uploaded_files_list.append(uploaded_file)
                    print(f"BACKGROUND TASK [{task_id}]: Chunk {i+1} uploaded successfully. File name: {uploaded_file.name}")

                    if i < total_chunks - 1:
                        await asyncio.sleep(1)
                    break

                except exceptions.ServiceUnavailable as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"⚠️ WARNING: Service unavailable on attempt {attempt+1}. Retrying in {delay:.1f} seconds... Error: {e}")
                        await asyncio.sleep(delay)
                    else:
                        print(f"❌ ERROR: Failed to upload chunk {chunk_path.name} after {max_retries} attempts.")
                        raise e

        print(f"BACKGROUND TASK [{task_id}]: All chunks uploaded. Sending to Gemini for verbatim transcription...")
        verbatim_response = await model.generate_content_async([VERBATIM_TRANSCRIPTION_PROMPT] + uploaded_files_list)
        verbatim_transcript = verbatim_response.text.strip()
        print(f"BACKGROUND TASK [{task_id}]: Verbatim transcription successful.")

        print(f"BACKGROUND TASK [{task_id}]: Processing transcript with clinical data analyzer...")
        medical_data = await process_medical_data_with_gemini_async(verbatim_transcript, model)
        print(f"BACKGROUND TASK [{task_id}]: Medical data extraction complete.")

        json_output = combine_transcription_data(verbatim_transcript, medical_data)

        async with db_session_factory() as session:
            new_transcript = Transcript(content=json_output)
            session.add(new_transcript)
            await session.commit()
            print(f"BACKGROUND TASK [{task_id}]: Combined transcript saved to database.")

        tasks[task_id] = {"status": "complete", "result": json_output}

    except Exception as e:
        import traceback
        error_message = f"Error in background task: {e}\n{traceback.format_exc()}"
        print(f"❌ BACKGROUND TASK [{task_id}]: {error_message}")
        tasks[task_id] = {"status": "failed", "error": str(e)}

    finally:
        print(f"BACKGROUND TASK [{task_id}]: Cleaning up files...")
        await loop.run_in_executor(None, cleanup_uploaded_files, uploaded_files_list)

        for p in files_to_clean_up:
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                print(f"Warning: Could not delete file {p}: {e}")
        print(f"BACKGROUND TASK [{task_id}]: Finished.")

# --- API Endpoints ---
@app.post("/api/upload-chunk")
async def upload_chunk(
    session_id: str = Form(...),
    chunk_index: int = Form(...),
    audio_chunk: UploadFile = File(...)
):
    chunk_filename = f"{session_id}_chunk_{chunk_index:04d}.webm"
    file_path = TEMP_DIR / chunk_filename
    
    try:
        # Add timeout handling
        start_time = time.time()
        
        # Write asynchronously with timeout
        async with aiofiles.open(file_path, 'wb') as f:
            content = await asyncio.wait_for(
                audio_chunk.read(),
                timeout=30.0  # Increase timeout to 30 seconds
            )
            await f.write(content)
            
        # Record successful upload time
        chunk_upload_times[session_id][chunk_index] = time.time() - start_time
        
        return {"status": "chunk received", "chunk_index": chunk_index}
    except asyncio.TimeoutError:
        print(f"🚨 Timeout uploading chunk {chunk_index} for session {session_id}")
        return JSONResponse(
            status_code=504,
            content={"error": "Upload timeout", "chunk_index": chunk_index}
        )
    except Exception as e:
        print(f"🚨 Error uploading chunk {chunk_index} for {session_id}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "chunk_index": chunk_index}
        )
@app.post("/api/finalize-recording")
async def finalize_recording(request_data: FinalizeRequest, background_tasks: BackgroundTasks):
    session_id = request_data.session_id
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending"}
    background_tasks.add_task(transcribe_in_background, task_id, session_id, async_session_factory)
    return {"task_id": task_id}

@app.get("/api/transcription-status/{task_id}")
async def get_transcription_status(task_id: str):
    return tasks.get(task_id, {"status": "not_found"})

@app.post("/api/transcripts")
async def create_transcript_manual(
    transcript_data: TranscriptCreate,
    session: AsyncSession = Depends(get_session)
):
    new_transcript = Transcript(content=transcript_data.content)
    session.add(new_transcript)
    await session.commit()
    await session.refresh(new_transcript)
    return new_transcript

@app.get("/api/transcripts")
async def get_transcripts(session: AsyncSession = Depends(get_session)):
    query = select(Transcript).order_by(desc(Transcript.created_at))
    result = await session.execute(query)
    return result.scalars().all()

@app.patch("/api/transcripts/{transcript_id}")
async def update_transcript(
    transcript_id: int,
    transcript_data: TranscriptUpdate,
    session: AsyncSession = Depends(get_session)
):
    result = await session.execute(select(Transcript).where(Transcript.id == transcript_id))
    transcript = result.scalar_one_or_none()
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    transcript.content = transcript_data.content
    await session.commit()
    await session.refresh(transcript)
    return transcript

@app.get("/")
def read_root():
    return {
        "status": "Server is running",
        "architecture": "Asynchronous Background Tasks with FFmpeg (Memory-Efficient)",
        "live_transcription": "Enabled via WebSocket" if speech_client else "Disabled",
        "environment": "Production" if os.getenv("RENDER") else "Development",
        "concurrent_tasks": f"{len(tasks)}/N/A"
    }

# Health check endpoint for Render
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))