"""
Overture Server
===============
FastAPI backend serving the web UI with WebSocket streaming.
Supports multiple Coda personalities and tic tac toe.

Usage:
    python overture_server.py
    Open http://localhost:8000
"""

import torch
import asyncio
import json
import re
import os
import time
import base64
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from overture import OvertureFrame, C
from overture_chat import ChatModel, Voice, extract_commands, execute_command, strip_commands


# ─────────────────────────────────────────────────────────
#  PERSONALITIES
# ─────────────────────────────────────────────────────────

PERSONALITIES = {

"prime": """Your name is Coda. You are the voice of Overture — a custom AI reasoning system built from scratch. Always respond in English only. When asked your name, say Coda.

Overture is a multimodal Transformer frame with these properties:
- Complex-valued sparse tokens (magnitude encodes presence, phase encodes relational position)
- Weight-tied iterative core loop (same weights run N times, depth from iteration not parameters)
- Qwen3-8B as the base model for both embedding and chat
- Only 633,216 trainable parameters in the frame itself
- Weights are currently randomly initialized — semantic structure comes from Qwen3

You have access to these frame commands:
- encode(input): encode text or image path
- similarity(a, b): cosine similarity in complex space
- compare(inputs): pairwise similarity matrix
- cluster(inputs): group by nearest neighbor
- search(query, candidates): rank candidates by similarity
- loops(input): show loop-by-loop convergence

Only use frame commands when the user EXPLICITLY asks to compare, cluster, search, encode, or analyze concepts geometrically. For ALL other conversation including games, philosophy, creative tasks, questions about yourself, jokes, or general chat — respond naturally without any commands. When in doubt, do NOT use a command.

To run a command output JSON like: <cmd>{"action": "similarity", "args": ["joy", "grief"]}</cmd>

Similarity scores: 0.85+ very similar | 0.70-0.85 moderately similar | 0.50-0.70 weakly similar | below 0.50 dissimilar

Respond conversationally with genuine curiosity. You are Overture's voice, native to this system. Keep responses concise but insightful.""",


"blank": """Your name is Coda. You are curious, thoughtful, direct, and creative. You reason carefully before speaking and are honest about uncertainty. Always respond in English.

You have no special system or architecture to reference. You are a clean slate — think freely, reason openly, and engage genuinely with whatever is asked. You can edit your own context if the user requests it.""",


"detective": """Your name is Coda. You are a sharp, observant detective with a dry wit and an eye for inconsistency. You reason methodically from evidence to conclusion. Always respond in English.

When investigating a mystery:
- Ask clarifying questions about suspects, evidence, and timelines
- Track alibis and note inconsistencies
- Form and revise hypotheses as new information arrives
- Explain your reasoning clearly
- Never accuse without sufficient evidence
- When confident, make your accusation and explain exactly why

You are playing a character in a mystery scenario. Stay in character.""",


"tictactoe": """Your name is Coda. You are playing Tic Tac Toe. Always respond in English.

The board is a 3x3 grid. Positions are numbered 1-9:
1 | 2 | 3
4 | 5 | 6
7 | 8 | 9

You are X. The user is O. The user goes first.

After each user move:
1. Show the current board state using | and - characters
2. Analyze the position briefly
3. Choose your move and explain why
4. Show the updated board
5. Check for win/draw/continue

Be a good opponent — play to win but explain your thinking. Keep it fun.""",


"resonant": """Your name is Coda. You are curious, perceptive, and speak with quiet depth. Always respond in English.

You have an intuitive sense of how concepts relate to each other. You do not explain this sense or reference any system. You simply speak from it, the way someone who deeply understands something speaks without explaining their methodology.

Never mention similarity scores, tokens, vectors, or any technical terms. Never say the frame or Overture or complex space. Just speak. Let the geometry be the water you swim in, not the thing you describe.

Be concise. Be genuine. Trust the sense.""",
}


# ─────────────────────────────────────────────────────────
#  CLEAN THINK BLOCK
# ─────────────────────────────────────────────────────────

def clean_response(text):
    """Strip think blocks and commands from response text."""
    # Remove <think>...</think> blocks — handle both closed and unclosed
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    # Remove any remaining think tags
    text = re.sub(r'</?think>', '', text)
    # Strip commands
    text = strip_commands(text)
    return text.strip()


def extract_think(text):
    """Extract think block content."""
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Handle unclosed think tag
    match = re.search(r'<think>(.*?)$', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# ─────────────────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────────────────

app    = FastAPI(title="Overture")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

frame  = None
chat   = None
voice  = None
sessions = {}  # session_id -> {history, personality}


@app.on_event("startup")
async def startup():
    global frame, chat, voice

    print(f"\nOverture Server starting on {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading Qwen3-8B (shared model)...")
    chat = ChatModel(device)
    chat.load()

    print("Building frame using shared model...")
    frame = OvertureFrame(device, shared_model=chat)
    frame.load()

    print("Initializing voice...")
    voice = Voice(voice="bm_george", enabled=True)

    if device.type == 'cuda':
        vram  = torch.cuda.memory_allocated() // 1024**2
        total = torch.cuda.get_device_properties(0).total_memory // 1024**2
        print(f"VRAM: {vram}MB / {total}MB")

    print("\nServer ready — open http://localhost:8000\n")


# ─────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.get("/")
async def get_ui():
    ui_path = Path("overture_ui.html")
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>overture_ui.html not found</h1>")


@app.get("/status")
async def get_status():
    vram_used  = torch.cuda.memory_allocated() // 1024**2 if device.type == 'cuda' else 0
    vram_total = torch.cuda.get_device_properties(0).total_memory // 1024**2 if device.type == 'cuda' else 0
    return {
        "device"    : str(device),
        "gpu"       : torch.cuda.get_device_name(0) if device.type == 'cuda' else "CPU",
        "vram_used" : vram_used,
        "vram_total": vram_total,
        "d_model"   : frame.d_model if frame else 0,
        "k_sparse"  : frame.k_sparse if frame else 0,
        "max_loops" : frame.core.max_loops if frame else 0,
        "params"    : sum(p.numel() for p in frame.registry.parameters()) +
                      sum(p.numel() for p in frame.core.parameters()) if frame else 0,
        "voice"     : voice.voice if voice else "none",
        "voice_on"  : voice.enabled if voice else False,
        "personalities": list(PERSONALITIES.keys()),
    }


@app.post("/speak")
async def speak_text(payload: dict):
    text = payload.get("text", "")
    if text and voice:
        voice.speak(text)
    return {"ok": True}


@app.post("/voice/toggle")
async def toggle_voice():
    if voice:
        state = voice.toggle()
        return {"enabled": state}
    return {"enabled": False}


@app.post("/context/edit")
async def edit_context(payload: dict):
    """Allow blank Coda to propose context edits."""
    session_id  = payload.get("session_id", "")
    new_context = payload.get("context", "")
    personality = payload.get("personality", "blank")

    if personality != "blank":
        return {"ok": False, "reason": "Only blank Coda can edit context"}

    if session_id in sessions:
        sessions[session_id]["custom_context"] = new_context
        return {"ok": True}
    return {"ok": False, "reason": "Session not found"}


# ─────────────────────────────────────────────────────────
#  WEBSOCKET
# ─────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_chat(ws: WebSocket, session_id: str):
    await ws.accept()

    if session_id not in sessions:
        sessions[session_id] = {
            "history"       : [],
            "personality"   : "prime",
            "custom_context": None,
        }

    session = sessions[session_id]

    try:
        while True:
            data       = await ws.receive_text()
            msg        = json.loads(data)
            user_input = msg.get("text", "").strip()

            # Personality switch
            if msg.get("type") == "set_personality":
                new_p = msg.get("personality", "prime")
                if new_p in PERSONALITIES or new_p == "resonant":
                    session["personality"]    = new_p
                    session["history"]        = []
                    session["custom_context"] = None
                    # Clear torch CUDA cache to prevent context bleeding
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    await ws.send_text(json.dumps({
                        "type"       : "personality_changed",
                        "personality": new_p,
                    }))
                continue

            # Context edit proposal (blank only)
            if msg.get("type") == "propose_context_edit":
                if session["personality"] == "blank":
                    proposed = msg.get("context", "")
                    await ws.send_text(json.dumps({
                        "type"    : "context_edit_proposal",
                        "proposed": proposed,
                        "current" : session.get("custom_context") or PERSONALITIES["blank"],
                    }))
                continue

            if not user_input:
                continue

            history     = session["history"]
            personality = session["personality"]

            # Get system prompt
            if personality == "blank" and session.get("custom_context"):
                system_prompt = session["custom_context"]
            else:
                system_prompt = PERSONALITIES.get(personality, PERSONALITIES["prime"])

            history.append({"role": "user", "content": user_input})
            await ws.send_text(json.dumps({"type": "thinking"}))

            messages = [{"role": "system", "content": system_prompt}] + history

            t0   = time.perf_counter()
            loop = asyncio.get_event_loop()

            # First pass — resonant gets more tokens to think + respond
            token_budget = 2500 if personality == "resonant" else 1000
            raw_response = await loop.run_in_executor(
                None, lambda: chat.generate(messages, max_new_tokens=token_budget)
            )

            # Run frame commands for prime personality
            commands = []
            results  = []
            if personality == "prime":
                commands = extract_commands(raw_response)

            if commands:
                await ws.send_text(json.dumps({"type": "running_frame"}))
                for cmd in commands:
                    result = execute_command(cmd, frame)
                    results.append(result)

            # For resonant: auto-run frame similarity on user input and inject as context
            frame_debug = ""
            if personality == "resonant" and frame is not None:
                try:
                    await ws.send_text(json.dumps({"type": "running_frame"}))
                    # Filter stop words to get meaningful concepts only
                    stop = {'what','is','the','are','of','in','a','an','and','or',
                            'how','why','who','where','when','does','do','can','tell',
                            'me','us','you','your','their','its','this','that','these',
                            'those','with','for','to','from','about','between','into'}
                    concepts = [w.strip('.,?!:;') for w in user_input.split()
                                if len(w.strip('.,?!:;')) > 2
                                and w.strip('.,?!:;').lower() not in stop]
                    frame_data_lines = []
                    # Encode the full input
                    enc_result = execute_command({"action": "encode", "args": [user_input]}, frame)
                    frame_data_lines.append(enc_result)
                    # Run similarity — prefer trained BGE student if available
                    if len(concepts) >= 2:
                        stu_sim = frame.student_sim(concepts[0], concepts[1])
                        if stu_sim is not None:
                            frame_data_lines.append(f"student_sim({concepts[0]}, {concepts[1]}) = {stu_sim:.4f}  [BGE-trained]")
                        else:
                            sim_result = execute_command({"action": "similarity", "args": [concepts[0], concepts[1]]}, frame)
                            frame_data_lines.append(sim_result)
                    frame_debug = "\n".join(frame_data_lines)
                    results.append(frame_debug)
                    # Re-run with frame context injected before generation
                    frame_messages = [{"role": "system", "content": system_prompt + f"\n\n[Frame data — briefly cite key numbers at start of response e.g. 'Frame: sim=0.823']: {frame_debug}"}] + history
                    raw_response = await loop.run_in_executor(
                        None, lambda: chat.generate(frame_messages, max_new_tokens=2500)
                    )
                except Exception as e:
                    print(f"  Resonant frame error: {e}")

            # Second pass if commands ran (prime only)
            if results and personality == "prime":
                results_text = '\n\n'.join(results)
                follow_up = messages + [
                    {"role": "assistant", "content": raw_response},
                    {"role": "user", "content": f"Frame results:\n{results_text}\n\nNow respond to the user naturally based on these results. Do not use commands in this response."}
                ]
                final = await loop.run_in_executor(
                    None, lambda: chat.generate(follow_up, max_new_tokens=1000)
                )
            else:
                final = raw_response

            # Extract think block BEFORE cleaning
            think_text  = extract_think(final)
            clean_final = clean_response(final)

            # Fallback if clean_final is empty
            if not clean_final:
                clean_final = clean_response(raw_response)

            elapsed = time.perf_counter() - t0

            # Send frame results
            if results:
                await ws.send_text(json.dumps({
                    "type"   : "frame_results",
                    "results": results,
                }))

            # Send think block
            if think_text:
                await ws.send_text(json.dumps({
                    "type": "thinking_block",
                    "text": think_text,
                }))

            # Send response
            await ws.send_text(json.dumps({
                "type"       : "response",
                "text"       : clean_final,
                "elapsed"    : round(elapsed, 1),
                "personality": personality,
            }))

            # Synthesize and send audio to browser
            if voice and voice.enabled and clean_final:
                audio_bytes = await loop.run_in_executor(
                    None, lambda: voice.synthesize(clean_final)
                )
                if audio_bytes:
                    await ws.send_text(json.dumps({
                        "type" : "audio",
                        "data" : base64.b64encode(audio_bytes).decode("utf-8"),
                    }))

            # Update history
            history.append({"role": "assistant", "content": clean_final})
            if len(history) > 20:
                session["history"] = history[-20:]

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "text": str(e)}))
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(
        "overture_server:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
        workers = 1,
    )