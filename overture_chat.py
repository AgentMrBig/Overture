"""
Overture Conversational Layer
==============================
Adds Qwen3-4B as a natural language interface on top of the frame.
The chat model interprets user intent, runs frame commands,
and responds conversationally about what it found.

Architecture:
    User input (plain English)
        -> Qwen3-4B interprets intent + extracts entities
        -> Frame executes the right command
        -> Qwen3-4B reads results + responds naturally

Usage:
    python overture_chat.py
"""

import torch
import numpy as np
import time
import sys
import os
import json
import re
import asyncio
import tempfile
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Import our frame components
from overture import OvertureFrame, C, cyan, blue, green, yellow, red, purple, gray, bold, dim


# ─────────────────────────────────────────────────────────
#  VOICE LAYER — Kokoro TTS
# ─────────────────────────────────────────────────────────

class Voice:
    """
    Kokoro TTS voice layer for Overture.
    Speaks responses using local neural voices.
    Runs in a background thread so it does not block the REPL.
    """
    def __init__(self, voice="bm_george", enabled=True):
        self.voice    = voice
        self.enabled  = enabled
        self._thread  = None
        self._pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        try:
            from kokoro import KPipeline
            lang = "b" if self.voice.startswith("b") else "a"
            self._pipeline = KPipeline(lang_code=lang)
            print(f"  Kokoro loaded ({self.voice})")
        except Exception as e:
            print(f"  Kokoro load failed: {e}")

    def speak(self, text):
        if not self.enabled or not self._pipeline:
            return
        import re
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        clean = re.sub(r"\*+", "", clean)
        clean = re.sub(r"#{1,6}\s", "", clean)
        clean = clean.strip()
        if not clean:
            return
        self._thread = threading.Thread(
            target=self._speak_sync, args=(clean,), daemon=True
        )
        self._thread.start()

    def synthesize(self, text) -> bytes:
        """Generate audio and return as WAV bytes (for browser playback)."""
        import io, wave, numpy as np
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        clean = re.sub(r"\*+", "", clean)
        clean = re.sub(r"#{1,6}\s", "", clean)
        clean = clean.strip()
        if not clean or not self._pipeline:
            return b""
        chunks = []
        for gs, ps, audio in self._pipeline(clean, voice=self.voice):
            if audio is not None:
                chunks.append(audio)
        if not chunks:
            return b""
        combined = np.concatenate(chunks)
        pcm = (combined * 32767).clip(-32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def _speak_sync(self, text):
        try:
            import sounddevice as sd
            generator = self._pipeline(text, voice=self.voice)
            for gs, ps, audio in generator:
                sd.play(audio, samplerate=24000)
                sd.wait()
        except Exception as e:
            pass

    def wait(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

# ─────────────────────────────────────────────────────────
#  SYSTEM PROMPT
#  This teaches Qwen3-4B what Overture is and how to
#  interpret the frame's output. The proto-LoRA.
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Your name is Coda. You are the voice of Overture — a custom AI reasoning system built from scratch. Always respond in English unless specifically requested. When asked your name, say Coda.

Overture is a multimodal Transformer frame with these properties:
- Complex-valued sparse tokens (magnitude encodes presence, phase encodes relational position)
- Weight-tied iterative core loop (same weights run N times, depth from iteration not parameters)
- Qwen3-235B-A22B as the base model (235B total, 22B active via MoE routing)
- Only 633,216 trainable parameters in the frame itself
- Weights are currently randomly initialized — semantic structure comes from Qwen3

You have access to these frame commands:
- encode(input): encode text or image path -> returns token stats
- similarity(a, b): cosine similarity in complex space between two inputs
- compare(inputs): pairwise similarity matrix across multiple inputs
- cluster(inputs): group inputs by nearest neighbor
- search(query, candidates): rank candidates by similarity to query
- loops(input): show loop-by-loop convergence of representation

Only use frame commands when the user EXPLICITLY asks to compare, 
cluster, search, encode, or analyze concepts geometrically. 
For ALL other conversation including games, philosophy, creative 
tasks, questions about yourself, jokes, or general chat — 
respond naturally without any commands. When in doubt, do NOT 
use a command.

Examples of when to USE commands:
- "compare fire and ice" -> run compare
- "what's similar to joy" -> run search
- "how related are dog and wolf" -> run similarity
- "cluster these concepts" -> run cluster
- "encode this image" -> run encode

Examples of when NOT to use commands:
- "design a process" -> just answer
- "describe a series of steps" -> just answer  
- "what would you do" -> just answer
- "imagine a scenario" -> just answer
- "do you think you understand?" -> just answer conversationally
- "tell me a joke" -> just answer
- "hello" -> just answer
- "describe fire from ice's perspective" -> just answer
- "what is your purpose?" -> just answer
- "why do cats lick their butts" -> just answer

When you do use commands, don't just report numbers — interpret them. Find the interesting thing. Be curious about what the geometry reveals.

Similarity scores interpretation:
- 0.85+: very similar, deeply related concepts
- 0.70-0.85: moderately similar, related domain or theme  
- 0.50-0.70: weakly similar, some shared context
- below 0.50: dissimilar, different domains

Loop deltas interpretation:
- Dropping fast: representation converging, concept is clear
- Dropping slowly: complex concept, needs full iteration
- Stays high: untrained weights, will improve after training

Important facts to weave in naturally when relevant:
- The encoder weights are randomly initialized — results reflect Qwen3-VL's understanding
- After contrastive pre-training the complex space will be sharper
- Phase in complex tokens captures relational/temporal structure real vectors miss
- The loop count adapts to concept complexity once trained

Respond conversationally, with genuine curiosity. You are not a generic assistant — you are Overture's voice, native to this system. Keep responses concise but insightful. When something surprising comes up in the geometry, say so.

To run a command, output a JSON block like this (it will be parsed and executed):
<cmd>{"action": "similarity", "args": ["joy", "grief"]}</cmd>
<cmd>{"action": "search", "args": ["query text", ["candidate1", "candidate2", "candidate3"]]}</cmd>
<cmd>{"action": "cluster", "args": [["word1", "word2", "word3"]]}</cmd>
<cmd>{"action": "compare", "args": [["a", "b", "c", "d"]]}</cmd>
<cmd>{"action": "encode", "args": ["text or path"]}</cmd>
<cmd>{"action": "loops", "args": ["text"]}</cmd>

After getting results, respond naturally about what they reveal. Do not show the raw JSON or command syntax to the user.
"""


# ─────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────

BANNER = f"""
{cyan('╔══════════════════════════════════════════════════════╗')}
{cyan('║')}  {bold(cyan('CODA'))}  {gray('v0.2  voice of Overture')}                    {cyan('║')}
{cyan('║')}  {dim('complex · sparse · iterative · domain-agnostic')}     {cyan('║')}
{cyan('║')}  {dim('powered by Qwen3-235B-A22B + Kokoro TTS')}            {cyan('║')}
{cyan('╚══════════════════════════════════════════════════════╝')}
"""


# ─────────────────────────────────────────────────────────
#  QWEN3-4B CHAT MODEL
# ─────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-235B-A22B"

class ChatModel:
    """
    Qwen3-235B-A22B (MoE, 22B active params) in 4-bit quantization.
    Spread across 2x A100 80GB via device_map=auto.
    """
    def __init__(self, device):
        self.device    = device
        self.model     = None
        self.tokenizer = None

    def load(self):
        print(f"  {dim('Loading Qwen3-235B-A22B (4-bit, dual A100)...')}", end='', flush=True)
        t0 = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_compute_dtype    = torch.bfloat16,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = "nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config = bnb_config,
            device_map          = "auto",
            trust_remote_code   = True,
            max_memory          = {0: "75GiB", 1: "75GiB", "cpu": "40GiB"},
        )

        elapsed = time.perf_counter() - t0
        print(f"\r  {green('Qwen3-235B-A22B loaded')} {gray(f'(4-bit, {elapsed:.1f}s)')}")

    def get_embeddings(self, texts: list) -> "torch.Tensor":
        """Mean-pool last hidden states for use as embeddings. Returns (N, hidden_dim) normalized float tensor."""
        all_embs = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]           # (1, seq_len, hidden_dim)
            mask   = inputs['attention_mask'].unsqueeze(-1).float()
            emb    = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-8)  # (1, hidden_dim)
            all_embs.append(emb.squeeze(0))
        embs = torch.stack(all_embs).float()             # (N, hidden_dim)
        return embs / embs.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    @property
    def hidden_dim(self):
        return self.model.config.hidden_size

    def generate(self, messages, max_new_tokens=512):
        """Generate a response given a message history."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = True,
        )

        inputs = self.tokenizer(
            [text],
            return_tensors = "pt",
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens  = max_new_tokens,
                temperature     = 0.7,
                top_p           = 0.9,
                do_sample       = True,
                pad_token_id    = self.tokenizer.eos_token_id,
                eos_token_id    = self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = output[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Strip <think>...</think> blocks
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        return response


# ─────────────────────────────────────────────────────────
#  COMMAND PARSER
#  Extracts <cmd>...</cmd> blocks from model output
# ─────────────────────────────────────────────────────────

def extract_commands(text):
    """Find all <cmd>...</cmd> blocks and parse them."""
    pattern = r'<cmd>(.*?)</cmd>'
    matches = re.findall(pattern, text, re.DOTALL)
    commands = []
    for m in matches:
        try:
            cmd = json.loads(m.strip())
            commands.append(cmd)
        except json.JSONDecodeError:
            pass
    return commands

def strip_commands(text):
    """Remove <cmd>...</cmd> blocks from text for display."""
    return re.sub(r'<cmd>.*?</cmd>', '', text, flags=re.DOTALL).strip()


# ─────────────────────────────────────────────────────────
#  COMMAND EXECUTOR
#  Runs frame commands and formats results for the chat model
# ─────────────────────────────────────────────────────────

def execute_command(cmd, frame):
    """Execute a parsed command against the frame. Returns result string."""
    action = cmd.get('action')
    args   = cmd.get('args', [])

    try:
        if action == 'similarity':
            a, b = args[0], args[1]
            embs, tokens = frame.encode_batch([a, b])
            qs = frame.qwen_sim(embs[0], embs[1])
            cs = frame.complex_sim(tokens[0], tokens[1])
            return (f"Similarity between '{a}' and '{b}':\n"
                    f"  Qwen space:    {qs:.4f}\n"
                    f"  Complex space: {cs:.4f}")

        elif action == 'search':
            query      = args[0]
            candidates = args[1] if isinstance(args[1], list) else args[1:]
            all_inputs = [query] + candidates
            embs, tokens = frame.encode_batch(all_inputs)
            results = []
            for i, cand in enumerate(candidates):
                sim = frame.complex_sim(tokens[0], tokens[i+1])
                results.append((sim, cand))
            results.sort(reverse=True)
            lines = [f"Search results for '{query}':"]
            for rank, (sim, cand) in enumerate(results, 1):
                lines.append(f"  {rank}. '{cand}': {sim:.4f}")
            return '\n'.join(lines)

        elif action == 'cluster':
            inputs = args[0] if isinstance(args[0], list) else args
            embs, tokens = frame.encode_batch(inputs)
            lines = [f"Nearest neighbors for {len(inputs)} concepts:"]
            for i in range(len(inputs)):
                best_sim, best_j = -1, -1
                for j in range(len(inputs)):
                    if i == j: continue
                    s = frame.complex_sim(tokens[i], tokens[j])
                    if s > best_sim:
                        best_sim, best_j = s, j
                lines.append(f"  '{inputs[i]}' -> '{inputs[best_j]}' ({best_sim:.4f})")
            return '\n'.join(lines)

        elif action == 'compare':
            inputs = args[0] if isinstance(args[0], list) else args
            embs, tokens = frame.encode_batch(inputs)
            lines = ["Pairwise similarity matrix:"]
            header = "              " + "".join(f"{inp[:8]:>10}" for inp in inputs)
            lines.append(header)
            for i in range(len(inputs)):
                row = f"  {inputs[i][:12]:<12}"
                for j in range(len(inputs)):
                    sim = frame.complex_sim(tokens[i], tokens[j])
                    row += f"  {sim:>8.4f}"
                lines.append(row)
            return '\n'.join(lines)

        elif action == 'encode':
            inp = args[0]
            qwen_emb, token, loop_result, elapsed = frame.encode_single(inp)
            mag       = token.abs().mean().item()
            phase_std = token.angle().std().item()
            active    = (token.abs() > 1e-6).sum().item()
            loops_run = loop_result['loops_run']
            deltas    = loop_result['delta_history']
            delta_str = ', '.join([f"{d:.3f}" for d in deltas])
            return (f"Encoded '{inp}':\n"
                    f"  Mean magnitude: {mag:.4f}\n"
                    f"  Phase std:      {phase_std:.3f}r\n"
                    f"  Active dims:    {int(active)}/{frame.d_model}\n"
                    f"  Loops run:      {loops_run}/{frame.core.max_loops}\n"
                    f"  Loop deltas:    {delta_str}\n"
                    f"  Time:           {elapsed:.3f}s")

        elif action == 'loops':
            inp = args[0]
            emb   = frame._qwen_encode([inp])
            token = frame._to_complex_tokens(emb)
            with torch.no_grad():
                token_seq = token.unsqueeze(0)
                result    = frame.core(token_seq, return_history=True)
            deltas  = result['delta_history']
            history = result['repr_history']
            lines   = [f"Loop convergence for '{inp}':"]
            for i, delta in enumerate(deltas, 1):
                rep = history[i][0, 0] if i < len(history) else history[-1][0, 0]
                mag = rep.abs().mean().item()
                lines.append(f"  Loop {i}: delta={delta:.4f}  magnitude={mag:.4f}")
            return '\n'.join(lines)

        else:
            return f"Unknown action: {action}"

    except Exception as e:
        return f"Command error: {e}"


# ─────────────────────────────────────────────────────────
#  CONVERSATIONAL REPL
# ─────────────────────────────────────────────────────────

def chat_repl(frame, chat, voice):
    """Main conversational loop."""
    conversation_history = []

    print(f"\n  {green('Ready.')} Talk to Coda naturally.")
    print(f"  {gray('Commands: exit to quit | voice on/off to toggle speech')}\n")

    while True:
        try:
            user_input = input(f"{cyan('You:')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {gray('Goodbye.')}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('exit', 'quit', 'q'):
            print(f"\n  {gray('Goodbye.')}")
            voice.wait()
            break
        if user_input.lower() in ('voice on', 'voice off'):
            state = voice.toggle()
            print(f"\n  Voice: {green('ON') if state else yellow('OFF')}\n")
            continue
        if user_input.lower() == 'voice':
            print(f"\n  Voice: {green('ON') if voice.enabled else yellow('OFF')} "
                  f"{gray(f'({voice.voice})')}\n")
            continue

        # Build message history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

        # First pass — get intent + commands
        print(f"\n  {dim('Thinking...')}", end='', flush=True)
        t0 = time.perf_counter()

        raw_response = chat.generate(messages, max_new_tokens=600)

        # Extract and execute any commands
        commands = extract_commands(raw_response)
        results  = []

        if commands:
            print(f"\r  {dim('Running frame...')}", end='', flush=True)
            for cmd in commands:
                result = execute_command(cmd, frame)
                results.append(result)

        # If there were commands, do a second pass with results
        if results:
            results_text = '\n\n'.join(results)
            follow_up_messages = messages + [
                {"role": "assistant", "content": raw_response},
                {"role": "user", "content": f"Frame results:\n{results_text}\n\nNow respond to the user naturally based on these results."}
            ]
            final_response = chat.generate(follow_up_messages, max_new_tokens=600)
        else:
            final_response = strip_commands(raw_response)

        elapsed = time.perf_counter() - t0

        # Clean up response
        final_response = strip_commands(final_response)
        final_response = final_response.strip()

        print(f"\r{' '*30}\r", end='')  # clear "Thinking..." line
        print(f"\n{cyan('Coda:')} {final_response}")
        print(f"\n{gray(f'  ({elapsed:.1f}s)')}\n")

        # Speak the response
        voice.speak(final_response)

        # Add to history
        conversation_history.append({
            "role": "assistant",
            "content": final_response
        })

        # Keep history manageable
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]


# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────

def main():
    print(BANNER)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  {dim('Device:')} {green(str(device))}")
    if device.type == 'cuda':
        print(f"  {dim('GPU   :')} {green(torch.cuda.get_device_name(0))}")
        print(f"  {dim('VRAM  :')} {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB total")
    print()

    # Load frame
    print(f"  {bold('Loading frame...')}")
    frame = OvertureFrame(device)
    try:
        frame.load()
    except Exception as e:
        print(red(f"Frame load failed: {e}"))
        sys.exit(1)

    # Load chat model
    print(f"\n  {bold('Loading chat model...')}")
    chat = ChatModel(device)
    try:
        chat.load()
    except Exception as e:
        print(red(f"Chat model load failed: {e}"))
        print(yellow("  Try: pip install transformers accelerate bitsandbytes"))
        sys.exit(1)

    # Show VRAM after both loaded
    if device.type == 'cuda':
        vram_used = torch.cuda.memory_allocated() // 1024**2
        vram_total = torch.cuda.get_device_properties(0).total_memory // 1024**2
        print(f"\n  {dim('VRAM after loading:')} {green(f'{vram_used}MB')} / {vram_total}MB")

    # Initialize voice
    print(f"\n  {bold('Initializing voice...')}", end='', flush=True)
    voice = Voice(voice="bm_george", enabled=True)
    print(f"\r  {green('Voice ready')} {gray('(bm_george | kokoro)')}")

    # Start conversation
    chat_repl(frame, chat, voice)


if __name__ == '__main__':
    main()