"""
Pronunciation Correction Agent
===============================
Agent-based pronunciation correction system that combines:
1. WavLM fine-tuned model for phoneme-level assessment
2. GPT-4o for intelligent feedback, learning plan generation, and dialogue
3. User memory for tracking progress across sessions

Can be called as an OpenClaw skill or standalone via CLI/API.

Usage:
    # CLI
    python pronunciation_agent.py --audio audio.mp3 --text "Hello, Peter." --user student_001

    # Python API
    from pronunciation_agent import PronunciationAgent
    agent = PronunciationAgent()
    result = agent.run(audio="audio.mp3", text="Hello, Peter.", user_id="student_001")

    # FastAPI server
    python pronunciation_agent.py --serve --port 8000
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from openai import OpenAI

# ============================================================
# User Memory (JSON-file based, per-user)
# ============================================================
MEMORY_DIR = Path(__file__).parent / "user_memory"


class UserMemory:
    """Simple file-based memory for tracking user progress."""

    def __init__(self, user_id, memory_dir=MEMORY_DIR):
        self.user_id = user_id
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.file_path = self.memory_dir / f"{user_id}.json"
        self.data = self._load()

    def _load(self):
        if self.file_path.exists():
            with open(self.file_path) as f:
                return json.load(f)
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "sessions": [],
            "weak_phonemes": {},     # phone -> {count, total, error_rate}
            "overall_stats": {
                "total_sessions": 0,
                "total_phonemes": 0,
                "total_errors": 0,
                "avg_score": 0,
            },
            "current_plan": None,
        }

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def add_session(self, evaluation_result, feedback, plan):
        """Record a session's evaluation, feedback, and plan."""
        session = {
            "timestamp": datetime.now().isoformat(),
            "text": evaluation_result.get("text", ""),
            "overall_score": evaluation_result.get("overall_score", 0),
            "n_phonemes": evaluation_result.get("n_phonemes", 0),
            "n_errors": evaluation_result.get("n_errors", 0),
            "error_phonemes": [],
            "feedback_summary": feedback[:200] if feedback else "",
        }

        # Track error phonemes
        for word in evaluation_result.get("words", []):
            for ph in word.get("phonemes", []):
                phone = ph["phone"]

                # Update weak phonemes stats
                if phone not in self.data["weak_phonemes"]:
                    self.data["weak_phonemes"][phone] = {
                        "count": 0, "errors": 0, "total_score": 0
                    }
                wp = self.data["weak_phonemes"][phone]
                wp["count"] += 1
                wp["total_score"] += ph.get("score", 0)
                if ph.get("error", False):
                    wp["errors"] += 1
                    session["error_phonemes"].append({
                        "phone": phone,
                        "word": word["word"],
                        "score": ph.get("score", 0),
                        "pherr_prob": ph.get("pherr_prob", 0),
                    })

        self.data["sessions"].append(session)

        # Update overall stats
        stats = self.data["overall_stats"]
        stats["total_sessions"] += 1
        stats["total_phonemes"] += session["n_phonemes"]
        stats["total_errors"] += session["n_errors"]
        all_scores = [s["overall_score"] for s in self.data["sessions"]]
        stats["avg_score"] = round(sum(all_scores) / len(all_scores), 1)

        # Update plan
        if plan:
            self.data["current_plan"] = plan

        self.save()

    def get_weak_phonemes(self, min_count=2, min_error_rate=0.3):
        """Get phonemes the user consistently struggles with."""
        weak = []
        for phone, stats in self.data["weak_phonemes"].items():
            if stats["count"] >= min_count:
                error_rate = stats["errors"] / stats["count"]
                avg_score = stats["total_score"] / stats["count"]
                if error_rate >= min_error_rate:
                    weak.append({
                        "phone": phone,
                        "error_rate": round(error_rate, 2),
                        "avg_score": round(avg_score, 1),
                        "occurrences": stats["count"],
                    })
        return sorted(weak, key=lambda x: -x["error_rate"])

    def get_recent_sessions(self, n=5):
        return self.data["sessions"][-n:]

    def get_summary(self):
        """Get a text summary for LLM context."""
        stats = self.data["overall_stats"]
        if stats["total_sessions"] == 0:
            return "This is a new user with no previous sessions."

        weak = self.get_weak_phonemes()
        weak_str = ", ".join(f"/{p['phone']}/ (error rate {p['error_rate']:.0%})" for p in weak[:5])

        recent = self.get_recent_sessions(3)
        recent_str = "; ".join(
            f"'{s['text']}' score={s['overall_score']}" for s in recent
        )

        return (
            f"User has completed {stats['total_sessions']} sessions. "
            f"Average score: {stats['avg_score']}/100. "
            f"Total phonemes practiced: {stats['total_phonemes']}, "
            f"total errors: {stats['total_errors']}. "
            f"Weak phonemes: {weak_str or 'none identified yet'}. "
            f"Recent sessions: {recent_str}."
        )


# ============================================================
# Pronunciation Agent
# ============================================================
SYSTEM_PROMPT = """You are an expert English pronunciation coach for children learning English.
Your role is to analyze pronunciation assessment results and provide helpful, encouraging feedback.

You will receive:
1. A structured evaluation log with phoneme-level scores and error flags
2. The user's learning history (if available)

Your tasks:
- Explain which sounds were pronounced incorrectly in simple, child-friendly language
- Give specific tips on how to improve each problematic sound
- Be encouraging and positive — celebrate what they got right
- Generate a short learning plan for next practice

Always respond in the language the user message is in (Chinese or English).
Keep feedback concise and actionable."""

FEEDBACK_PROMPT_TEMPLATE = """Here is the pronunciation evaluation result:

Text: "{text}"
Overall Score: {overall_score}/100
Errors: {n_errors}/{n_phonemes}

Word-by-word breakdown:
{word_details}

User history:
{user_summary}

Please provide:
1. **Feedback**: Explain what was good and what needs improvement. For each error, explain how to correctly pronounce that sound.
2. **Learning Plan**: A short plan (2-3 items) for what to practice next.
3. **Encouragement**: A positive, motivating message.

Respond in Chinese (the student is a Chinese child learning English)."""


def format_word_details(result):
    """Format evaluation result into readable text for LLM."""
    lines = []
    for word in result.get("words", []):
        status = "ERROR" if word["has_error"] else "OK"
        lines.append(f"\n  [{status}] {word['word']} (score={word['score']})")
        for ph in word["phonemes"]:
            err = " <-- ERROR" if ph.get("error") else ""
            lines.append(
                f"    /{ph['phone']}/  score={ph.get('score', 'N/A')}  "
                f"pherr={ph.get('pherr_prob', 'N/A')}{err}"
            )
    return "\n".join(lines)


class PronunciationAgent:
    """
    Main agent that orchestrates:
    1. Pronunciation assessment (WavLM model)
    2. LLM analysis and feedback (GPT-4o)
    3. User memory management
    """

    def __init__(self, model="gpt-4o", device=None, memory_dir=MEMORY_DIR,
                 api_key=None, base_url=None):
        self.llm_model = model
        self.device = device
        self.memory_dir = memory_dir
        self._api_key = api_key
        self._base_url = base_url
        self._assessor = None
        self._client = None

    def _load_assessor(self):
        if self._assessor is not None:
            return
        from pipeline_v2 import PronunciationAssessorV2
        self._assessor = PronunciationAssessorV2.from_pretrained(device=self.device)

    def _load_client(self):
        if self._client is not None:
            return
        # Support multiple LLM providers via env vars
        # Priority: explicit args > provider-specific env > OPENAI_API_KEY
        api_key = self._api_key
        base_url = self._base_url

        if not api_key:
            # Auto-detect provider from model name
            if self.llm_model.startswith("qwen"):
                api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
                if not base_url:
                    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            else:
                api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "API key not found. Set OPENAI_API_KEY (for GPT) or "
                "DASHSCOPE_API_KEY / QWEN_API_KEY (for Qwen)"
            )

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)

    def _call_llm(self, user_message, system_prompt=SYSTEM_PROMPT):
        """Call GPT-4o for feedback generation."""
        self._load_client()
        response = self._client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def run(self, audio, text, user_id="default"):
        """
        Run the full pronunciation correction pipeline.

        Args:
            audio: Path to audio file
            text: Reference text
            user_id: User identifier for memory tracking

        Returns:
            {
                "evaluation": { ... },      # Raw assessment result
                "feedback": "...",           # LLM-generated feedback
                "plan": { ... },            # Learning plan
                "user_stats": { ... },      # Updated user statistics
            }
        """
        # Step 1: Assess pronunciation
        print("Step 1: Assessing pronunciation...", file=sys.stderr)
        self._load_assessor()
        evaluation = self._assessor.assess(audio, text)

        if "error" in evaluation:
            return {"evaluation": evaluation, "error": evaluation["error"]}

        # Step 2: Load user memory
        print("Step 2: Loading user memory...", file=sys.stderr)
        memory = UserMemory(user_id, self.memory_dir)
        user_summary = memory.get_summary()

        # Step 3: Generate feedback via LLM
        print("Step 3: Generating feedback...", file=sys.stderr)
        word_details = format_word_details(evaluation)
        prompt = FEEDBACK_PROMPT_TEMPLATE.format(
            text=evaluation["text"],
            overall_score=evaluation["overall_score"],
            n_errors=evaluation["n_errors"],
            n_phonemes=evaluation["n_phonemes"],
            word_details=word_details,
            user_summary=user_summary,
        )

        feedback = self._call_llm(prompt)

        # Step 4: Extract plan (ask LLM for structured plan)
        plan_prompt = (
            f"Based on this feedback:\n{feedback}\n\n"
            f"And the user's weak phonemes: {memory.get_weak_phonemes()}\n\n"
            "Generate a JSON learning plan with this exact format:\n"
            '{"focus_phonemes": ["ph1", "ph2"], '
            '"practice_words": ["word1", "word2", "word3"], '
            '"tips": ["tip1", "tip2"]}'
            "\nReturn ONLY the JSON, no other text."
        )
        try:
            plan_raw = self._call_llm(plan_prompt)
            # Extract JSON from response
            plan_raw = plan_raw.strip()
            if plan_raw.startswith("```"):
                plan_raw = plan_raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            plan = json.loads(plan_raw)
        except (json.JSONDecodeError, Exception):
            plan = {"focus_phonemes": [], "practice_words": [], "tips": []}

        # Step 5: Update memory
        print("Step 4: Updating user memory...", file=sys.stderr)
        memory.add_session(evaluation, feedback, plan)

        result = {
            "evaluation": evaluation,
            "feedback": feedback,
            "plan": plan,
            "user_stats": memory.data["overall_stats"],
            "weak_phonemes": memory.get_weak_phonemes(),
        }

        return result


# ============================================================
# FastAPI Server
# ============================================================
def create_app():
    """Create FastAPI app for serving the agent as an API."""
    try:
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import JSONResponse
    except ImportError:
        print("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart",
              file=sys.stderr)
        sys.exit(1)

    app = FastAPI(
        title="Pronunciation Correction Agent",
        description="Phoneme-level pronunciation assessment with AI-powered feedback",
        version="2.0",
    )
    # Auto-detect LLM provider from env
    llm_model = os.environ.get("LLM_MODEL", "gpt-4o")
    agent = PronunciationAgent(model=llm_model)

    @app.post("/assess")
    async def assess(
        audio: UploadFile = File(...),
        text: str = Form(...),
        user_id: str = Form(default="default"),
    ):
        """Assess pronunciation and return feedback."""
        # Save uploaded audio to temp file
        import tempfile
        suffix = Path(audio.filename).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = agent.run(audio=tmp_path, text=text, user_id=user_id)
            return JSONResponse(content=result)
        finally:
            os.unlink(tmp_path)

    @app.get("/user/{user_id}")
    async def get_user(user_id: str):
        """Get user's learning history and stats."""
        memory = UserMemory(user_id)
        return JSONResponse(content={
            "stats": memory.data["overall_stats"],
            "weak_phonemes": memory.get_weak_phonemes(),
            "recent_sessions": memory.get_recent_sessions(),
            "current_plan": memory.data.get("current_plan"),
        })

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# ============================================================
# Pretty Print
# ============================================================
def print_result(result):
    """Print agent result in a readable format."""
    eval_r = result["evaluation"]
    print(f"\n{'='*60}")
    print(f"Text: \"{eval_r['text']}\"")
    print(f"Overall Score: {eval_r['overall_score']}/100  "
          f"(errors: {eval_r['n_errors']}/{eval_r['n_phonemes']})")
    print(f"{'='*60}")

    for wd in eval_r.get("words", []):
        status = "\u2717" if wd["has_error"] else "\u2713"
        print(f"\n  {status} {wd['word']:<15s}  score={wd['score']:5.1f}  "
              f"errors={wd['n_errors']}/{wd['n_phonemes']}")
        for ph in wd["phonemes"]:
            marker = " \u2190 ERROR" if ph["error"] else ""
            print(f"      /{ph['phone']:<4s}/  score={ph['score']:5.1f}  "
                  f"pherr={ph['pherr_prob']:.2f}{marker}")

    print(f"\n{'='*60}")
    print("AI Feedback:")
    print(f"{'='*60}")
    print(result["feedback"])

    if result.get("plan"):
        print(f"\n{'='*60}")
        print("Learning Plan:")
        print(f"{'='*60}")
        plan = result["plan"]
        if plan.get("focus_phonemes"):
            print(f"  Focus: {', '.join('/' + p + '/' for p in plan['focus_phonemes'])}")
        if plan.get("practice_words"):
            print(f"  Practice words: {', '.join(plan['practice_words'])}")
        if plan.get("tips"):
            for tip in plan["tips"]:
                print(f"  - {tip}")

    if result.get("weak_phonemes"):
        print(f"\n  Weak phonemes (historical):")
        for wp in result["weak_phonemes"][:5]:
            print(f"    /{wp['phone']}/  error_rate={wp['error_rate']:.0%}  "
                  f"avg_score={wp['avg_score']}  ({wp['occurrences']} occurrences)")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Pronunciation Correction Agent")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--text", type=str, help="Reference text")
    parser.add_argument("--user", type=str, default="default", help="User ID")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="LLM model (default: gpt-4o, e.g. qwen-plus, qwen-turbo)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key (overrides env vars)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL (auto-detected for qwen* models)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0, cpu, mps)")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return

    if not args.audio or not args.text:
        parser.error("--audio and --text are required (or use --serve)")

    agent = PronunciationAgent(
        model=args.model, device=args.device,
        api_key=args.api_key, base_url=args.base_url,
    )
    result = agent.run(audio=args.audio, text=args.text, user_id=args.user)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
