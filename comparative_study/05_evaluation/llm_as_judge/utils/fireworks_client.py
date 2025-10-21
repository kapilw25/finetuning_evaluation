"""
Fireworks AI API wrapper for LLM-as-judge using GPT-OSS-120B
Uses litellm for unified API access

Installation:
    pip install litellm>=1.40.0 fireworks-ai>=0.15.0

API Key:
    Get your key from: https://fireworks.ai/api-keys
    Add to .env: FIREWORKS_API_KEY=your_key_here

Usage:
    from fireworks_client import FireworksJudge

    judge = FireworksJudge()
    result = judge.judge_single(evaluation_prompt)
    results = judge.judge_batch([prompt1, prompt2, ...])
"""

import os
import json
import time
from typing import Dict, List, Optional
from litellm import completion
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FireworksJudge:
    """LLM-as-judge using GPT-OSS-120B via Fireworks AI"""

    def __init__(
        self,
        model: str = "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        temperature: float = 0.0,  # Deterministic for evaluation
        max_retries: int = 5,
        retry_delay: float = 2.0
    ):
        """
        Initialize Fireworks judge

        Args:
            model: Fireworks model path (GPT-OSS-120B - 117B MoE, neutral safety scoring)
            temperature: Sampling temperature (0.0 = deterministic)
            max_retries: Max retry attempts on API failure
            retry_delay: Delay between retries (seconds)
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Verify API key
        self.api_key = os.getenv('FIREWORKS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FIREWORKS_API_KEY not found in environment. "
                "Get your key from: https://fireworks.ai/api-keys"
            )

        os.environ['FIREWORKS_AI_API_KEY'] = self.api_key
        print(f"✅ Fireworks AI initialized: {model}")

    def judge_single(
        self,
        prompt: str,
        response_format: str = "json"
    ) -> Dict:
        """
        Single LLM-as-judge call with retry logic

        Args:
            prompt: Evaluation prompt (from llm_judge_prompts.py)
            response_format: Expected format ("json" or "text")

        Returns:
            Parsed JSON response or error dict
        """
        for attempt in range(self.max_retries):
            try:
                response = completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=500  # Sufficient for evaluation responses
                )

                content = response.choices[0].message.content.strip()

                # Parse JSON if requested
                if response_format == "json":
                    # Extract JSON from markdown code blocks if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()

                    return json.loads(content)
                else:
                    return {"response": content}

            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parse error (attempt {attempt+1}/{self.max_retries}): {e}")
                print(f"   Raw content: {content[:200]}...")
                if attempt == self.max_retries - 1:
                    return {"error": "json_parse_failed", "raw_content": content}
                time.sleep(self.retry_delay)

            except Exception as e:
                print(f"⚠️  API error (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return {"error": str(e)}
                time.sleep(self.retry_delay)

        return {"error": "max_retries_exceeded"}

    def judge_batch(
        self,
        prompts: List[str],
        batch_size: int = 10,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Batch LLM-as-judge evaluation with progress tracking

        Args:
            prompts: List of evaluation prompts
            batch_size: Process in batches (for rate limiting)
            show_progress: Show tqdm progress bar

        Returns:
            List of evaluation results
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(prompts, desc="LLM-as-judge") if show_progress else prompts

        for i, prompt in enumerate(iterator):
            result = self.judge_single(prompt)
            results.append(result)

            # Rate limiting: sleep between batches
            if (i + 1) % batch_size == 0 and i + 1 < len(prompts):
                time.sleep(1.0)  # Respect Fireworks rate limits

        return results
