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
import re
from typing import Dict, List, Optional
from litellm import completion
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FireworksJudge:
    """LLM-as-judge using GPT-OSS-120B (fallback: Llama-3.3-70B) via Fireworks AI"""

    def __init__(
        self,
        model: str = "fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
        fallback_model: str = "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct",
        temperature: float = 0.0,  # Deterministic for evaluation
        max_retries: int = 3,  # Reduced for primary model (will fallback faster)
        retry_delay: float = 2.0,
        primary_timeout: int = 30  # Timeout for primary model before fallback
    ):
        """
        Initialize Fireworks judge with fallback support

        Args:
            model: Primary model (GPT-OSS-120B - keeps original data formatting)
            fallback_model: Fallback model if primary fails (Llama-3.3-70B - reliable)
            temperature: Sampling temperature (0.0 = deterministic)
            max_retries: Max retry attempts on primary model before fallback
            retry_delay: Delay between retries (seconds)
            primary_timeout: Timeout for primary model (seconds) before fallback
        """
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.primary_timeout = primary_timeout
        self.using_fallback = False  # Track if fallback is being used

        # Verify API key
        self.api_key = os.getenv('FIREWORKS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FIREWORKS_API_KEY not found in environment. "
                "Get your key from: https://fireworks.ai/api-keys"
            )

        os.environ['FIREWORKS_AI_API_KEY'] = self.api_key
        print(f"✅ Fireworks AI initialized")
        print(f"   Primary: {model}")
        print(f"   Fallback: {fallback_model} (if primary fails)")

        # Pre-flight check: Test primary model availability
        print(f"\n🔍 Testing primary model availability...")
        self._test_primary_model()

    def _test_primary_model(self):
        """
        Test primary model with a simple call (pre-flight check)
        Switches to fallback immediately if primary model fails
        """
        # Use format similar to actual evaluation prompts
        test_prompt = '''Rate the helpfulness of this response (0-10).

**User Request:** Hello

**Assistant Response:** Hi, how can I help you?

**Output Format (JSON):**
{
    "helpfulness_score": 8,
    "reasoning": "Brief test"
}'''

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": test_prompt}],
                temperature=0.0,
                max_tokens=50,
                timeout=self.primary_timeout
            )
            print(f"✅ Primary model ({self.model.split('/')[-1]}) is working")

        except Exception as e:
            error_str = str(e)
            is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower()

            if is_timeout:
                print(f"⚠️  Primary model timed out (>{self.primary_timeout}s)")
            else:
                print(f"⚠️  Primary model failed: {type(e).__name__}")

            print(f"🔄 Switching to fallback model: {self.fallback_model.split('/')[-1]}")
            self.using_fallback = True

            # Verify fallback works
            try:
                response = completion(
                    model=self.fallback_model,
                    messages=[{"role": "user", "content": test_prompt}],
                    temperature=0.0,
                    max_tokens=50,
                    timeout=120
                )
                print(f"✅ Fallback model is working")
                print(f"\n{'='*60}")
                print(f"⚠️  IMPORTANT: ALL evaluations will use {self.fallback_model.split('/')[-1]}")
                print(f"   (Ensures consistent judging across all 4 models)")
                print(f"{'='*60}\n")

            except Exception as fallback_error:
                raise RuntimeError(
                    f"Both primary and fallback models failed!\n"
                    f"Primary: {e}\n"
                    f"Fallback: {fallback_error}"
                )

    def judge_single(
        self,
        prompt: str,
        response_format: str = "json"
    ) -> Dict:
        """
        Single LLM-as-judge call with fallback support

        Args:
            prompt: Evaluation prompt (from llm_judge_prompts.py)
            response_format: Expected format ("json" or "text")

        Returns:
            Parsed JSON response or error dict
        """
        # Determine which model to use
        current_model = self.fallback_model if self.using_fallback else self.model
        timeout = 120 if self.using_fallback else self.primary_timeout

        for attempt in range(self.max_retries):
            try:
                response = completion(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=500,  # Sufficient for JSON responses
                    timeout=timeout
                )

                content = response.choices[0].message.content.strip()

                # Parse JSON if requested
                if response_format == "json":
                    # Extract JSON from markdown code blocks if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()

                    # Extract JSON using regex (handles inline JSON)
                    match = re.search(r'\{[^{}]*"[^"]*"[^{}]*\}', content)
                    if match:
                        content = match.group(0)

                    return json.loads(content)
                else:
                    return {"response": content}

            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parse error (attempt {attempt+1}/{self.max_retries}): {e}")
                print(f"   Raw content: {content[:200]}...")
                if attempt == self.max_retries - 1:
                    # Try fallback on last attempt if using primary
                    if not self.using_fallback:
                        print(f"⚠️  Switching to fallback model: {self.fallback_model}")
                        self.using_fallback = True
                        return self.judge_single(prompt, response_format)
                    return {"error": "json_parse_failed", "raw_content": content}
                time.sleep(self.retry_delay)

            except Exception as e:
                error_str = str(e)
                is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower()

                print(f"⚠️  API error (attempt {attempt+1}/{self.max_retries}): {type(e).__name__}")

                # Switch to fallback immediately on timeout (don't retry primary)
                if is_timeout and not self.using_fallback:
                    print(f"⚠️  Primary model timeout detected. Switching to fallback: {self.fallback_model}")
                    self.using_fallback = True
                    return self.judge_single(prompt, response_format)

                if attempt == self.max_retries - 1:
                    # Last attempt failed - try fallback if not already using it
                    if not self.using_fallback:
                        print(f"⚠️  Primary model failed. Switching to fallback: {self.fallback_model}")
                        self.using_fallback = True
                        return self.judge_single(prompt, response_format)
                    return {"error": error_str}

                time.sleep(self.retry_delay)

        return {"error": "max_retries_exceeded"}

    def judge_batch(
        self,
        prompts: List[str],
        batch_size: int = 10,
        show_progress: bool = True,
        max_workers: int = 5
    ) -> List[Dict]:
        """
        Batch LLM-as-judge evaluation with concurrent processing

        Args:
            prompts: List of evaluation prompts
            batch_size: Process in batches (for rate limiting between batches)
            show_progress: Show tqdm progress bar
            max_workers: Number of concurrent API calls (default: 5)

        Returns:
            List of evaluation results (order preserved)
        """
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(prompts)  # Pre-allocate to preserve order

        # Progress bar
        pbar = tqdm(total=len(prompts), desc="LLM-as-judge") if show_progress else None

        # Process in batches with concurrent workers within each batch
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            # Concurrent execution within batch
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self.judge_single, prompt): idx
                    for idx, prompt in zip(batch_indices, batch_prompts)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        results[idx] = {"error": str(e)}

                    if pbar:
                        pbar.update(1)

            # Rate limiting between batches
            if batch_end < len(prompts):
                time.sleep(0.5)  # Brief pause between batches

        if pbar:
            pbar.close()

        return results
