from pathlib import Path

import sentencepiece
import torch
from huggingface_hub import hf_hub_download

from moshi.models import LMGen, loaders
from moshi.models import lm as lm_module
from moshi.offline import _get_voice_prompt_dir, wrap_with_system_tags

PROMPT = (
    "You are a wise and friendly teacher. Answer questions or provide advice in a "
    "clear and engaging way."
)
VOICE = "NATF2.pt"


def distribution(logits, temperature, top_k):
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    values, indices = torch.topk(probabilities, top_k, dim=-1)
    output = torch.zeros_like(probabilities)
    output.scatter_(-1, indices, values)
    return output / output.sum(dim=-1, keepdim=True)


class Probe:
    def __init__(self, repo, device):
        self.metadata = {
            "repo": repo,
            "prompt": PROMPT,
            "voice": VOICE,
            "audio_temperature": 0.8,
            "audio_top_k": 250,
            "text_temperature": 0.7,
            "text_top_k": 25,
        }
        self.device = torch.device(device)
        self.mimi = loaders.get_mimi(
            hf_hub_download(repo, loaders.MIMI_NAME), self.device
        )
        self.tokenizer = sentencepiece.SentencePieceProcessor(
            hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME)
        )
        self.lm = loaders.get_moshi_lm(
            hf_hub_download(repo, loaders.MOSHI_NAME),
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.lm_gen = LMGen(
            self.lm,
            device=self.device,
            sample_rate=self.mimi.sample_rate,
            frame_rate=self.mimi.frame_rate,
            return_logits=True,
            temp=0.8,
            temp_text=0.7,
            top_k=250,
            top_k_text=25,
        )
        voice_dir = Path(_get_voice_prompt_dir(None, repo))
        self.lm_gen.load_voice_prompt_embeddings(str(voice_dir / VOICE))
        self.lm_gen.text_prompt_tokens = self.tokenizer.encode(
            wrap_with_system_tags(PROMPT)
        )
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.zero = torch.zeros(1, 1, self.frame_size, device=self.device)
        self.needed = self.lm.num_codebooks - lm_module.AUDIO_TOKENS_PER_STREAM - 1
        self.pad_token = self.lm_gen.zero_text_code
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def reset(self):
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.lm_gen.step_system_prompts(self.mimi)
        self.mimi.reset_streaming()

    def step(self, pcm=None, forced=None, paired=False):
        self.logits = None
        sample_token = lm_module.sample_token
        first = True

        def sample(logits, use_sampling, temperature, top_k):
            nonlocal first
            if first:
                self.logits = logits.clone()
                first = False
            token = sample_token(logits, use_sampling, temperature, top_k)
            return token

        codes = self.mimi.encode(self.zero if pcm is None else pcm)
        text_token = None
        if forced is not None:
            text_token = torch.tensor([forced], device=self.device)
        if paired:
            lm_module.sample_token = sample
        try:
            _, logits = self.lm_gen.step(codes[:, : self.needed], text_token=text_token)
        finally:
            lm_module.sample_token = sample_token
        if logits is None:
            return None, None
        state = self.lm_gen._streaming_state
        position = (state.offset - 1) % state.cache.shape[2]
        token = int(state.cache[0, 0, position].item())
        text_logits = logits[0] if self.logits is None else self.logits
        probabilities = distribution(
            text_logits, self.lm_gen.temp_text, self.lm_gen.top_k_text
        )[0, 0, 0]
        return token, probabilities

    def text(self, tokens):
        return self.tokenizer.decode(
            [token for token in tokens if token is not None and token >= 4]
        )
