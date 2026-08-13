import torch

from moshi.models import LMGen, loaders
from moshi.utils.sampling import sample_token


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
            "audio_temperature": 0.8,
            "audio_top_k": 250,
            "text_temperature": 0.7,
            "text_top_k": 25,
        }
        checkpoint = loaders.CheckpointInfo.from_hf_repo(repo)
        self.tokenizer = checkpoint.get_text_tokenizer()
        self.mimi = checkpoint.get_mimi(device=device)
        self.lm = checkpoint.get_moshi(device=device, dtype=torch.bfloat16)
        self.logits = None
        self.sample = None
        self.forced = None
        self.block = False
        self.paired = False
        self.condition = False
        self.blocked = False
        self.proposal = None
        self.lm_gen = LMGen(
            self.lm,
            on_text_logits_hook=self.logits_hook,
            on_text_hook=self.text_hook,
            **checkpoint.lm_gen_config,
        )
        self.lm_gen.temp = 0.8
        self.lm_gen.temp_text = 0.7
        self.lm_gen.top_k = 250
        self.lm_gen.top_k_text = 25
        self.device = device
        self.pad_token = 0
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.zero = torch.zeros(1, 1, self.frame_size, device=device)
        self.needed = self.lm.num_codebooks - self.lm.dep_q - 1
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def logits_hook(self, logits):
        self.logits = logits.clone()
        self.blocked = False
        if self.forced is not None:
            logits.fill_(-torch.inf)
            logits[..., self.forced] = 0
        elif self.block or self.paired:
            cpu_state = torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(self.device)
            token = int(
                sample_token(
                    logits.float(),
                    True,
                    self.lm_gen.temp_text,
                    self.lm_gen.top_k_text,
                ).item()
            )
            self.proposal = token
            torch.random.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(cuda_state, self.device)
            self.blocked = self.block and token >= 4
            if self.blocked:
                logits.fill_(-torch.inf)
                logits[..., 0] = 0
        elif self.condition:
            logits[..., 4:] = -torch.inf

    def text_hook(self, token):
        self.sample = int(token.item())

    def step(self, pcm=None, forced=None, block=False, paired=False, condition=False):
        self.logits = None
        self.sample = None
        self.proposal = None
        self.forced = forced
        self.block = block
        self.paired = paired
        self.condition = condition
        codes = self.mimi.encode(self.zero if pcm is None else pcm)
        self.lm_gen.step(codes[:, : self.needed])
        self.forced = None
        self.block = False
        self.paired = False
        self.condition = False
        if self.logits is None:
            return self.sample, None
        probabilities = distribution(
            self.logits, self.lm_gen.temp_text, self.lm_gen.top_k_text
        )[0, 0, 0]
        return self.sample, probabilities

    def reset(self):
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    def text(self, tokens):
        return self.tokenizer.decode(
            [token for token in tokens if token is not None and token >= 4]
        )
