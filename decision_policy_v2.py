"""Context-correct restricted action and confidence tokens for Qwen3.5."""
import torch

from core import CONFIDENCES
from model import append_tokens


def one_token_suffixes(tokenizer, prefix_text, suffixes):
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    ids = []
    for suffix in suffixes:
        completed = tokenizer.encode(prefix_text + suffix, add_special_tokens=False)
        if completed[:len(prefix)] != prefix or len(completed) != len(prefix) + 1:
            raise ValueError(f'Expected one context token for {prefix_text!r} + {suffix!r}')
        ids.append(completed[-1])
    if len(set(ids)) != len(ids):
        raise ValueError('Candidate tokens are not distinct')
    return prefix, ids


def action_tokens(processor, num_choices):
    return one_token_suffixes(processor.tokenizer, 'Action:',
                              [f' {chr(65+i)}' for i in range(num_choices)])


def confidence_tokens(processor, action_index):
    return one_token_suffixes(processor.tokenizer,
                              f'Action: {chr(65+action_index)}\nConfidence: ',
                              [str(digit) for digit in range(10)])


def candidate_logits(model, batch, prefix_ids, candidate_ids):
    return model(**append_tokens(batch, prefix_ids), use_cache=False,
                 logits_to_keep=1).logits[0, -1, candidate_ids].float()


def greedy_decision(model, processor, batch, num_choices):
    action_prefix, action_ids = action_tokens(processor, num_choices)
    with torch.no_grad():
        action_probs = candidate_logits(model, batch, action_prefix, action_ids).softmax(-1)
        action = int(action_probs.argmax())
        digit_prefix, digit_ids = confidence_tokens(processor, action)
        digit_probs = candidate_logits(model, batch, digit_prefix, digit_ids).softmax(-1)
        digit = int(digit_probs.argmax())
        expected_q = float((digit_probs * torch.tensor(CONFIDENCES, device=digit_probs.device)).sum())
    return {'action_index': action, 'confidence_bin': digit,
            'confidence': CONFIDENCES[digit], 'expected_q': expected_q,
            'action_probabilities': action_probs.tolist(),
            'digit_probabilities': digit_probs.tolist(),
            'output': f'Action: {chr(65+action)}\nConfidence: {digit}'}
