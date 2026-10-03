"""Two restricted VideoJev decision tokens: action letter and confidence digit."""
import torch

from core import CONFIDENCES, parse_output
from model import append_tokens


def token_ids(processor, num_choices):
    tokenizer = processor.tokenizer
    actions = [tokenizer.encode(chr(65+i), add_special_tokens=False) for i in range(num_choices)]
    digits = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)]
    if any(len(x) != 1 for x in actions+digits):
        raise ValueError('Each constrained action and digit must be one token')
    return [x[0] for x in actions], [x[0] for x in digits]


def logits_for(model, batch, prefix_ids, candidate_ids):
    return model(**append_tokens(batch, prefix_ids), use_cache=False,
                 logits_to_keep=1).logits[0, -1, candidate_ids].float()


def action_prefix(processor):
    return processor.tokenizer.encode('Action: ', add_special_tokens=False)


def digit_prefix(processor, action_index):
    return processor.tokenizer.encode(f'Action: {chr(65+action_index)}\nConfidence: ',
                                      add_special_tokens=False)


def greedy_decision(model, processor, batch, num_choices):
    actions, digits = token_ids(processor, num_choices)
    with torch.no_grad():
        action_probs = logits_for(model, batch, action_prefix(processor), actions).softmax(-1)
        action_index = int(action_probs.argmax())
        digit_probs = logits_for(model, batch, digit_prefix(processor, action_index), digits).softmax(-1)
        digit_index = int(digit_probs.argmax())
        expected_q = float((digit_probs * torch.tensor(CONFIDENCES, device=digit_probs.device)).sum())
    output = f'Action: {chr(65+action_index)}\nConfidence: {digit_index}'
    decision = parse_output(output, num_choices)
    if not decision.valid_format:
        raise RuntimeError('Constrained output was invalid')
    return {
        'output': output, 'action_index': action_index, 'confidence_bin': digit_index,
        'confidence': decision.confidence, 'expected_q': expected_q,
        'action_probabilities': action_probs.tolist(),
        'digit_probabilities': digit_probs.tolist(),
    }
