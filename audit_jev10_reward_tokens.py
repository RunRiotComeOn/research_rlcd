"""Check complete-context action/digit token IDs before reward training."""
import json

from transformers import AutoProcessor

from decision_policy_v2 import action_tokens, confidence_tokens
from train_jev10_reward_compare import reward


def main():
    processor = AutoProcessor.from_pretrained(
        '/pfs/siqingyi/code/token_credit/Valen/models/Qwen3.5-2B', local_files_only=True)
    tokenizer = processor.tokenizer
    for text in ('Action: A\nConfidence:', 'Action: A\nConfidence: 0',
                 'Action: A\nConfidence: ', 'Action: A\nConfidence: 1'):
        ids = tokenizer.encode(text, add_special_tokens=False)
        print(json.dumps({'text': text, 'ids': ids,
                          'pieces': [tokenizer.decode([token]) for token in ids]}), flush=True)
    ap, action_ids = action_tokens(processor, 26)
    digits = {}
    for action in range(26):
        dp, digit_ids = confidence_tokens(processor, action)
        digits[chr(65+action)] = {'prefix_tokens': len(dp), 'ids': digit_ids}
    for y in (0,1):
        for q in (0,.05,.35,.95,1):
            old = reward(y,q,'old_brier')
            new = reward(y,q,'proposed')
            if abs((new-old)-.2*y) > 1e-9:
                raise AssertionError((y,q,old,new))
    print(json.dumps({'action_prefix_tokens': len(ap),
                      'action_ids': action_ids,
                      'action_decoded': [tokenizer.decode([token]) for token in action_ids],
                      'digit_ids_A': digits['A']['ids'],
                      'digit_decoded_A': [tokenizer.decode([token]) for token in digits['A']['ids']],
                      'all_actions_audited': len(digits), 'reward_identity': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
