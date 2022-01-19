import re
import torch
import numpy as np
import string
from word2number.w2n import word_to_num
from allennlp.nn.util import replace_masked_values
from itertools import permutations


def get_mask(mask_templates, numbers, ops):
    """get mask for next token"""
    with torch.no_grad():
        outmasks = torch.zeros((numbers.shape[0], numbers.shape[1], mask_templates.shape[-1]), device=numbers.device)
        mask_indices = (numbers > ops + 1).long().unsqueeze(-1).expand_as(outmasks)
        return torch.gather(mask_templates, 1, mask_indices, out=outmasks)

import pickle
def beam_search(K, log_probs, number_mask, op_mask, END, NUM_OPS):
    """beam search algorithm"""
    with torch.no_grad():
        # log_probs : (batch,seqlen,vocab)
        # number_mask : (batch,vocab)
        # op_mask : (batch,vocab)
        (batch_size, maxlen, V) = log_probs.shape
        
        # possible masks
        # mask_templates[0] : #nums = #ops + 1
        # mask_templates[1] : #nums > #ops + 1
        mask_templates = torch.zeros((batch_size, 2,V), device=log_probs.device)
        mask_templates[number_mask.unsqueeze(1).expand_as(mask_templates).byte()] = 1
        mask_templates[:,0,:][op_mask.byte()] = 0
        mask_templates[:,1,END] = 0
        mask_templates[:,0,END] = 1
        
        # expanded log_probs (for convinience)
        # log_probs2 : (batch,seqlen,K,vocab)
        log_probs2 = log_probs.unsqueeze(2).expand(-1,-1,K,-1)
        
        # #numbers so far
        numbers = torch.zeros((batch_size, K),device=log_probs.device).int()
        # #ops so far
        ops = torch.zeros((batch_size, K), device=log_probs.device).int()
              
        # best sequences and scores so far
        best_seqs = [[-100]] * batch_size
        best_scores = [-np.inf] * batch_size
        
        # initial mask
        init_mask = number_mask.clone()
        
        # first term
        scores = replace_masked_values(log_probs[:,0], init_mask, -np.inf)
        kscores, kidxs = scores.topk(K, dim=-1, largest=True, sorted=True)
          
        # update numbers and ops
        numbers += 1
        
        # K hypothesis for each batch
        # (batch, K, seqlen)
        seqs = kidxs.unsqueeze(-1)
        
        for t in range(1,maxlen):
            mask = get_mask(mask_templates, numbers, ops)
            scores = replace_masked_values(log_probs2[:,t], mask, -np.inf)
            tscores = (scores + kscores.unsqueeze(-1)).view(batch_size, -1)
            kscores, kidxs = tscores.topk(K, dim=-1, largest=True, sorted=True)
            # prev_hyps : (batch,K)
            prev_hyps = kidxs / V
            
            # next_tokens : (batch,K,1)
            next_tokens = (kidxs % V).unsqueeze(-1)
            
            if prev_hyps.max() >= K:
                print("problem")
                prev_hyps = torch.clamp(prev_hyps, max = K -1, min=0)
                
            # check how many have ended
            ended = next_tokens == END
            # update best_seqs and scores as needed
            for batch in range(batch_size):
                if ended[batch].sum() > 0:
                    ends = ended[batch].nonzero()
                    idx = ends[0,0]
                    token = next_tokens[batch, idx].cpu().item()
                    score = kscores[batch, idx].cpu().item()
                    if score > best_scores[batch]:
                        best_seqs[batch] = seqs[batch, prev_hyps[batch, idx]]
                        best_scores[batch] = score
                    for end in ends:
                        kscores[batch, end[0]] = -np.inf
            
            # update numbers and ops
            numbers = torch.gather(numbers, 1, prev_hyps)
            ops = torch.gather(ops, 1, prev_hyps)
            is_num = (next_tokens.squeeze() >= NUM_OPS).int()
            numbers += is_num.int()
            ops += (1 - is_num.int())
            
            # update seqs
            new_seqs = torch.gather(seqs, 1, prev_hyps.unsqueeze(-1).expand_as(seqs))
            seqs = torch.cat([new_seqs, next_tokens], -1)
#             print(seqs)
#         with open("output.txt", "a") as myfile:
#             print("best_seqs : ", best_seqs, file=myfile)
#             print("seqs : ", seqs, file=myfile)
        return best_seqs, best_scores
