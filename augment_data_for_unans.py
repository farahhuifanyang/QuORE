from itertools import permutations
import copy
import random

def aug_data(converted_data_QM, dataset_name, mode):
  data_to_aug = converted_data_QM
  aug_data = copy.deepcopy(data_to_aug)
  aug_rel_ind = 0
  has_aug = False

  for key, ins in aug_data.items():
    args = ins['args']
    args_len = len(args)
    args_pos_perm = permutations(range(args_len), 2)
    args_pos_perm_list = list(args_pos_perm) + [[0, 0]]
    random.shuffle(args_pos_perm_list)
    has_aug = False

    qa_pairs = ins['qa_pairs']
    exist_arg_pair = []
    for pair in qa_pairs:
      pair_pos = pair['arg_pair']
      exist_arg_pair.append(pair_pos)
    
    for p in args_pos_perm_list:
      p_list = list(p)
      if p_list in exist_arg_pair:
        continue
      if has_aug or p_list == [0, 0]:
        break
      
      aug_pair = {}
      aug_pair['arg_pair'] = p_list
      aug_pair['query_id'] = "AUG_" + dataset_name + "_" + mode + "_REL_" + str(aug_rel_ind)       
      answer = {}
      answer['spans'] = []
      aug_pair['answer'] = answer

      arg0 = args[p_list[0]]
      arg1 = args[p_list[1]]
      # question = "What is the relation of \"" + arg0 + "\" to \"" + arg1 + "\" ?"          ##########################
      question = "\"" + arg0 + "\"" + " ? " + "\"" + arg1 + "\""
      aug_pair['question'] = question

      qa_pairs.append(aug_pair)
      aug_rel_ind += 1
      has_aug = True
    
    ins['qa_pairs'] = qa_pairs
  
  return aug_data