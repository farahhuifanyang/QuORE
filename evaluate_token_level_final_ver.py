import collections
import json
import jsonlines


def compute_em(pred: list, truth: list) -> int:
    return int(str(pred) == str(truth))


def compute_f1_p_r(pred_toks: list, gold_toks: list) -> list:
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        f1 = precision = recall = int(gold_toks == pred_toks)
        return f1, precision, recall
    if num_same == 0:
        return 0, 0, 0
    
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def token_level_evaluate(rel_id: str, pred: list, truth: list) -> dict:
    em = f1 = p = r = 0

    if 'AUG' not in rel_id: # answerable
        if truth == []:
            # em = f1 = p = r = 0
            if pred != []:
                print(rel_id, pred, truth)
        else:
            em = compute_em(pred, truth)
            f1, p, r = compute_f1_p_r(pred, truth)

    else: # unanswerable
        em = compute_em(pred, truth)
        f1, p, r = compute_f1_p_r(pred, truth)
    
    evaluation = {'em': em, 'f1': f1, 'p': p, 'r': r}
    return evaluation


def calculate_global_eva_result(all_evaluations: list) -> dict:
    cnt_ins = len(all_evaluations)

    # sum the values with same keys
    counter = collections.Counter()
    for res in all_evaluations: 
        counter.update(res)

    dict_counter = dict(counter)
    dict_counter['em'] /= float(cnt_ins)
    dict_counter['f1'] /= float(cnt_ins)
    dict_counter['p'] /= float(cnt_ins)
    dict_counter['r'] /= float(cnt_ins)

    return dict_counter


def eva_lorem_output(file_name: str):
    with jsonlines.open(file_name + '.jsonl', 'r') as file:
        lorem_output = [line for line in file]
    
    all_evaluations = []
    answerable_evaluations = []
    unanswerable_evaluations = []
    for ins in lorem_output:
        rel_id = str(ins['rel_id'])
        lorem_pred = ins['lorem_pred']
        truth_rel = ins['truth_rel']

        evaluation = token_level_evaluate(rel_id, lorem_pred, truth_rel)
        ins['evaluation'] = evaluation
        all_evaluations.append(evaluation)
        if 'AUG' not in rel_id:
            answerable_evaluations.append(evaluation)
        else:
            unanswerable_evaluations.append(evaluation)
        # print(ins)

    global_eva_result = calculate_global_eva_result(all_evaluations)
    print(global_eva_result)
    global_ans_eva_result = calculate_global_eva_result(answerable_evaluations)
    print(global_ans_eva_result)
    if unanswerable_evaluations:
        global_unans_eva_result = calculate_global_eva_result(unanswerable_evaluations)
        print(global_unans_eva_result)

    with jsonlines.open(file_name + '_evaluation.jsonl', 'w') as writer:
        writer.write_all(lorem_output)


def eva_musst_output(file_name: str):
    with jsonlines.open(file_name + '.jsonl', 'r') as file:
        musst_output = [line for line in file]
    
    new_musst_output = []
    all_evaluations = []
    sse_evaluations = []
    qasl_evaluations = []

    # For SAOKE evaluation of multi-span
    saoke_1_span_evaluations = []
    saoke_2_span_evaluations = []
    saoke_3_span_evaluations = []
    saoke_4_span_evaluations = []
    saoke_5_span_evaluations = []
    cnt_saoke_1_span_prediction_len = 0
    cnt_saoke_2_span_prediction_len = 0
    cnt_saoke_3_span_prediction_len = 0
    cnt_saoke_4_span_prediction_len = 0
    cnt_saoke_5_span_prediction_len = 0
    # saoke_6_span_evaluations = []
    # saoke_7_span_evaluations = []
    for ins in musst_output:
        rel_id = str(ins['query_id'])
        predicted_ability = ins['predicted_ability']
        
        musst_pred = []
        truth_rel = []
        if 'SAOKE' not in file_name and 'COER' not in file_name:
            if predicted_ability == 'passage_span_extraction':
                musst_pred = ins['answer']['value'].split()
            else:
                for span in ins['answer']['value']:
                    musst_pred += span.split()

            for span in ins['maximizing_ground_truth']['spans']:
                truth_rel += span.split()
        
        else:
            if predicted_ability == 'passage_span_extraction':
                musst_pred = list(ins['answer']['value'])
            else:
                for span in ins['answer']['value']:
                    musst_pred += list(span)

            for span in ins['maximizing_ground_truth']['spans']:
                truth_rel += list(span)


        evaluation = token_level_evaluate(rel_id, musst_pred, truth_rel)
        if predicted_ability == 'passage_span_extraction':
            sse_evaluations.append(evaluation)
        elif predicted_ability == 'multiple_spans':
            qasl_evaluations.append(evaluation)
        else:
            print("Not supporting ability.")
        all_evaluations.append(evaluation)

        # For SAOKE evaluation of multi-span
        if 'SAOKE' in file_name:
            if len(truth_rel) == 1:
                saoke_1_span_evaluations.append(evaluation)
            elif len(truth_rel) == 2:
                saoke_2_span_evaluations.append(evaluation)
            elif len(truth_rel) == 3:
                saoke_3_span_evaluations.append(evaluation)
            elif len(truth_rel) == 4:
                saoke_4_span_evaluations.append(evaluation)
            elif len(truth_rel) == 5:
                saoke_5_span_evaluations.append(evaluation)
            # elif len(truth_rel) == 6:
            #     saoke_6_span_evaluations.append(evaluation)
            # elif len(truth_rel) == 7:
            #     saoke_7_span_evaluations.append(evaluation)
            else:
                print('SAOKE: Exceed max length of span.')

        if 'SAOKE' in file_name:
            if len(musst_pred) == 1:
                cnt_saoke_1_span_prediction_len += len(musst_pred)
            elif len(musst_pred) == 2:
                cnt_saoke_2_span_prediction_len += len(musst_pred)
            elif len(musst_pred) == 3:
                cnt_saoke_3_span_prediction_len += len(musst_pred)
            elif len(musst_pred) == 4:
                cnt_saoke_4_span_prediction_len += len(musst_pred)
            elif len(musst_pred) == 5:
                cnt_saoke_5_span_prediction_len += len(musst_pred)
            # elif len(musst_pred) == 6:
            #     saoke_6_span_predictions.append(evaluation)
            # elif len(musst_pred) == 7:
            #     saoke_7_span_predictions.append(evaluation)
            else:
                print('SAOKE: Exceed max length of span.')

        new_ins = {}
        new_ins['rel_id'] = rel_id
        new_ins['musst_pred'] = musst_pred
        new_ins['truth_rel'] = truth_rel
        new_ins['evaluation'] = evaluation
        new_ins['passage_id'] = ins['passage_id']        
        new_ins['predicted_ability'] = predicted_ability    
        print(new_ins)
        new_musst_output.append(new_ins)

    global_eva_result = calculate_global_eva_result(all_evaluations)
    print("global_eva_result: ")
    print(json.dumps(global_eva_result, indent = 4))
    if sse_evaluations:
        global_sse_eva_result = calculate_global_eva_result(sse_evaluations)
        print("global_sse_eva_result: ")
        print(json.dumps(global_sse_eva_result, indent = 4))
    if qasl_evaluations:
        global_qasl_eva_result = calculate_global_eva_result(qasl_evaluations)
        print("global_qasl_eva_result: ")
        print(json.dumps(global_qasl_eva_result, indent = 4))

    if saoke_1_span_evaluations:
        global_saoke_1_span_eva_result = calculate_global_eva_result(saoke_1_span_evaluations)
        print("global_saoke_1_span_eva_result: ")
        print(len(saoke_1_span_evaluations))
        print(float(cnt_saoke_1_span_prediction_len) / len(saoke_1_span_evaluations))
        print(json.dumps(global_saoke_1_span_eva_result, indent = 4))
    if saoke_2_span_evaluations:
        global_saoke_2_span_eva_result = calculate_global_eva_result(saoke_2_span_evaluations)
        print("global_saoke_2_span_eva_result: ")
        print(len(saoke_2_span_evaluations))
        print(float(cnt_saoke_2_span_prediction_len) / len(saoke_2_span_evaluations))
        print(json.dumps(global_saoke_2_span_eva_result, indent = 4))
    if saoke_3_span_evaluations:
        global_saoke_3_span_eva_result = calculate_global_eva_result(saoke_3_span_evaluations)
        print("global_saoke_3_span_eva_result: ")
        print(len(saoke_3_span_evaluations))
        print(float(cnt_saoke_3_span_prediction_len) / len(saoke_3_span_evaluations))
        print(json.dumps(global_saoke_3_span_eva_result, indent = 4))
    if saoke_4_span_evaluations:
        global_saoke_4_span_eva_result = calculate_global_eva_result(saoke_4_span_evaluations)
        print("global_saoke_4_span_eva_result: ")
        print(len(saoke_4_span_evaluations))
        print(float(cnt_saoke_4_span_prediction_len) / len(saoke_4_span_evaluations))
        print(json.dumps(global_saoke_4_span_eva_result, indent = 4))
    if saoke_5_span_evaluations:
        global_saoke_5_span_eva_result = calculate_global_eva_result(saoke_5_span_evaluations)
        print("global_saoke_5_span_eva_result: ")
        print(len(saoke_5_span_evaluations))
        print(float(cnt_saoke_5_span_prediction_len) / len(saoke_5_span_evaluations))
        print(json.dumps(global_saoke_5_span_eva_result, indent = 4))
    # if saoke_6_span_evaluations:
    #     global_saoke_6_span_eva_result = calculate_global_eva_result(saoke_6_span_evaluations)
    #     print("global_saoke_6_span_eva_result: ")
    #     print(len(saoke_6_span_evaluations) / float(len(all_evaluations)))
    #     print(json.dumps(global_saoke_6_span_eva_result, indent = 4))
    # if saoke_7_span_evaluations:
    #     global_saoke_7_span_eva_result = calculate_global_eva_result(saoke_7_span_evaluations)
    #     print("global_saoke_7_span_eva_result: ")
    #     print(len(saoke_7_span_evaluations) / float(len(all_evaluations)))
    #     print(json.dumps(global_saoke_7_span_eva_result, indent = 4))
    # print(
    #     (len(saoke_1_span_evaluations) +
    #     len(saoke_2_span_evaluations) +
    #     len(saoke_3_span_evaluations) +
    #     len(saoke_4_span_evaluations) +
    #     len(saoke_5_span_evaluations) +
    #     len(saoke_6_span_evaluations) +
    #     len(saoke_7_span_evaluations)) / float(len(all_evaluations))
    # )

    with jsonlines.open(file_name + '_evaluation.jsonl', 'w') as writer:
        writer.write_all(new_musst_output)


def evaluate_ans_and_unans(file_path: str) -> dict:
    with jsonlines.open(file_path + '.jsonl', 'r') as file:
        predictions = [line for line in file]
    
    results = {
        'both_heads': {
            'answerable': {
                'percent': 0.0,
                'count': 0,
                'evaluation': {
                    'f1': 0.0,
                    'em': 0.0,
                    'p': 0.0,
                    'r': 0.0
                }
            },
            'unanswerable': {
                'percent': 0.0,
                'count': 0,
                'evaluation': {
                    'f1': 0.0,
                    'em': 0.0,
                    'p': 0.0,
                    'r': 0.0
                }
            },
            'whole_data': {
                'percent': 1.0,
                'count': 0,
                'evaluation': {
                    'f1': 0.0,
                    'em': 0.0,
                    'p': 0.0,
                    'r': 0.0
                }
            }
        },

        'sse_head': {
            'answerable': {
                'percent': 0.0,
                'count': 0,
                'evaluation': {
                    'f1': 0.0,
                    'em': 0.0,
                    'p': 0.0,
                    'r': 0.0
                }
            },
            'unanswerable': {
                'percent': 0.0,
                'count': 0,
                'evaluation': {
                    'f1': 0.0,
                    'em': 0.0,
                    'p': 0.0,
                    'r': 0.0
                }
            },
            'whole_data': {
                'percent': 0.0,
                'count': 0,
                'evaluation': {
                    'f1': 0.0,
                    'em': 0.0,
                    'p': 0.0,
                    'r': 0.0
                }
            }
        },

        'qasl_head': {
            'answerable': {
                'percent': 0.0,
                'count': 0,
                'evaluation': {
                    'f1': 0.0,
                    'em': 0.0,
                    'p': 0.0,
                    'r': 0.0
                }
            },
            'unanswerable': {
                'percent': 0.0,
                'count': 0,
                'evaluation': {
                    'f1': 0.0,
                    'em': 0.0,
                    'p': 0.0,
                    'r': 0.0
                }
            },
            'whole_data': {
                'percent': 0.0,
                'count': 0,
                'evaluation': {
                    'f1': 0.0,
                    'em': 0.0,
                    'p': 0.0,
                    'r': 0.0
                }
            }
        }
    }

    for pred in predictions:
        head = pred['predicted_ability']
        em = pred['evaluation']['em']
        f1 = pred['evaluation']['f1']
        p = pred['evaluation']['p']
        r = pred['evaluation']['r']

        is_answerable = True
        rel_id = pred['rel_id']
        if 'AUG' in rel_id:
            is_answerable = False
        
        results['both_heads']['whole_data']['evaluation']['f1'] += f1
        results['both_heads']['whole_data']['evaluation']['em'] += em
        results['both_heads']['whole_data']['evaluation']['p'] += p
        results['both_heads']['whole_data']['evaluation']['r'] += r
        results['both_heads']['whole_data']['count'] += 1

        if is_answerable:
            results['both_heads']['answerable']['evaluation']['f1'] += f1
            results['both_heads']['answerable']['evaluation']['em'] += em
            results['both_heads']['answerable']['evaluation']['p'] += p
            results['both_heads']['answerable']['evaluation']['r'] += r
            results['both_heads']['answerable']['count'] += 1
        else:
            results['both_heads']['unanswerable']['evaluation']['f1'] += f1
            results['both_heads']['unanswerable']['evaluation']['em'] += em
            results['both_heads']['unanswerable']['evaluation']['p'] += p
            results['both_heads']['unanswerable']['evaluation']['r'] += r
            results['both_heads']['unanswerable']['count'] += 1
        
        if head == 'passage_span_extraction':
            results['sse_head']['whole_data']['evaluation']['f1'] += f1
            results['sse_head']['whole_data']['evaluation']['em'] += em
            results['sse_head']['whole_data']['evaluation']['p'] += p
            results['sse_head']['whole_data']['evaluation']['r'] += r
            results['sse_head']['whole_data']['count'] += 1

            if is_answerable:
                results['sse_head']['answerable']['count'] += 1
                results['sse_head']['answerable']['evaluation']['f1'] += f1
                results['sse_head']['answerable']['evaluation']['em'] += em
                results['sse_head']['answerable']['evaluation']['p'] += p
                results['sse_head']['answerable']['evaluation']['r'] += r
            else:
                results['sse_head']['unanswerable']['count'] += 1
                results['sse_head']['unanswerable']['evaluation']['f1'] += f1
                results['sse_head']['unanswerable']['evaluation']['em'] += em
                results['sse_head']['unanswerable']['evaluation']['p'] += p
                results['sse_head']['unanswerable']['evaluation']['r'] += r
        elif head == 'multiple_spans':
            results['qasl_head']['whole_data']['evaluation']['f1'] += f1
            results['qasl_head']['whole_data']['evaluation']['em'] += em
            results['qasl_head']['whole_data']['evaluation']['p'] += p
            results['qasl_head']['whole_data']['evaluation']['r'] += r
            results['qasl_head']['whole_data']['count'] += 1

            if is_answerable:
                results['qasl_head']['answerable']['count'] += 1
                results['qasl_head']['answerable']['evaluation']['f1'] += f1
                results['qasl_head']['answerable']['evaluation']['em'] += em
                results['qasl_head']['answerable']['evaluation']['p'] += p
                results['qasl_head']['answerable']['evaluation']['r'] += r
            else:
                results['qasl_head']['unanswerable']['count'] += 1
                results['qasl_head']['unanswerable']['evaluation']['f1'] += f1
                results['qasl_head']['unanswerable']['evaluation']['em'] += em
                results['qasl_head']['unanswerable']['evaluation']['p'] += p
                results['qasl_head']['unanswerable']['evaluation']['r'] += r
    
    cnt_pred = results['both_heads']['whole_data']['count']

    results['both_heads']['whole_data']['evaluation']['f1'] /= cnt_pred
    results['both_heads']['whole_data']['evaluation']['em'] /= cnt_pred
    results['both_heads']['whole_data']['evaluation']['p'] /= cnt_pred
    results['both_heads']['whole_data']['evaluation']['r'] /= cnt_pred

    cnt_both_heads_ans = results['both_heads']['answerable']['count']
    if cnt_both_heads_ans:
        results['both_heads']['answerable']['evaluation']['f1'] /= cnt_both_heads_ans
        results['both_heads']['answerable']['evaluation']['em'] /= cnt_both_heads_ans
        results['both_heads']['answerable']['evaluation']['p'] /= cnt_both_heads_ans
        results['both_heads']['answerable']['evaluation']['r'] /= cnt_both_heads_ans
        results['both_heads']['answerable']['percent'] = cnt_both_heads_ans / cnt_pred

    cnt_both_heads_unans = results['both_heads']['unanswerable']['count']
    if cnt_both_heads_unans:
        results['both_heads']['unanswerable']['evaluation']['f1'] /= cnt_both_heads_unans
        results['both_heads']['unanswerable']['evaluation']['em'] /= cnt_both_heads_unans
        results['both_heads']['unanswerable']['evaluation']['p'] /= cnt_both_heads_unans
        results['both_heads']['unanswerable']['evaluation']['r'] /= cnt_both_heads_unans
        results['both_heads']['unanswerable']['percent'] = cnt_both_heads_unans / cnt_pred

    cnt_sse_head_ans = results['sse_head']['answerable']['count']
    if cnt_sse_head_ans:
        results['sse_head']['answerable']['evaluation']['f1'] /= cnt_sse_head_ans
        results['sse_head']['answerable']['evaluation']['em'] /= cnt_sse_head_ans
        results['sse_head']['answerable']['evaluation']['p'] /= cnt_sse_head_ans
        results['sse_head']['answerable']['evaluation']['r'] /= cnt_sse_head_ans
        results['sse_head']['answerable']['percent'] = cnt_sse_head_ans / cnt_pred

    cnt_sse_head_unans = results['sse_head']['unanswerable']['count']
    if cnt_sse_head_unans:
        results['sse_head']['unanswerable']['evaluation']['f1'] /= cnt_sse_head_unans
        results['sse_head']['unanswerable']['evaluation']['em'] /= cnt_sse_head_unans
        results['sse_head']['unanswerable']['evaluation']['p'] /= cnt_sse_head_unans
        results['sse_head']['unanswerable']['evaluation']['r'] /= cnt_sse_head_unans
        results['sse_head']['unanswerable']['percent'] = cnt_sse_head_unans / cnt_pred

    cnt_sse_head_whole = results['sse_head']['whole_data']['count']
    if cnt_sse_head_whole:
        results['sse_head']['whole_data']['evaluation']['f1'] /= cnt_sse_head_whole
        results['sse_head']['whole_data']['evaluation']['em'] /= cnt_sse_head_whole
        results['sse_head']['whole_data']['evaluation']['p'] /= cnt_sse_head_whole
        results['sse_head']['whole_data']['evaluation']['r'] /= cnt_sse_head_whole
        results['sse_head']['whole_data']['percent'] = cnt_sse_head_whole / cnt_pred

    cnt_qasl_head_ans = results['qasl_head']['answerable']['count']
    if cnt_qasl_head_ans:
        results['qasl_head']['answerable']['evaluation']['f1'] /= cnt_qasl_head_ans
        results['qasl_head']['answerable']['evaluation']['em'] /= cnt_qasl_head_ans
        results['qasl_head']['answerable']['evaluation']['p'] /= cnt_qasl_head_ans
        results['qasl_head']['answerable']['evaluation']['r'] /= cnt_qasl_head_ans
        results['qasl_head']['answerable']['percent'] = cnt_qasl_head_ans / cnt_pred

    cnt_qasl_head_unans = results['qasl_head']['unanswerable']['count']
    if cnt_qasl_head_unans:
        results['qasl_head']['unanswerable']['evaluation']['f1'] /= cnt_qasl_head_unans
        results['qasl_head']['unanswerable']['evaluation']['em'] /= cnt_qasl_head_unans
        results['qasl_head']['unanswerable']['evaluation']['p'] /= cnt_qasl_head_unans
        results['qasl_head']['unanswerable']['evaluation']['r'] /= cnt_qasl_head_unans
        results['qasl_head']['unanswerable']['percent'] = cnt_qasl_head_unans / cnt_pred

    cnt_qasl_head_whole = results['qasl_head']['whole_data']['count']
    if cnt_qasl_head_whole:
        results['qasl_head']['whole_data']['evaluation']['f1'] /= cnt_qasl_head_whole
        results['qasl_head']['whole_data']['evaluation']['em'] /= cnt_qasl_head_whole
        results['qasl_head']['whole_data']['evaluation']['p'] /= cnt_qasl_head_whole
        results['qasl_head']['whole_data']['evaluation']['r'] /= cnt_qasl_head_whole
        results['qasl_head']['whole_data']['percent'] = cnt_qasl_head_whole / cnt_pred
    
    print(json.dumps(results, indent = 4))

    with open(file_path + '_global.jsonl', 'w', encoding='utf8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
