from typing import Any, Dict, List, Optional
import logging
from collections import OrderedDict

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from pytorch_transformers import BertModel

from src.compute_em_and_f1 import EmAndF1Evaluator
from src.multispan_heads import multispan_heads_mapping, decode_token_spans, remove_substring_from_prediction

logger = logging.getLogger(__name__)


@Model.register("QuoreOpenRelExtractor")
class QuoreOpenRelExtractor(Model):
    """
    This class implements single-span and multi-span answering ability.
    The code is based on NABERT+ implementation.
    """
    def __init__(self, 
                 vocab: Vocabulary, 
                 bert_pretrained_model: str, 
                 dropout_prob: float = 0.1, 
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 answering_abilities: List[str] = None,
                 unique_on_multispan: bool = False,
                 multispan_head_name: str = "flexible_loss_bio",
                 multispan_generation_top_k: int = 0,
                 multispan_prediction_beam_size: int = 1,
                 multispan_use_prediction_beam_search: bool = False,
                 multispan_use_bio_wordpiece_mask: bool = False,
                 dont_add_substrings_to_ms: bool = True) -> None:
        super().__init__(vocab, regularizer)

        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "multiple_spans"]
        else:
            self.answering_abilities = answering_abilities

        self.BERT = BertModel.from_pretrained(bert_pretrained_model)
        bert_dim = self.BERT.pooler.dense.out_features
        
        self.dropout = dropout_prob
        self._dont_add_substrings_to_ms = dont_add_substrings_to_ms

        self.multispan_head_name = multispan_head_name
        self.multispan_use_prediction_beam_search = multispan_use_prediction_beam_search
        self.multispan_use_bio_wordpiece_mask = multispan_use_bio_wordpiece_mask

        self._passage_weights_predictor = torch.nn.Linear(bert_dim, 1)
        self._multispan_weights_predictor = torch.nn.Linear(bert_dim, 1)
            
        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = \
                self.ff(2 * bert_dim, bert_dim, len(self.answering_abilities))

        if "passage_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._passage_span_start_predictor = torch.nn.Linear(bert_dim, 1)
            self._passage_span_end_predictor = torch.nn.Linear(bert_dim, 1)

        if "multiple_spans" in self.answering_abilities:
            if "flexible_loss" in self.multispan_head_name:
                self.multispan_head = multispan_heads_mapping[multispan_head_name](bert_dim, 
                    generation_top_k=multispan_generation_top_k, prediction_beam_size=multispan_prediction_beam_size)
            else:
                self.multispan_head = multispan_heads_mapping[multispan_head_name](bert_dim)
            
            self._multispan_module = self.multispan_head.module
            self._multispan_log_likelihood = self.multispan_head.log_likelihood
            self._multispan_prediction = self.multispan_head.prediction
            self._unique_on_multispan = unique_on_multispan

        self._metrics = EmAndF1Evaluator()
        initializer(self)

    def summary_vector(self, encoding, mask, in_type="passage"):
        """
        In NABERT (and in NAQANET), a 'summary_vector' is created for some entities, such as the
        passage or the question. This vector is created as a weighted sum of the elements of the
        entity, e.g. the passage summary vector is a weighted sum of the passage tokens.

        The specific weighting for every entity type is a learned.

        Parameters
        ----------
        encoding : BERT's output layer
        mask : a Tensor with 1s only at the positions relevant to ``in_type``
        in_type : the entity we want to summarize, e.g the passage

        Returns
        -------
        The summary vector according to ``in_type``.
        """
        if in_type == "passage":
            # Shape: (batch_size, seqlen)
            alpha = self._passage_weights_predictor(encoding).squeeze()
        elif in_type == "multiple_spans":
            #TODO: currenttly not using it...
            alpha = self._multispan_weights_predictor(encoding).squeeze()
        else:
            # Shape: (batch_size, #num of numbers, seqlen)
            alpha = torch.zeros(encoding.shape[:-1], device=encoding.device)
        # Shape: (batch_size, seqlen) 
        alpha = masked_softmax(alpha, mask)
        # Shape: (batch_size, out)
        h = util.weighted_sum(encoding, alpha)
        return h
    
    def ff(self, input_dim, hidden_dim, output_dim):
        return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
                                   torch.nn.ReLU(),
                                   torch.nn.Dropout(self.dropout),
                                   torch.nn.Linear(hidden_dim, output_dim))
    
    def forward(self,  # type: ignore
                question_passage: Dict[str, torch.LongTensor],
                mask_indices: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_text_to_disjoint_bios: torch.LongTensor = None,
                answer_as_list_of_bios: torch.LongTensor = None,
                span_bio_labels: torch.LongTensor = None,
                bio_wordpiece_mask: torch.LongTensor = None,
                is_bio_mask: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        # Shape: (batch_size, seqlen)
        question_passage_tokens = question_passage["tokens"]
        # Shape: (batch_size, seqlen)
        pad_mask = question_passage["mask"] 
        # Shape: (batch_size, seqlen)
        seqlen_ids = question_passage["tokens-type-ids"]
        
        max_seqlen = question_passage_tokens.shape[-1]
        batch_size = question_passage_tokens.shape[0]
                
        # Shape: (batch_size, 3)
        mask = mask_indices.squeeze(-1)
        # Shape: (batch_size, seqlen)
        cls_sep_mask = \
            torch.ones(pad_mask.shape, device=pad_mask.device).long().scatter(1, mask, torch.zeros(mask.shape, device=mask.device).long())
        # Shape: (batch_size, seqlen)
        passage_mask = seqlen_ids * pad_mask * cls_sep_mask
        # Shape: (batch_size, seqlen)
        question_mask = (1 - seqlen_ids) * pad_mask * cls_sep_mask

        question_and_passage_mask = question_mask | passage_mask
        if bio_wordpiece_mask is None or not self.multispan_use_bio_wordpiece_mask:
            multispan_mask = question_and_passage_mask
        else:
            multispan_mask = question_and_passage_mask * bio_wordpiece_mask

        # Shape: (batch_size, seqlen, bert_dim)
        bert_out, _ = self.BERT(question_passage_tokens, seqlen_ids, pad_mask)

        # Shape: (batch_size, qlen, bert_dim)
        question_end = max(mask[:,1])
        question_out = bert_out[:,:question_end]
        # Shape: (batch_size, qlen)
        question_mask = question_mask[:,:question_end]
        # Shape: (batch_size, out)
        question_vector = self.summary_vector(question_out, question_mask, "question")
        
        passage_out = bert_out
        del bert_out
        
        # Shape: (batch_size, bert_dim)
        passage_vector = self.summary_vector(passage_out, passage_mask)

        top_two_answer_abilities = None
        
        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = \
                self._answer_ability_predictor(torch.cat([passage_vector, question_vector], -1))
            answer_ability_log_probs = torch.nn.functional.log_softmax(answer_ability_logits, -1)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)
            top_two_answer_abilities = torch.topk(answer_ability_log_probs, k=2, dim=1)

        if "passage_span_extraction" in self.answering_abilities:
            passage_span_start_log_probs, passage_span_end_log_probs, best_passage_span = \
                self._passage_span_module(passage_out, passage_mask)

        if "multiple_spans" in self.answering_abilities:
            if "flexible_loss" in self.multispan_head_name:
                multispan_log_probs, multispan_logits = self._multispan_module(passage_out, seq_mask=multispan_mask)
            else:
                multispan_log_probs, multispan_logits = self._multispan_module(passage_out)
            
        output_dict = {}
        del passage_out, question_out
        # If answer is given, compute the loss.
        if answer_as_passage_spans is not None or span_bio_labels is not None:
            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "passage_span_extraction":
                    log_marginal_likelihood_for_passage_span = \
                        self._passage_span_log_likelihood(answer_as_passage_spans,
                                                            passage_span_start_log_probs,
                                                            passage_span_end_log_probs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)

                elif answering_ability == "multiple_spans":
                    if "flexible_loss" in self.multispan_head_name:
                        log_marginal_likelihood_for_multispan = \
                            self._multispan_log_likelihood(answer_as_text_to_disjoint_bios,
                                                        answer_as_list_of_bios,
                                                        span_bio_labels,
                                                        multispan_log_probs,
                                                        multispan_logits,
                                                        multispan_mask,
                                                        bio_wordpiece_mask,
                                                        is_bio_mask)
                    else:
                        log_marginal_likelihood_for_multispan = \
                            self._multispan_log_likelihood(span_bio_labels,
                                                        multispan_log_probs,
                                                        multispan_mask,
                                                        is_bio_mask,
                                                        logits=multispan_logits)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_multispan)
                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")

            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]
        
            output_dict["loss"] = - marginal_log_likelihood.mean()

        with torch.no_grad():
            # Compute the metrics and add the tokenized input to the output.
            if metadata is not None:
                if not self.training:
                    output_dict["passage_id"] = []
                    output_dict["query_id"] = []
                    output_dict["answer"] = []
                    output_dict["predicted_ability"] = []
                    output_dict["maximizing_ground_truth"] = []
                    output_dict["em"] = []
                    output_dict["f1"] = []
                    output_dict["precision"] = []
                    output_dict["recall"] = []
                    output_dict["invalid_spans"] = []
                    output_dict["max_passage_length"] = []

                i = 0
                while i < batch_size:
                    if len(self.answering_abilities) > 1:
                        predicted_ability_str = self.answering_abilities[best_answer_ability[i]]
                    else:
                        predicted_ability_str = self.answering_abilities[0]
                    
                    answer_json: Dict[str, Any] = {}

                    invalid_spans = []

                    q_text = metadata[i]['original_question']
                    p_text = metadata[i]['original_passage']
                    qp_tokens = metadata[i]['question_passage_tokens']
                    if predicted_ability_str == "passage_span_extraction":
                        answer_json["answer_type"] = "passage_span"
                        answer_json["value"], answer_json["spans"] = \
                            self._span_prediction(qp_tokens, best_passage_span[i], p_text, q_text, 'p')

                    elif predicted_ability_str == "multiple_spans":
                        answer_json["answer_type"] = "multiple_spans"
                        if "flexible_loss" in self.multispan_head_name:
                            answer_json["value"], answer_json["spans"], invalid_spans = \
                                self._multispan_prediction(multispan_log_probs[i], multispan_logits[i], qp_tokens, p_text, q_text,
                                                        multispan_mask[i], bio_wordpiece_mask[i], self.multispan_use_prediction_beam_search and not self.training)
                        else:
                            answer_json["value"], answer_json["spans"], invalid_spans = \
                                self._multispan_prediction(multispan_log_probs[i], multispan_logits[i], qp_tokens, p_text, q_text,
                                                        multispan_mask[i])
                        if self._unique_on_multispan:
                            answer_json["value"] = list(OrderedDict.fromkeys(answer_json["value"]))

                            if self._dont_add_substrings_to_ms:
                                answer_json["value"] = remove_substring_from_prediction(answer_json["value"])

                        # if len(answer_json["value"]) == 0 and len(self.answering_abilities) > 1:
                        #     best_answer_ability[i] = top_two_answer_abilities.indices[i][1]
                        #     continue

                    else:
                        raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")
                    
                    maximizing_ground_truth = None
                    em, f1, precision, recall = None, None, None, None
                    answer_annotations = metadata[i].get('answer_annotations', [])
                    if answer_annotations:
                        (em, f1, precision, recall), maximizing_ground_truth = self._metrics.call(answer_json["value"], answer_annotations, predicted_ability_str)

                    if not self.training:
                        output_dict["passage_id"].append(metadata[i]["passage_id"])
                        output_dict["query_id"].append(metadata[i]["question_id"])
                        output_dict["answer"].append(answer_json)
                        output_dict["predicted_ability"].append(predicted_ability_str)
                        output_dict["maximizing_ground_truth"].append(maximizing_ground_truth)
                        output_dict["em"].append(em)
                        output_dict["f1"].append(f1)
                        output_dict["precision"].append(precision)
                        output_dict["recall"].append(recall)
                        output_dict["invalid_spans"].append(invalid_spans)
                        output_dict["max_passage_length"].append(metadata[i]["max_passage_length"])

                    i += 1

        return output_dict

    def _passage_span_module(self, passage_out, passage_mask):
        # Shape: (batch_size, passage_length)
        passage_span_start_logits = self._passage_span_start_predictor(passage_out).squeeze(-1)

        # Shape: (batch_size, passage_length)
        passage_span_end_logits = self._passage_span_end_predictor(passage_out).squeeze(-1)

        # Shape: (batch_size, passage_length)
        passage_span_start_log_probs = util.masked_log_softmax(passage_span_start_logits, passage_mask)
        passage_span_end_log_probs = util.masked_log_softmax(passage_span_end_logits, passage_mask)

        # Info about the best passage span prediction
        passage_span_start_logits = util.replace_masked_values(passage_span_start_logits, passage_mask, -1e7)
        passage_span_end_logits = util.replace_masked_values(passage_span_end_logits, passage_mask, -1e7)

        # Shape: (batch_size, 2)
        best_passage_span = get_best_span(passage_span_start_logits, passage_span_end_logits)
        return passage_span_start_log_probs, passage_span_end_log_probs, best_passage_span

    def _passage_span_log_likelihood(self,
                                     answer_as_passage_spans,
                                     passage_span_start_log_probs,
                                     passage_span_end_log_probs):
        # Shape: (batch_size, # of answer spans)
        gold_passage_span_starts = answer_as_passage_spans[:, :, 0]
        gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_passage_span_mask = (gold_passage_span_starts != -1).long()
        clamped_gold_passage_span_starts = \
            util.replace_masked_values(gold_passage_span_starts, gold_passage_span_mask, 0)
        clamped_gold_passage_span_ends = \
            util.replace_masked_values(gold_passage_span_ends, gold_passage_span_mask, 0)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_passage_span_starts = \
            torch.gather(passage_span_start_log_probs, 1, clamped_gold_passage_span_starts)
        log_likelihood_for_passage_span_ends = \
            torch.gather(passage_span_end_log_probs, 1, clamped_gold_passage_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood_for_passage_spans = \
            log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood_for_passage_spans = \
            util.replace_masked_values(log_likelihood_for_passage_spans, gold_passage_span_mask, -1e7)
        # Shape: (batch_size, )
        log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)
        return log_marginal_likelihood_for_passage_span

    def _span_prediction(self, question_passage_tokens, best_span, passage_text, question_text, context):
        (predicted_start, predicted_end)  = tuple(best_span.detach().cpu().numpy())
        answer_tokens = question_passage_tokens[predicted_start:predicted_end + 1]
        spans_text, spans_indices = decode_token_spans([(context, answer_tokens)], passage_text, question_text)
        predicted_answer = spans_text[0]
        return predicted_answer, spans_indices

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        (exact_match, f1_score, p_score, r_score), scores_per_answer_type_and_head, \
        scores_per_answer_type, scores_per_head = self._metrics.get_metric(reset)
        metrics = {'em': exact_match, 'f1': f1_score, 'p': p_score, 'r': r_score}

        for answer_type, type_scores_per_head in scores_per_answer_type_and_head.items():
            for head, (answer_type_head_exact_match, answer_type_head_f1_score, answer_type_head_p_score, answer_type_head_r_score, type_head_count) in type_scores_per_head.items():
                if 'multi' in head and 'span' in answer_type:
                    metrics[f'em_{answer_type}_{head}'] = answer_type_head_exact_match
                    metrics[f'f1_{answer_type}_{head}'] = answer_type_head_f1_score
                    metrics[f'p_{answer_type}_{head}'] = answer_type_head_p_score
                    metrics[f'r_{answer_type}_{head}'] = answer_type_head_r_score
                else:
                    metrics[f'_em_{answer_type}_{head}'] = answer_type_head_exact_match
                    metrics[f'_f1_{answer_type}_{head}'] = answer_type_head_f1_score
                    metrics[f'p_{answer_type}_{head}'] = answer_type_head_p_score
                    metrics[f'r_{answer_type}_{head}'] = answer_type_head_r_score
                metrics[f'_counter_{answer_type}_{head}'] = type_head_count
        
        for answer_type, (type_exact_match, type_f1_score, type_p_score, type_r_score, type_count) in scores_per_answer_type.items():
            if 'span' in answer_type:
                metrics[f'em_{answer_type}'] = type_exact_match
                metrics[f'f1_{answer_type}'] = type_f1_score
                metrics[f'p_{answer_type}'] = type_p_score
                metrics[f'r_{answer_type}'] = type_r_score
            else:
                metrics[f'_em_{answer_type}'] = type_exact_match
                metrics[f'_f1_{answer_type}'] = type_f1_score
                metrics[f'p_{answer_type}'] = type_p_score
                metrics[f'r_{answer_type}'] = type_r_score
            metrics[f'_counter_{answer_type}'] = type_count

        for head, (head_exact_match, head_f1_score, head_p_score, head_r_score, head_count) in scores_per_head.items():
            if 'multi' in head:
                metrics[f'em_{head}'] = head_exact_match
                metrics[f'f1_{head}'] = head_f1_score
                metrics[f'p_{head}'] = head_p_score
                metrics[f'r_{head}'] = head_r_score
            else:
                metrics[f'_em_{head}'] = head_exact_match
                metrics[f'_f1_{head}'] = head_f1_score
                metrics[f'p_{head}'] = head_p_score
                metrics[f'r_{head}'] = head_r_score
            metrics[f'_counter_{head}'] = head_count
        
        return metrics
