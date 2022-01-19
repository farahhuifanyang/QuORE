import itertools
import json
import re
from overrides import overrides
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

from pytorch_transformers import BasicTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields import (Field, TextField, IndexField, LabelField, ListField,
                                  MetadataField, SequenceLabelField, SpanField, ArrayField)
from tqdm import tqdm

from src.nhelpers import *
from src.preprocessing.utils import SPAN_ANSWER_TYPE, SPAN_ANSWER_TYPES, ALL_ANSWER_TYPES, MULTIPLE_SPAN
from src.preprocessing.utils import get_answer_type, fill_token_indices, token_to_span, standardize_dataset, extract_answer_info_from_annotation, create_bio_labels, create_io_labels

from src.preprocessing.find_spans import find_valid_spans, find_valid_spans_chinese

@DatasetReader.register("GeneralDatasetReader")
class GeneralDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 max_pieces: int = 512,
                 answer_types: List[str] = None,
                 bio_types: List[str] = None,
                 use_validated: bool = True,
                 max_depth: int = 3,
                 max_instances=-1,
                 uncased: bool = False,
                 is_training: bool = True,
                 is_minor_lang: bool = False,
                 bio_or_io: str = 'bio',
                 standardize_texts: bool = True,
                 flexibility_threshold: int = 1000,
                 multispan_allow_all_heads_to_answer: bool = False):
        super(GeneralDatasetReader, self).__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_pieces = max_pieces
        self.max_instances = max_instances
        self.answer_types = answer_types or ALL_ANSWER_TYPES
        self.bio_types = bio_types or [MULTIPLE_SPAN]
        self.use_validated = use_validated
        self.max_depth = max_depth

        self._uncased = uncased
        self._is_training = is_training
        self._is_minor_lang = is_minor_lang
        self.bio_or_io = bio_or_io
        self.standardize_texts = standardize_texts
        self.flexibility_threshold = flexibility_threshold
        self.multispan_allow_all_heads_to_answer = multispan_allow_all_heads_to_answer

        self.basic_tokenizer = BasicTokenizer(do_lower_case=uncased)
    
    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path, encoding = "utf8") as dataset_file:
            dataset = json.load(dataset_file)

        if self.standardize_texts and self._is_training:
            dataset = standardize_dataset(dataset)

        is_chinese_dataset = False
        instances_count = 0
        for passage_id, passage_info in tqdm(dataset.items()):
            passage_text = passage_info["passage"].strip()
            if re.search(u'[\u4e00-\u9fff]', passage_text):
                is_chinese_dataset = True

            word_tokens = self.tokenizer.tokenize(passage_text)

            passage_tokens = word_tokens
            # curr_index = 0

            print(passage_text)
            print(word_tokens)
            print()

            ### Not use wordpiece tokenization
            # for token in word_tokens:
            #     # Wordpiece tokenization is done here.
            #     wordpieces = self.tokenizer.tokenize(token.text)
            #     num_wordpieces = len(wordpieces)
            #     passage_tokens += wordpieces
            #     curr_index += num_wordpieces
            ###

            passage_tokens = fill_token_indices(passage_tokens, passage_text, self._uncased, self.basic_tokenizer, word_tokens)

            # Process questions from this passage
            for qa_pair in passage_info["qa_pairs"]:
                if 0 < self.max_instances <= instances_count:
                    return

                question_id = qa_pair["query_id"]
                question_text = qa_pair["question"].strip()
                
                answer_annotations: List[Dict] = list()
                specific_answer_type = None
                if 'answer' in qa_pair and qa_pair['answer']:
                    answer = qa_pair['answer']

                    specific_answer_type = get_answer_type(answer)
                    print(specific_answer_type)
                    if specific_answer_type not in self.answer_types:
                        continue

                    answer_annotations.append(answer)

                if self.use_validated and "validated_answers" in qa_pair and qa_pair["validated_answers"]:
                    answer_annotations += qa_pair["validated_answers"]
                
                print(question_text, '\n', passage_text, '\n', passage_tokens, '\n', question_id, '\n', passage_id, '\n', answer_annotations, '\n', specific_answer_type)

                # Filter noisy instances of SAOKE
                if question_id in [12796, 11003, 1825, 22332, 27103]:
                    continue

                instance = self.text_to_instance(question_text,
                                                 passage_text,
                                                 passage_tokens,
                                                 question_id,
                                                 passage_id,
                                                 answer_annotations,
                                                 specific_answer_type,
                                                 self.bio_or_io,
                                                 is_chinese_dataset,
                                                 self._is_minor_lang)
                if instance is not None:
                    instances_count += 1
                    yield instance

    @overrides
    def text_to_instance(self,
                         question_text: str,
                         passage_text: str,
                         passage_tokens: List[Token],
                         question_id: str = None,
                         passage_id: str = None,
                         answer_annotations: List[Dict] = None,
                         specific_answer_type: str = None,
                         bio_or_io: str = 'bio',
                         is_chinese_dataset: bool = False,
                         is_minor_lang: bool = False) -> Optional[Instance]:
        # Tokenize question and passage
        question_tokens = self.tokenizer.tokenize(question_text)
        question_tokens = fill_token_indices(question_tokens, question_text, self._uncased, self.basic_tokenizer)

        qlen = len(question_tokens)

        qp_tokens = [Token('[CLS]')] + question_tokens + [Token('[SEP]')] + passage_tokens

        # if qp has more than max_pieces tokens (including CLS and SEP), clip the passage
        max_passage_length = -1
        if len(qp_tokens) > self.max_pieces - 1:
            qp_tokens = qp_tokens[:self.max_pieces - 1]
            passage_tokens = passage_tokens[:self.max_pieces - qlen - 3]
            plen = len(passage_tokens)
            max_passage_length = token_to_span(passage_tokens[-1])[1] if plen > 0 else 0
        
        qp_tokens += [Token('[SEP]')]
        
        mask_indices = [0, qlen + 1, len(qp_tokens) - 1]
        
        fields: Dict[str, Field] = {}
            
        # Add feature fields
        qp_field = TextField(qp_tokens, self.token_indexers)
        fields["question_passage"] = qp_field
       
        mask_index_fields: List[Field] = [IndexField(index, qp_field) for index in mask_indices]
        fields["mask_indices"] = ListField(mask_index_fields)

        # Compile question, passage, answer metadata
        metadata = {"original_passage": passage_text,
                    "original_question": question_text,
                    "passage_tokens": passage_tokens,
                    "question_tokens": question_tokens,
                    "question_passage_tokens": qp_tokens,
                    "passage_id": passage_id,
                    "question_id": question_id,
                    "max_passage_length": max_passage_length,
                    "bio_or_io": bio_or_io}

        # in a word broken up into pieces, every piece except the first should be ignored when calculating the loss
        wordpiece_mask = [not token.text.startswith('##') for token in qp_tokens]
        wordpiece_mask = np.array(wordpiece_mask)
        fields['bio_wordpiece_mask'] = ArrayField(wordpiece_mask, dtype=np.int64)

        if answer_annotations:            
            # Get answer type, answer text, tokenize
            # For multi-span, remove repeating answers. Although possible, in the dataset it is mostly mistakes.
            answer_type, answer_texts = extract_answer_info_from_annotation(answer_annotations[0])
            print(answer_texts)
            if answer_type == SPAN_ANSWER_TYPE and not is_minor_lang:
                answer_texts = list(OrderedDict.fromkeys(answer_texts))
            tokenized_answer_texts = []
            if not is_minor_lang:
                for answer_text in answer_texts:
                    answer_tokens = self.tokenizer.tokenize(answer_text)
                    tokenized_answer_text = ' '.join(token.text for token in answer_tokens)
                    if tokenized_answer_text not in tokenized_answer_texts:
                        tokenized_answer_texts.append(tokenized_answer_text)
            else:
                tokenized_answer_texts = [token.text for token in self.tokenizer.tokenize(answer_texts)]

            metadata["answer_annotations"] = answer_annotations
            metadata["answer_texts"] = answer_texts
            metadata["answer_tokens"] = tokenized_answer_texts
            
            # Find answer text in passage
            valid_passage_spans = None
            if is_chinese_dataset or is_minor_lang:
                valid_passage_spans = find_valid_spans_chinese(passage_tokens, tokenized_answer_texts)
            else:
                valid_passage_spans = find_valid_spans(passage_tokens, tokenized_answer_texts)
            for span_ind, span in enumerate(valid_passage_spans):
                valid_passage_spans[span_ind] = (span[0] + qlen + 2, span[1] + qlen + 2)

            # throw away an instance in training if a span appearing in the answer is missing from the passage
            if self._is_training:
                if specific_answer_type in SPAN_ANSWER_TYPES:
                    for tokenized_answer_text in tokenized_answer_texts:
                        temp_spans = None
                        if is_chinese_dataset or is_minor_lang:
                            temp_spans = find_valid_spans_chinese(qp_field, [tokenized_answer_text])
                        else:
                            temp_spans = find_valid_spans(qp_field, [tokenized_answer_text])
                        # if len(temp_spans) == 0:
                        #     return None
            
            # Update metadata with answer info
            answer_info = {"answer_passage_spans": valid_passage_spans}
            metadata["answer_info"] = answer_info
        
            # Add answer fields
            passage_span_fields: List[Field] = []
            print(specific_answer_type)
            print(self.multispan_allow_all_heads_to_answer)
            if specific_answer_type != MULTIPLE_SPAN or self.multispan_allow_all_heads_to_answer:
                passage_span_fields: List[Field] = [SpanField(span[0], span[1], qp_field) for span in valid_passage_spans]
            print(passage_span_fields)
            if not passage_span_fields:
                passage_span_fields.append(SpanField(-1, -1, qp_field))
            fields["answer_as_passage_spans"] = ListField(passage_span_fields)
            print(fields["answer_as_passage_spans"])

            no_answer_bios = SequenceLabelField([0] * len(qp_tokens), sequence_field=qp_field)
            if (specific_answer_type in self.bio_types) and (len(valid_passage_spans) > 0):
                
                # Used for flexible BIO loss
                # START
                
                spans_dict = {}
                text_to_disjoint_bios: List[ListField] = []
                flexibility_count = 1
                for tokenized_answer_text in tokenized_answer_texts:
                    spans = None
                    if is_chinese_dataset or is_minor_lang:
                        spans = find_valid_spans_chinese(qp_tokens, [tokenized_answer_text])
                    else:
                        spans = find_valid_spans(qp_tokens, [tokenized_answer_text])
                    # if len(spans) == 0:
                    #     # possible if the passage was clipped, but not for all of the answers
                    #     continue
                    spans_dict[tokenized_answer_text] = spans

                    disjoint_bios: List[SequenceLabelField] = []
                    for span_ind, span in enumerate(spans):
                        bios = create_bio_labels([span], len(qp_field))
                        if bio_or_io == 'io':
                            bios = create_io_labels([span], len(qp_field))
                        disjoint_bios.append(SequenceLabelField(bios, sequence_field=qp_field))
                    
                    text_to_disjoint_bios.append(ListField(disjoint_bios))
                    flexibility_count *= ((2**len(spans)) - 1)

                fields["answer_as_text_to_disjoint_bios"] = ListField(text_to_disjoint_bios)

                if (flexibility_count < self.flexibility_threshold):
                    # generate all non-empty span combinations per each text
                    spans_combinations_dict = {}
                    for key, spans in spans_dict.items():
                        spans_combinations_dict[key] = all_combinations = []
                        for i in range(1, len(spans) + 1):
                            all_combinations += list(itertools.combinations(spans, i))

                    # calculate product between all the combinations per each text
                    packed_gold_spans_list = itertools.product(*list(spans_combinations_dict.values()))
                    bios_list: List[SequenceLabelField] = []
                    for packed_gold_spans in packed_gold_spans_list:
                        gold_spans = [s for sublist in packed_gold_spans for s in sublist]
                        bios = create_bio_labels(gold_spans, len(qp_field))
                        if bio_or_io == 'io':
                            bios = create_io_labels(gold_spans, len(qp_field))
                        bios_list.append(SequenceLabelField(bios, sequence_field=qp_field))
                    
                    fields["answer_as_list_of_bios"] = ListField(bios_list)
                    fields["answer_as_text_to_disjoint_bios"] = ListField([ListField([no_answer_bios])])
                else:
                    fields["answer_as_list_of_bios"] = ListField([no_answer_bios])

                # END

                # Used for both "require-all" BIO loss and flexible loss
                bio_labels = create_bio_labels(valid_passage_spans, len(qp_field))
                if bio_or_io == 'io':
                    bio_labels = create_io_labels(valid_passage_spans, len(qp_field))
                fields['span_bio_labels'] = SequenceLabelField(bio_labels, sequence_field=qp_field)

                fields["is_bio_mask"] = LabelField(1, skip_indexing=True)
            else:
                fields["answer_as_text_to_disjoint_bios"] = ListField([ListField([no_answer_bios])])
                fields["answer_as_list_of_bios"] = ListField([no_answer_bios])

                # create all 'O' BIO labels for non-span questions
                fields['span_bio_labels'] = no_answer_bios
                fields["is_bio_mask"] = LabelField(0, skip_indexing=True)

        fields["metadata"] = MetadataField(metadata)
        
        return Instance(fields)
