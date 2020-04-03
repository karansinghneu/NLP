import spacy
from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.data.dataset_readers import conll2003

torch.manual_seed(1)


class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                metadata,
                tags: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if tags is not None:
            self.accuracy(tag_logits, tags, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, tags, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


def importData():
    reader = conll2003.Conll2003DatasetReader()
    train_dataset = reader.read(cached_path('http://www.ccs.neu.edu/home/dasmith/onto.train.ner.sample'))
    validation_dataset = reader.read(cached_path('http://www.ccs.neu.edu/home/dasmith/onto.development.ner.sample'))
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
    return train_dataset, validation_dataset, vocab, reader


def trainModel(train_dataset, validation_dataset, vocab):
    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, bidirectional=False, batch_first=True))
    model = LstmTagger(word_embeddings, lstm, vocab)
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, eps=1e-8)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    iterator = BucketIterator(batch_size=2, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=100,
                      cuda_device=cuda_device)
    trainer.train()
    return model


def evaluation(model, reader):
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    return predictor


def tag_sentence(predictor, model, s):
    tag_ids = np.argmax(predictor.predict_instance(s)['tag_logits'], axis=-1)
    fields = zip(s['tokens'], s['tags'], [model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
    return list(fields)


def getBaseline(predictor, model, validation_dataset):
    baseline_output = [tag_sentence(predictor, model, i) for i in validation_dataset]
    return baseline_output


# TODO: count the number of NER label violations,
# such as O followed by I-TYPE or B-TYPE followed by
# I-OTHER_TYPE
# Take tagger output as input
def violations(tagged):
    count = 0
    list_of_all = []
    for i in range(0, len(tagged)):
        for j in range(0, len(tagged[i])):
            list_of_all.append(tagged[i][j][2])
    m = 0
    k = 1
    while m < len(list_of_all) - 1:
        while k < len(list_of_all):
            if list_of_all[m] == 'O' and list_of_all[k] == 'O':
                m += 1
                k += 1
            elif list_of_all[m] == 'O':
                if (zeroI_Violation(list_of_all[k]) == 1):
                    count += 1
                    m += 1
                    k += 1
                else:
                    m += 1
                    k += 1
            elif list_of_all[k] == 'O':
                m += 1
                k += 1
            else:
                if (violation(list_of_all[m], list_of_all[k])) == 1:
                    count += 1
                    m += 1
                    k += 1
                else:
                    m += 1
                    k += 1
    return count


# TODO: return the span-level precision, recall, and F1
# Take tagger output as input
def span_stats(tagged):
    precision = 0
    recall = 0
    f1 = 0
    list_of_golden = []
    list_of_actual = []
    for i in range(0, len(tagged)):
        for j in range(0, len(tagged[i])):
            list_of_golden.append(tagged[i][j][1])
            list_of_actual.append(tagged[i][j][2])
    tup_list_gold = getSpan(list_of_golden)
    tup_list_model = getSpan(list_of_actual)
    line_up = 0
    for tup in tup_list_gold:
        if tup in tup_list_model:
            line_up += 1
    precision = line_up / len(tup_list_model)
    print('Precision is ', precision)
    recall = line_up / len(tup_list_gold)
    print('Recall is ', recall)
    f1 = 2 * (precision * recall) / (precision + recall)
    print('F1 is ', f1)

    return {'precision': precision,
            'recall': recall,
            'f1': f1}


## You can check how many violations are made by the model output in predictor.


def violation(str1, str2):
    arr1 = str1.split('-')
    arr2 = str2.split('-')
    if arr1[0] == 'B' and arr2[0] == 'I':
        if arr1[1] == arr2[1]:
            return 0
        else:
            return 1
    if arr1[0] == 'B' and arr2[0] == 'B':
        return 0
    if arr1[0] == 'I' and arr2[0] == 'B':
        return 0
    if arr1[0] == 'I' and arr2[0] == 'I':
        if arr1[1] == arr2[1]:
            return 0
        else:
            return 1
    else:
        return 0


def zeroI_Violation(str1):
    arr = str1.split('-')
    if arr[0] == 'I':
        return 1
    else:
        return 0


def decoding(vocab):
    # This code show how to map from output vector components to labels
    dict_labels = vocab.get_index_to_token_vocabulary('labels')
    transition = np.zeros((len(dict_labels), len(dict_labels)))
    for i in range(0, len(transition)):
        for j in range(0, len(transition[0])):
            temp_list = []
            temp_list.append(dict_labels[i])
            temp_list.append(dict_labels[j])
            if temp_list[0] == 'O' and temp_list[1] == 'O':
                transition[i][j] = np.log(1)
            elif temp_list[0] == 'O':
                if (zeroI_Violation(temp_list[1]) == 1):
                    transition[i][j] = np.log(0)
                else:
                    transition[i][j] = np.log(1)
            elif temp_list[1] == 'O':
                transition[i][j] = np.log(1)
            else:
                if (violation(temp_list[0], temp_list[1])) == 1:
                    transition[i][j] = np.log(0)
                else:
                    transition[i][j] = np.log(1)
    return transition


def viterbi(s, model, predictor, transition):
    emission = np.asarray(predictor.predict_instance(s)['tag_logits'])
    dp = np.zeros((len(emission), len(emission[0])))
    backpointers = np.zeros((len(emission) - 1, len(emission[0])))
    dp[0, :] = emission[0, :]
    result = [0] * len(emission)
    s_prob = 0.0
    for i in range(1, len(emission)):
        for j in range(0, len(emission[0])):
            dp[i][j] = emission[i][j] + np.max([transition[k][j] + dp[i - 1][k] for k in range(0, len(emission[0]))])
            backpointers[i - 1][j] = np.argmax([transition[k][j] + dp[i - 1][k] for k in range(0, len(emission[0]))])
    s_prob = np.max([dp[-1][k] for k in range(0, len(emission[0]))])
    b_next = np.argmax([dp[len(emission) - 1][k] for k in range(0, len(emission[0]))])
    result[len(emission) - 1] = b_next
    for l in range(len(emission) - 2, -1, -1):
        b_next = backpointers[l][int(b_next)]
        result[l] = b_next
    fields_new = zip(s['tokens'], s['tags'], [model.vocab.get_token_from_index(i, 'labels') for i in result])
    return list(fields_new)


def getSpan(golden):
    golden_tup_list = []
    m = 0
    while m != len(golden):
        if not golden[m] == 'O':
            split_golden1 = golden[m].split('-')
            k = m + 1
            if m == len(golden) - 1:
                golden_tup_list.append((split_golden1[1], m, k - 1))
                break
            while k != len(golden):
                if not golden[k] == 'O':
                    split_golden2 = golden[k].split('-')
                    if split_golden1[1] == split_golden2[1]:
                        k += 1
                        if k == len(golden):
                            golden_tup_list.append((split_golden1[1], m, k - 1))
                            m = k
                            break
                    else:
                        golden_tup_list.append((split_golden1[1], m, k - 1))
                        m = k
                        break
                else:
                    golden_tup_list.append((split_golden1[1], m, k - 1))
                    m = k
                    break
        else:
            m += 1
    return golden_tup_list


train, val, voc, rd = importData()
md = trainModel(train, val, voc)
pd = evaluation(md, rd)
bline = getBaseline(pd, md, val)
## You can check how many violations are made by the model output in predictor.
print('The number of violations made by the model output in predictor are: ', violations(bline))
print(span_stats(bline))
trans = decoding(voc)
new_out = [viterbi(i, md, pd, trans) for i in val]
print('The number of violations in new output of viterbi decoder is:', violations(new_out))
print(span_stats(new_out))
