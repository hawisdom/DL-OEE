from torch.nn.utils.rnn import pad_sequence
import logging
import os
import pickle
from collections import Counter, defaultdict
from copy import copy, deepcopy
from torch import nn
import numpy as np
import torch
from torch.utils.data import Dataset
from tree_Operate import *
import scipy.sparse as sp
from gensim.models import word2vec
import gensim
import math
import json

logger = logging.getLogger(__name__)

#word object
class wordObj(object):
    def __init__(self,sen_id=-1,word_id_doc=-1,word_id_sen=-1,word='',pos='',head=-1,dep='',isPred='',argus=[],A0=[],A1=[]):
        self.sen_id = sen_id
        self.word_id_doc = word_id_doc
        self.word_id_sen = word_id_sen
        self.word = word
        self.pos = pos
        self.head = head
        self.dep = dep
        self.isPred = isPred
        self.argus = argus
        self.A0 = A0
        self.A1 = A1

def save_tree(doc_trees):
    for i,doc_tree in enumerate(doc_trees):
        file = './data/tree/'+str(i)+'.txt'
        doc_tree.dp_tree.save2file(file)

def load_datasets_and_vocabs(args):
    train_data_file = os.path.join(args.output_dir, 'data_cache/train_data_catch.txt')
    test_data_file = os.path.join(args.output_dir, 'data_cache/test_data_catch.txt')
    train_weight_file = os.path.join(args.output_dir, 'data_cache/train_weight_catch.txt')
    test_weight_file = os.path.join(args.output_dir, 'data_cache/test_weight_catch.txt')

    if os.path.exists(train_data_file) and os.path.exists(test_data_file):
        logger.info('Loading train_data from %s', train_data_file)
        with open(train_data_file, 'rb') as f:
            train_all_unrolled = json.load(f)
        for i in range(len(train_all_unrolled)):
            train_all_unrolled[i]['adj'] = torch.Tensor(train_all_unrolled[i]['adj'])
            train_all_unrolled[i]['adj_node_type'] = torch.Tensor(train_all_unrolled[i]['adj_node_type'])

        logger.info('Loading train_weight from %s', train_weight_file)
        with open(train_weight_file, 'rb') as f:
            train_labels_weight = torch.Tensor(json.load(f))

        logger.info('Loading train_data from %s', test_data_file)
        with open(test_data_file, 'rb') as f:
            test_all_unrolled = json.load(f)
        for i in range(len(test_all_unrolled)):
            test_all_unrolled[i]['adj'] = torch.Tensor(test_all_unrolled[i]['adj'])
            test_all_unrolled[i]['adj_node_type'] = torch.Tensor(test_all_unrolled[i]['adj_node_type'])

        logger.info('Loading test_weight from %s', train_weight_file)
        with open(test_weight_file, 'rb') as f:
            test_labels_weight = torch.Tensor(json.load(f))
    else:
        train, test = get_dataset()

        train_trees = create_doc_tree(train)
        # save_tree(train_trees)
        test_trees = create_doc_tree(test)
        # save_tree(train_trees)

        senid2docid_argus(train_trees)
        senid2docid_argus(test_trees)
        adjust_tree(train_trees)
        adjust_tree(test_trees)

        update_neighbors(train_trees)
        update_neighbors(test_trees)

        train_all_unrolled,train_labels_weight = get_rolled_and_unrolled_data(train_trees,args)
        test_all_unrolled,test_labels_weight = get_rolled_and_unrolled_data(test_trees,args)
        logger.info('Creating train_data_cache')
        with open(train_data_file,'w') as f:
            json.dump(train_all_unrolled,f)
        logger.info('Creating train_weight_cache')
        with open(train_weight_file,'w') as wf:
            json.dump(train_labels_weight.detach().cpu().numpy().tolist(),wf)

        logger.info('Creating test_data_cache')
        with open(test_data_file,'w') as f:
            json.dump(test_all_unrolled,f)
        logger.info('Creating test_weight_cache')
        with open(test_weight_file,'w') as wf:
            json.dump(test_labels_weight.detach().cpu().numpy().tolist(),wf)

    logger.info('****** After unrolling ******')
    logger.info('Train set size: %s', len(train_all_unrolled))
    logger.info('Test set size: %s,', len(test_all_unrolled))

    # Build word vocabulary(part of speech, dep_tag) and save pickles.
    word_vecs,word_vocab,dep_tag_vocab, pos_tag_vocab = load_and_cache_vocabs(train_all_unrolled+test_all_unrolled, args)

    if args.embedding_type == 'word2vec':
        embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32))
        args.word2vec_embedding = embedding

    train_dataset = EE_Depparsed_Dataset(
        train_all_unrolled, args,word_vocab,dep_tag_vocab,pos_tag_vocab)
    test_dataset = EE_Depparsed_Dataset(
        test_all_unrolled, args,word_vocab,dep_tag_vocab,pos_tag_vocab)

    return train_dataset,train_labels_weight, test_dataset,test_labels_weight, word_vocab, dep_tag_vocab, pos_tag_vocab


def get_dataset():
    train_name = './data/CoNLL2009-ST-Chinese-train-event.txt'
    test_name = './data/CoNLL2009-ST-evaluation-Chinese-event.txt'

    train = read_sentence(train_name)
    test = read_sentence(test_name)

    return train, test

def read_sentence(file_path):
    docs = []
    sentences = []
    sentence = []
    words = ''
    word_id_doc = 0
    sen_id = 1
    pred_id_list = []
    with open(file_path, 'r',encoding='utf-8-sig') as f:
        datas = f.readlines()
        for line in datas:
            elements = line.split('\t')
            if (line.strip() == ''):
                sen_id += 1
                if words == "（完）" or words == "完":
                    word_id_doc = 0
                    sen_id = 0
                    docs.append(sentences)
                    sentences = []
                    sentence = []
                    words = ''
                    continue
                sentence = update_argus(sentence, pred_id_list)
                sentences.append(sentence)
                sentence = []
                pred_id_list = []
                words = ''
                continue
            words += elements[1]
            word_id_doc += 1
            if elements[12] == 'Y':
                pred_id_list.append(elements[0])
            wordobj = wordObj(sen_id,word_id_doc,int(elements[0]),elements[1],elements[4],int(elements[8]),elements[10],elements[12],elements[14:],[],[])
            sentence.append(wordobj)

    return docs

def update_argus(sentence,pred_id_list):
    for i,wordobj in enumerate(sentence):
        argus = wordobj.argus
        for m, arg in enumerate(argus):
            if arg.strip() == 'A0':
                sentence[i].A0.append(int(pred_id_list[m]))
            if arg.strip() == 'A1':
                sentence[i].A1.append(int(pred_id_list[m]))
    return sentence

def create_doc_tree(docs):
    doc_trees = []
    for doc in docs:
        doc_tree = Doc_Tree(doc)
        doc_trees.append(doc_tree)
    return doc_trees

#convert sen_id of argus to doc_id
def senid2docid_argus(doc_trees):
    for doc_tree in doc_trees:
        doc_tree.senid2docid_argu_tree()

def adjust_tree(doc_trees):
    for doc_tree in doc_trees:
        doc_tree.adjust_tree()

def remove_stop_word_nodes(doc_trees):
    for doc_tree in doc_trees:
        doc_tree.remove_stop_word_nodes_tree()

def update_neighbors(doc_trees):
    for doc_tree in doc_trees:
        doc_tree.update_neighbors_tree()

def get_rolled_and_unrolled_data(doc_trees,args):
    unroll_datas = []
    labels = []
    roles_lookup = {'none':0,'sub': 1, 'pred': 2, 'obj': 3}
    for doc_tree in doc_trees:
        core_nodes = doc_tree.dp_tree.children(DROOT)
        core_nodes.sort(key=doc_tree.node_sort)
        for core_node in core_nodes:
            unroll_nodes = []
            edges_unordered = []
            edges = []
            node_types = []
            brother_node_ids = core_node.data.brother['left']
            for brother_node_id in brother_node_ids:
                brother_node = doc_tree.dp_tree.get_node(brother_node_id)
                bcnodes = get_brothers_all_nodes(doc_tree, brother_node)
                new_bcnodes = deepcopy(bcnodes)
                update_brother_label_unroll(new_bcnodes, core_node)
                unroll_nodes += new_bcnodes
            cnodes = []
            doc_tree.get_all_node(core_node,cnodes)
            unroll_nodes += cnodes

            unroll_nodes.append(core_node)

            #remove repetitive nodes in unroll_nodes, and sort them by node id
            unroll_nodes = list(set(unroll_nodes))
            unroll_nodes.sort(key=doc_tree.node_sort)

            data = {'sentence':'','tokens':[],'ids':[],'pos':[],'dep':[],'level':[],'lchild':[],'rchild':[],'lgrandson':[],'rgrandson':[],'lbrother':[],'rbrother':[],'parent':[],'role':[]}
            for node in unroll_nodes:
                data['sentence'] = data['sentence'] + node.tag + ' '
                data['ids'].append(node.identifier)
                data['tokens'].append(node.tag)
                data['pos'].append(node.data.pos)
                data['dep'].append(node.data.dep)
                data['level'].append(math.exp(-node.data.level))
                #self join
                edges_unordered.append([node.identifier,node.identifier])
                node_types.append(args.self_weight)
                #edges of other nodes
                edges_unordered += get_edges_unordered(node.identifier, node.data.children['left'],args.children_weight, node_types)
                edges_unordered += get_edges_unordered(node.identifier, node.data.children['right'],args.children_weight, node_types)
                edges_unordered += get_edges_unordered(node.identifier, node.data.grandsons['left'],args.grandsons_weight, node_types)
                edges_unordered += get_edges_unordered(node.identifier, node.data.grandsons['right'],args.grandsons_weight, node_types)
                edges_unordered += get_edges_unordered(node.identifier, node.data.brother['left'], args.brother_weight,node_types)
                if node.data.parent != 0:
                    edges_unordered.append([node.identifier,node.data.parent])
                    node_types.append(args.parent_weight)

                if node.data.A0:
                    data['role'].append(roles_lookup['sub'])
                elif node.data.A1:
                    data['role'].append(roles_lookup['obj'])
                elif node.data.isPred == 'Y':
                    data['role'].append(roles_lookup['pred'])
                else:
                    data['role'].append(roles_lookup['none'])

            data['sentence'].strip('\t')
            labels += data['role']

            # build adj
            ids = np.array(data['ids'], dtype=np.int32)
            idx_map = {j: i for i, j in enumerate(ids)}
            for edge in edges_unordered:
                edges.append([idx_map[edge[0]], idx_map[edge[1]]])
            edges = np.array(edges, dtype=np.int32).reshape(np.array(edges_unordered).shape)
            node_types = np.array(node_types)
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(np.array(data['role']).shape[0], np.array(data['role']).shape[0]),
                                dtype=np.float32)
            adj_node_type= sp.coo_matrix((node_types, (edges[:, 0], edges[:, 1])),
                                shape=(np.array(data['role']).shape[0], np.array(data['role']).shape[0]),
                                dtype=np.float32)

            adj = torch.FloatTensor(np.array(adj.todense()))
            adj_node_type = torch.FloatTensor(np.array(adj_node_type.todense()))

            data['adj']=adj.numpy().tolist()
            data['adj_node_type'] = adj_node_type.numpy().tolist()
            unroll_datas.append(data)

    weight_tensor = get_labels_weight(labels, args)
    return unroll_datas,weight_tensor

def get_labels_weight(labels,args):
    label_ids = labels
    nums_labels = Counter(labels)
    nums_labels = [(l,k) for k, l in sorted([(j, i) for i, j in nums_labels.items()], reverse=True)]
    size = len(nums_labels)
    if size % 2 == 0:
        median = (nums_labels[size // 2][1] + nums_labels[size//2-1][1])/2
    else:
        median = nums_labels[(size - 1) // 2][1]

    weight_list = []
    roles_lookup = {'none': 0, 'sub': 1, 'pred': 2, 'obj': 3}
    for value_id in roles_lookup.values():
        if value_id not in label_ids:
            weight_list.append(0)
        else:
            for label in nums_labels:
                if label[0] == value_id:
                    weight_list.append(median/label[1])
                    break

    weight_tensor = torch.tensor(weight_list,dtype=torch.float32)

    return weight_tensor.to(args.device)

def get_brothers_all_nodes(doc_tree,brother_node):
    bcnodes = []
    doc_tree.get_all_node(brother_node, bcnodes)
    bcnodes.append(brother_node)
    bbrother_node_ids = brother_node.data.brother['left']
    for bbrother_node_id in bbrother_node_ids:
        bbrother_node = doc_tree.dp_tree.get_node(bbrother_node_id)
        bcnodes += get_brothers_all_nodes(doc_tree,bbrother_node)

    return bcnodes


def get_edges_unordered(node_id,come_nodes,type_weight,node_types):
    edges_unordered = []
    for come_node in come_nodes:
        edges_unordered.append([node_id,come_node])
        node_types.append(type_weight)
    return edges_unordered

def update_brother_label_unroll(brother_nodes,cur_core_node):
    for i,brother_node in enumerate(brother_nodes):
        if brother_node.data.isPred == 'Y':
            brother_nodes[i].data.isPred = '_'
        A0s = brother_node.data.A0
        for A0 in A0s:
            if A0 != cur_core_node.identifier:
                brother_nodes[i].data.A0.remove(A0)

        A1s = brother_node.data.A1
        for A1 in A1s:
            if A1 != cur_core_node.identifier:
                brother_nodes[i].data.A1.remove(A1)

def load_and_cache_vocabs(data, args):
    '''
    Build vocabulary of words, part of speech tags, dependency tags and cache them.
    Load glove embedding if needed.
    '''
    pkls_path = os.path.join(args.output_dir, 'pkls')
    if not os.path.exists(pkls_path):
        os.makedirs(pkls_path)

    # Build or load word vocab and word2vec embeddings.
    if args.embedding_type == 'word2vec':
        cached_word_vocab_file = os.path.join(
            pkls_path, 'cached_{}_{}_word_vocab.pkl'.format(args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vocab_file):
            logger.info('Loading word vocab from %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'rb') as f:
                word_vocab = pickle.load(f)
        else:
            logger.info('Creating word vocab from dataset %s',
                        args.dataset_name)
            word_vocab = build_text_vocab(data)
            logger.info('Word vocab size: %s', word_vocab['len'])
            logging.info('Saving word vocab to %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'wb') as f:
                pickle.dump(word_vocab, f, -1)

        cached_word_vecs_file = os.path.join(pkls_path, 'cached_{}_{}_word_vecs.pkl'.format(
            args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vecs_file):
            logger.info('Loading word vecs from %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'rb') as f:
                word_vecs = pickle.load(f)
        else:
            logger.info('Creating word vecs from %s', args.word2vec_dir)
            word_vecs = load_word2vec_embedding(
                word_vocab['itos'], args,0.25)
            logger.info('Saving word vecs to %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'wb') as f:
                pickle.dump(word_vecs, f, -1)
    else:
        word_vocab = None
        word_vecs = None

    # Build vocab of dependency tags
    cached_dep_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_dep_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        dep_tag_vocab = build_dep_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of pos tags from %s',
                    cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of pos tags.')
        pos_tag_vocab = build_pos_tag_vocab(data, min_freq=0)
        logger.info('Saving pos tags  vocab, size: %s, to file %s',
                    pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    return word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab


def load_word2vec_embedding(words,args,uniform_scale):
    if args.our_corpus_w2v:
        in_file = generate_train_datas(words, args)
        w2v_model = Train_w2v_model(in_file, args)
    else:
        path = os.path.join(args.word2vec_dir,'baike_26g_news_13g_novel_229g.model')
        w2v_model = gensim.models.Word2Vec.load(path)

    w2v_vocabs = [word for word, Vocab in w2v_model.wv.vocab.items()]  # 存储 所有的 词语

    word_vectors = []
    for word in words:
        if word in w2v_vocabs:
            word_vectors.append(w2v_model.wv[word])
        elif word == '<pad>':
            word_vectors.append(np.zeros(w2v_model.vector_size, dtype=np.float32))
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, w2v_model.vector_size))

    return word_vectors


def _default_unk_index():
    return 1

def generate_train_datas(datas,args):
    words = ''
    for data in datas:
        words += data
        words += ' '
    path = os.path.join(args.word2vec_dir,args.dataset_name+'_word2vec.txt')
    with open(path,'w',encoding='utf-8-sig') as f:
        f.write(words)
    return path

def Train_w2v_model(in_file,args):
    sentences = word2vec.Text8Corpus(in_file)
    model = word2vec.Word2Vec(sentences,min_count=1, size=args.embedding_dim, workers=6)
    model_path = os.path.join(args.word2vec_dir,args.dataset_name+'.model')
    model.save(model_path)
    return model

def build_text_vocab(data, vocab_size=100000, min_freq=2):
    counter = Counter()
    for d in data:
        s = d['tokens']
        counter.update(s)

    itos = ['[PAD]', '[UNK]']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_pos_tag_vocab(datas, vocab_size=1000, min_freq=1):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for data in datas:
        tags = data['pos']
        counter.update(tags)

    itos = ['<pad>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_dep_tag_vocab(datas, vocab_size=1000, min_freq=0):
    counter = Counter()
    for data in datas:
        tags = data['dep']
        counter.update(tags)

    itos = ['<pad>', '<unk>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word == '<pad>':
            continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


class EE_Depparsed_Dataset(Dataset):
    def __init__(self, data, args, word_vocab,dep_tag_vocab, pos_tag_vocab):
        self.data = data
        self.args = args
        self.word_vocab = word_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.pos_tag_vocab = pos_tag_vocab

        self.convert_features()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        items = e['tokens_ids'],e['pos_class'],e['dep_ids'], e['text_len'], e['level'],e['adj'],e['adj_node_type'],e['role']

        items_tensor = tuple(torch.tensor(t) for t in items)
        return items_tensor

    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.data)):
            self.data[i]['tokens_ids'] = [self.word_vocab['stoi'][w] for w in self.data[i]['tokens']]
            self.data[i]['text_len'] = len(self.data[i]['tokens'])
            self.data[i]['dep_ids'] = [self.dep_tag_vocab['stoi'][w] for w in self.data[i]['dep']]
            self.data[i]['pos_class'] = [self.pos_tag_vocab['stoi'][w] for w in self.data[i]['pos']]


def my_collate(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.
    '''
    tokens_ids,pos_class, dep_ids, text_len,level,adj,adj_node_type,role = zip(
        *batch)  # from Dataset.__getitem__()
    text_len = torch.tensor(text_len)

    # Pad sequences.
    tokens_ids = pad_sequence(tokens_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)
    dep_ids = pad_sequence(dep_ids, batch_first=True, padding_value=0)
    level = pad_sequence(level, batch_first=True, padding_value=0)
    role = pad_sequence(role, batch_first=True, padding_value=0)

    adj_list = []
    for i, t in enumerate(adj):
        pad = nn.ZeroPad2d(padding=(0, role.shape[1] - t.shape[1], 0, role.shape[1] - t.shape[1]))
        adj_list.append(pad(t))
    new_adj = torch.stack(adj_list,dim=0)

    adj_node_type_list = []
    for i, t in enumerate(adj_node_type):
        pad = nn.ZeroPad2d(padding=(0, role.shape[1] - t.shape[1], 0, role.shape[1] - t.shape[1]))
        adj_node_type_list.append(pad(t))
    new_adj_node_type = torch.stack(adj_node_type_list, dim=0)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    tokens_ids = tokens_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    dep_ids = dep_ids[sorted_idx]
    text_len = text_len[sorted_idx]
    level = level[sorted_idx]
    new_adj = new_adj[sorted_idx]
    new_adj_node_type = new_adj_node_type[sorted_idx]
    role = role[sorted_idx]

    return tokens_ids, pos_class, dep_ids,text_len, level,new_adj,new_adj_node_type,role
