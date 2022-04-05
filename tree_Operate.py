from treelib import Tree

DROOT = 0

class Tree_Node(object):
    def __init__(self,sen_id=-1,word_id_sen=-1,pos='',head=-1,dep='',isPred='',A0=[],A1=[],level=-1):
        self.sen_id = sen_id
        self.word_id_sen = word_id_sen
        self.pos = pos
        self.head = head
        self.dep = dep
        self.isPred = isPred
        self.A0 = A0
        self.A1 = A1
        self.children = {'left': [], 'right': []}
        self.grandsons = {'left': [], 'right': []}
        self.brother = {'left': [], 'right': []}
        self.parent = -1
        self.ellipsis = {'left': [], 'right': []}
        self.align = {'left': [], 'right': []}
        self.level = level

class Doc_Tree(object):
    def __init__(self,doc):
        self.dp_tree = Tree()
        # create root of dp tree
        self.dp_tree.create_node('DROOT', DROOT, data=Tree_Node(level=-1))
        for sentence in doc:
            for wordobj in sentence:
                # is the root of sentence
                if wordobj.isPred == 'Y' and wordobj.dep == 'ROOT':
                    self.dp_tree.create_node(wordobj.word, wordobj.word_id_doc, parent=DROOT,
                                                 data=Tree_Node(wordobj.sen_id, wordobj.word_id_sen, wordobj.pos,
                                                                wordobj.head, wordobj.dep, wordobj.isPred, wordobj.A0,
                                                                wordobj.A1,0))
                    self.build_dp_tree(sentence, wordobj.word_id_doc)
                    break

    def node_sort(self,node):
        return node.identifier

    #add all children node information of the current node
    def complete_ee(self,par_node):
        child_nodes = self.dp_tree.children(par_node.identifier)
        left_info = ''
        right_info = ''
        if not child_nodes:
            return par_node.tag

        #The nodes are sorted by number
        child_nodes.sort(key= self.node_sort)

        for child_node in child_nodes:
            if child_node.identifier < par_node.identifier:
                left_info += self.complete_ee(child_node)
            elif child_node.identifier > par_node.identifier:
                right_info += self.complete_ee(child_node)

        return left_info + par_node.tag + right_info

    # obtain children of specific node
    def get_all_node(self,cur_node,cnode_list):
        if not cur_node:
            return cnode_list
        child_nodes = self.dp_tree.children(cur_node.identifier)
        for child_node in child_nodes:
            self.get_all_node(child_node,cnode_list)
            cnode_list.append(child_node)

    #adjust tree structure
    def adjust_tree(self):
        nodes = self.dp_tree.all_nodes()
        for node in nodes:
            if node.identifier == DROOT:
                continue
            pnode = self.dp_tree.parent(node.identifier)
            ppnode = self.dp_tree.parent(pnode.identifier)
            # adjust parallel structure caused by punctuation
            if pnode.data.pos == 'CC' or (pnode.data.pos == 'PU' and pnode.data.dep == 'CJTN'):
                if node.identifier < ppnode.identifier:
                    self.dp_tree.nodes[ppnode.identifier].data.brother['left'].append(node.identifier)
                    self.dp_tree.nodes[node.identifier].data.brother['right'].append(ppnode.identifier)
                else:
                    self.dp_tree.nodes[ppnode.identifier].data.brother['right'].append(node.identifier)
                    self.dp_tree.nodes[node.identifier].data.brother['left'].append(ppnode.identifier)

                diff_level = abs(ppnode.data.level - node.data.level)
                cnodes = []
                self.get_all_node(node,cnodes)
                for cnode in cnodes:
                    self.dp_tree.nodes[cnode.identifier].data.level -= diff_level
                    if self.dp_tree.nodes[cnode.identifier].data.level <=0:
                        print('level update error')
                self.dp_tree.nodes[node.identifier].data.level = ppnode.data.level

                # adjust structure
                pppnode = self.dp_tree.parent(ppnode.identifier)
                if pppnode is not None:
                    self.dp_tree.move_node(node.identifier, pppnode.identifier)

    #convert sen_id of to doc_id
    def senid2docid_argu_tree(self):
        nodes = self.dp_tree.all_nodes()
        for node in nodes:
            if node.data.A0:
                self.update_nodes_argus(node, node.data.sen_id, node.data.A0,'A0')
            if node.data.A1:
                self.update_nodes_argus(node, node.data.sen_id, node.data.A1,'A1')

   
    def update_nodes_argus(self,cur_node,sen_id,argu_ids,argu):
        snodes = []
        nodes = self.dp_tree.all_nodes()
        #get the node whose id is sen_id
        for node in nodes:
            if node.data.sen_id == sen_id:
                snodes.append(node)

        #update predicate doc_id
        for i,argu_pred_id in enumerate(argu_ids):
            for node in snodes:
                if node.data.word_id_sen == argu_pred_id and argu == 'A0':
                    self.dp_tree.nodes[cur_node.identifier].data.A0[i] = node.identifier
                elif node.data.word_id_sen == argu_pred_id and argu == 'A1':
                    self.dp_tree.nodes[cur_node.identifier].data.A1[i] = node.identifier

   
    def build_dp_tree(self,sentence,pnode_id):
        pnode = self.dp_tree.get_node(pnode_id)
        for wordobj in sentence:
            if wordobj.head == pnode.data.word_id_sen:
                self.dp_tree.create_node(wordobj.word,wordobj.word_id_doc,parent=pnode.identifier,data=Tree_Node(wordobj.sen_id,wordobj.word_id_sen,wordobj.pos,wordobj.head,wordobj.dep,wordobj.isPred,wordobj.A0,wordobj.A1,pnode.data.level+1))
                self.build_dp_tree(sentence,wordobj.word_id_doc)

    def update_neighbors_tree(self):
        nodes = self.dp_tree.all_nodes()
        for node in nodes:
            if node.identifier == DROOT:
                continue
            pnode = self.dp_tree.parent(node.identifier)
            #update neighbor of parent
            self.dp_tree.nodes[node.identifier].data.parent = pnode.identifier
            #updata neighbor of child
            self.update_children_neighbors(node)
    

    def update_children_neighbors(self,cur_node):
        gcnodes = []
        self.get_all_node(cur_node, gcnodes)
        for node in gcnodes:
            pnode = self.dp_tree.parent(node.identifier)
            #left child
            if pnode.identifier == cur_node.identifier and node.identifier<cur_node.identifier:
                self.dp_tree.nodes[cur_node.identifier].data.children['left'].append(node.identifier)
            #right child
            elif pnode.identifier == cur_node.identifier and node.identifier > cur_node.identifier:
                self.dp_tree.nodes[cur_node.identifier].data.children['right'].append(node.identifier)
            #left grandson
            if pnode.identifier != cur_node.identifier and node.identifier < cur_node.identifier:
                self.dp_tree.nodes[cur_node.identifier].data.grandsons['left'].append(node.identifier)
            #right grandson
            elif pnode.identifier != cur_node.identifier and node.identifier > cur_node.identifier:
                self.dp_tree.nodes[cur_node.identifier].data.grandsons['right'].append(node.identifier)

