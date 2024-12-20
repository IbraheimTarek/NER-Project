"""
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

// SPDX-License-Identifier: CC-BY-NC-4.0
"""
import re
from abc import ABC, abstractmethod
from anytree import AnyNode, RenderTree
from sexp_reader import build_parent_group_mapping, parse_sexp
import numpy as np

class SemanticTree(ABC):
    ROOT_SYMBOL = 'DUMMY-ROOT'

    def __init__(self, *, flat_string=None, tree_rep=None, root_symbol=None, children=None):
        '''
        This construtor is used to instantiate an object of derived classes
        using either a flat string `flat_string`, a tree representation
        `tree_rep`, or the combination of a root symbol and a list of tree_rep children.

        :param flat_string: (str) input flat string to construct a tree, if possible.
        :param tree_rep: (AnyNode) a tree.
        :param root_symbol: (str) a string that will be used as id of root node.
        :param children: (list) a list of SemanticTree objects to be defined as children of root_symbol.
        '''
        try:
            # we pass `flat_string` when we instantiate an object of "Tree" class
            # when no underlying Tree representation is currently existing, i.e we
            # create an object from ground up. 
            if flat_string:
                self.tree_rep = self._linearized_rep_to_tree_rep(flat_string)
            # we pass `tree_rep` when we instantiate an object of a derived class
            # when an underlying Tree representation already exists but we want to
            # have a new tree representation from an existing one.
            elif tree_rep:
                self.tree_rep = tree_rep
            # we pass root_symbol and children when we want to construct a new tree
            # object from a list of valid tree children and a symbol for the parent root node.
            elif root_symbol and children:
                self.tree_rep = AnyNode(id=root_symbol, children=[c.tree_rep for c in children])

        except Exception as e:
            raise ValueError() from e

    def pretty_string(self):
        '''
        Return a string for the rendered tree. 

        :return: (str) Returns a rendered tree string
        '''
        tree_string = ""
        for tree_string_prelude, _, node in RenderTree(self.tree_rep):
            tree_string += f"{tree_string_prelude}{node.id}\n"
        return tree_string


    def tree_num_nodes(self):
        '''
        Return a node-count for the rendered tree. 

        :return: (int) Returns a rendered tree node count
        '''
        count = 0
        for _, _, _ in RenderTree(self.tree_rep):
            count += 1
        return count

    # Since we don't want to expose the internal implementation details of 
    # the how we construct a tree, i.e using AnyNode class we pass a dict 
    # with different attributes. In this case, its just `id`.  
    def root_symbol(self):
        '''
        Returns the name of the root for this tree.

        :return: (str) Name of the root of the tree 
        '''
        return self.tree_rep.id

    def is_leaf(self):
        '''
        Check if a tree is a leaf or not

        :return: (bool) Return True if tree is a leaf else False
        '''
        return not self.children()


    def lcs(self, str1, str2):
        str1 = str1.lower()
        str2 = str2.lower()
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        max_len = max(len(str1), len(str2))
        return dp[m][n] / max_len if max_len > 0 else 0


    def is_unordered_simi_match(self, otree):
        '''
        This method is used to check if `otree`, a tree object, is an exact match 
        with the tree object calling this method.

        :param otree: (TopSemanticTree/ExpressSemanticTree) Tree object to compare with.
        :return: (bool) If the two trees have an unordered exact match or not.  
        '''
        if not isinstance(otree, type(self)):
            raise TypeError(f"Expected both trees to be of type {type(self)} "
                f"but one of them is of type {type(otree)}")
        

        if self.lcs(self.root_symbol(), otree.root_symbol()) > 0.7:
            if len(self.children()) == 0 and len(otree.children()) == 0:
                return 1
        # check if the roots of trees are same. 
        # If not, its not an exact match.
        if self.root_symbol() != otree.root_symbol():
            return 0
        
        # create numpy zeros matrix of scores and fill it.
        scores = np.zeros((len(self.children()), len(otree.children())))
        for i in range(len(self.children())):
            for j in range(len(otree.children())):
                scores[i, j] = self.children()[i].is_unordered_simi_match(otree.children()[j])

        # pick the max match and remove its row/col and repeat until some axis is 0
        max_setup_scores = []
        
        while scores.shape[0] != 0 and scores.shape[1] != 0:
            max_idx = np.unravel_index(scores.argmax(), scores.shape)
            score = scores[max_idx[0], max_idx[1]]
            scores = np.delete(scores, [max_idx[0]], axis=0)
            scores = np.delete(scores, [max_idx[1]], axis=1)
            max_setup_scores.append(score)
        
        overall_score = max(0, sum(max_setup_scores) - max(scores.shape) + 1)
        return overall_score


    @abstractmethod
    def children(self):
        '''Get the children of the root calling this method'''

    @abstractmethod
    def _linearized_rep_to_tree_rep(self, flat_string):
        '''Get the tree representation for flat input string `flat_string`'''

class TopSemanticTree(SemanticTree):
    def __init__(self, *, flat_string=None, tree_rep=None, root_symbol=None, children=None):
        super(TopSemanticTree, self).__init__(flat_string=flat_string, tree_rep=tree_rep, root_symbol=root_symbol,
                                              children=children)

    def children(self):
        '''
        :return: (List) Return a list of TopSemanticTree objects that are children of `self` 
        '''
        return [TopSemanticTree(tree_rep=c) for c in self.tree_rep.children]

    @classmethod
    def get_semantics_only_tree(cls, tree_rep):
        '''
        Returns a class object by removing the non-semantic nodes from its tree 
        representation. 
        
        :param tree_rep: (AnyNode) A tree representation

        :return: (TopSemanticTree) A tree class object with the non-semantic nodes removed.
        '''
        tree_rep_ = cls.remove_non_semantic_nodes(tree_rep)
        return cls(tree_rep=tree_rep_)

    @staticmethod
    def remove_non_semantic_nodes(tree_rep):
        '''
        Method functionally removes the non-semantic nodes from a tree representation.

        :param: (AnyNode) Pointer to the input tree.
        :return: (AnyNode) Pointer to a new tree carrying only semantic nodes.
        '''
        # Check if all the children are terminal.
        if all(c.is_leaf for c in tree_rep.children):
            return AnyNode(id=tree_rep.id, children=[AnyNode(id=c.id) for c in tree_rep.children])

        # If the above check fails, filter the terminal children
        # and get the non-terminal child nodes.
        non_terminal_children = filter(lambda c: not c.is_leaf, tree_rep.children)
        new_children = [TopSemanticTree.remove_non_semantic_nodes(c) for c in non_terminal_children]
        
        return AnyNode(id=tree_rep.id, children=new_children)

    def _linearized_rep_to_tree_rep(self, flat_string):
        '''
        Get the tree representation for flat input string `flat_string`
        Example input string:
        "(ORDER can i have (PIZZAORDER (NUMBER a ) (SIZE large ) (TOPPING bbq pulled pork ) ) please )"

        Invalid flat strings include those with misplaced brackets, mismatched brackets,
        or semantic nodes with no children

        :param flat_string: (str) input flat string to construct a tree, if possible.
        :raises ValueError: when s is not a valid flat string
        :raises IndexError: when s is not a valid flat string
        
        :return: (AnyNode) returns a pointer to a tree node.
        '''
        # Keep track of all the semantics in the input string.
        semantic_stack = [AnyNode(id=TopSemanticTree.ROOT_SYMBOL)]

        for token in flat_string.split():
            if '(' in token:
                node = AnyNode(id=token.strip('('), parent=semantic_stack[-1])
                semantic_stack.append(node)
            elif token == ')':
                # If the string is not valid an error will be thrown here.
                # E.g. (PIZZAORDER (SIZE LARGE ) ) ) ) ) ) )
                try:
                    # If there are no children within this semantic node, throw an error
                    # E.g. (PIZZAORDER (SIZE LARGE ) (NOT ) )
                    if not semantic_stack[-1].children:
                        raise Exception("Semantic node with no children")
                    semantic_stack.pop()
                except Exception as e:
                    raise IndexError(e) from e
            else:
                AnyNode(id=token, parent=semantic_stack[-1])
        # If there are more than one elements in semantic stack, that means
        # the input string is malformed, i.e. it cant be used to construct a tree
        if len(semantic_stack) > 1:
            raise ValueError()
        return semantic_stack[-1]


class ExpressSemanticTree(SemanticTree):
    def __init__(self, *, flat_string=None, tree_rep=None, root_symbol=None, children=None):
        super(ExpressSemanticTree, self).__init__(flat_string=flat_string, tree_rep=tree_rep, root_symbol=root_symbol,
                                                  children=children)

    def children(self):
        '''
        Get the children for the root calling this method

        :return: (List) Return a list of ExpressSemanticTree objects that are children of `self` 
        '''
        return [ExpressSemanticTree(tree_rep=c) for c in self.tree_rep.children]

    @staticmethod
    def tokenize(flat_string):
        '''
        Tokenize EXR string
        Example input string:
        "(ORDER (PIZZAORDER (NUMBER 1) (TOPPING HAM) (COMPLEX_TOPPING (TOPPING ONIONS) (QUANTITY EXTRA))))"

        :param flat_string: (str) EXR-format input flat string to construct a tree

        :return: (list) A list of tokens obtained after tokenizing `flat_string`.
        '''
        special_characters = [',', '\n']
        return [t for t in re.split('([^a-zA-Z0-9._-])',flat_string) \
            if t and not(t.isspace()) and t not in special_characters]

    def _linearized_rep_to_tree_rep(self, flat_string):
        '''
        Get the tree representation for flat input string `flat_string`
        Example input string:
        "(ORDER (PIZZAORDER (NUMBER 1) (TOPPING HAM) (COMPLEX_TOPPING (TOPPING ONIONS) (QUANTITY EXTRA))))"

        :param flat_string: (str) EXR-format input flat string to construct a tree, if possible.
        :return: (AnyNode) returns a pointer to a tree.
        '''
        flat_string = f'({ExpressSemanticTree.ROOT_SYMBOL} {flat_string})'
        # Split the flat string using a regular expression and filter the special characters.
        toks = ExpressSemanticTree.tokenize(flat_string)
        parent_group_mapping = build_parent_group_mapping(toks)
        return parse_sexp(toks, 0, len(toks), parent_group_mapping)
