# encoding: utf-8
"""
Copyright (C) 2022 Anna L.D. Latour

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

@author: Anna L.D. Latour
@contact: latour@nus.edu.sg
@time: 4/22/22 5:42 PM
@file: identifying_codes.py
@desc: Contains the class for an Identifying Codes problem instance.
"""

import sys
from contextlib import suppress
from datetime import datetime
from itertools import combinations
import networkx as nx
import os
from scipy.io import mmread
import socket
import subprocess
from subprocess import PIPE

PBLIB_DIR = os.getenv('PBLIB_DIR')
VERITAS_PBLIB_DIR = os.getenv('VERITAS_PBLIB_DIR')
PROJECT_DIR = os.getenv('PROJECT_DIR')

def log_message(message):
    print('{date}: {message}'.format(
        date=datetime.now().strftime("%Y-%m-%d, %Hh%Mm%Ss"), message=message))
    sys.stdout.flush()

def cardinality_constraint(variables: list,
                           lb: int = None,
                           ub: int = None,
                           start_idx: int = -1,
                           infix = None,
                           clean_up = False) -> tuple:
    """Calls pblib to encode a cardinality constraint into CNF.
    :param start_idx: start index for the auxiliary variables
    :param variables: iterable with sympy Symbols
    :param lb: desired lower bound
    :param ub: desired upper bound
    :return: a Sympy encoding of a cardinality constraint on variables
    """
    assert lb is not None or ub is not None, "Specify upper bound and/or lower bound for cardinality constraint."
    assert start_idx > len(variables), "Specify a start index for the auxiliary variables."

    nvars = len(variables)
    var_2_pblib_index = {var: idx + 1 for idx, var in enumerate(sorted(variables))}
    pblib_index_2_var = {idx: var for var, idx in var_2_pblib_index.items()}
    pbs = ''    # pblib input string

    # Create pblib input string and write it to a temporary pblib input file
    if ub is not None and lb is None:
        pbs = ['* #variable= {nvars} #constraint= 1\n'.format(nvars=nvars),
               '* \n',  # pblib parser breaks if I don't include this. Don't know why.
               ' '.join(['-1 x{i}'.format(i=i + 1) for i, _ in enumerate(variables)]) + ' >= -{ub};'.format(
                   ub=ub)
               ]
    elif ub is None and lb is not None:
        pbs = ['* #variable= {nvars} #constraint= 1\n'.format(nvars=nvars),
               '* \n',  # pblib parser breaks if I don't include this. Don't know why.
               ' '.join(['+1 x{i}'.format(i=i + 1) for i, _ in enumerate(variables)]) + ' >= {lb};'.format(
                   lb=lb)
               ]
    elif ub == lb:
        pbs = ['* #variable= {nvars} #constraint= 1\n'.format(nvars=nvars),
               '* \n',  # pblib parser breaks if I don't include this. Don't know why.
               ' '.join(['+1 x{i}'.format(i=i + 1) for i, _ in enumerate(variables)]) + ' = {ub};'.format(ub=ub)
               ]

    cwd = os.getcwd()
    temp_pbs_pbo = cwd + '/TEMP_' + infix + '_pbs.pbo'
    temp_pbs_cnf = cwd + '/TEMP_' + infix + '_pbs.cnf'

    with open(temp_pbs_pbo, 'w') as pbsf:
        pbsf.writelines(pbs)

    # Call pblib and write the result to a temporary DIMACS file
    with open(temp_pbs_cnf, 'w') as temp_cnf_file:
        subprocess.call([PBLIB_DIR + '/pbencoder', temp_pbs_pbo],
                       stdout=temp_cnf_file)

    # Parse the DIMACS file that is created by pblib, so it can be
    # integrated in the Sympy formula that we are building
    with open(temp_pbs_cnf, 'r') as temp_cnf_file:
        clauses = []
        n_vars_card_cnf = 0
        for line in temp_cnf_file.readlines():
            if line.startswith('p'):
                _, _, nvars_str, _ = line.split()
                n_vars_card_cnf = int(nvars_str)
                continue
            lits = [int(lit) for lit in line.split()[:-1]]
            clauses.append(lits)  # each line ends in 0

    if clean_up:
        if os.path.exists(temp_pbs_pbo):
            os.remove(temp_pbs_pbo)
        if os.path.exists(temp_pbs_cnf):
            os.remove(temp_pbs_cnf)

    # The CNF encoding of a cardinality constraint creates auxiliary
    # variables that must be re-indexed and mapped to logical symbols
    for var in range(1, n_vars_card_cnf + 1):
        if var not in pblib_index_2_var:
            pblib_index_2_var[var] = start_idx
            var_2_pblib_index[start_idx] = var
            start_idx += 1
    reindexed_clauses = []
    for clause in clauses:
        reindexed_clauses.append(' '.join([str(pblib_index_2_var[lit])
                                           if lit > 0
                                           else str(-pblib_index_2_var[-lit])
                                           for lit in clause]))
    return reindexed_clauses, max(start_idx-1, max(var_2_pblib_index))


def check_datatype(network_file):
    with open(network_file, 'r') as infile:
    # with gzip.open(network_file, 'rt', encoding='utf-8') as infile:
        for line in infile.readlines():
            if line.startswith('%') \
                    or line.startswith('#') \
                    or ('mtx' in network_file and len(line.split()) == 3):
                continue
            else:
                s, t = line.split()
                if s.isdigit() and t.isdigit():
                    return "int"
    return "str"


def twin_removal(G):
    """ Merge nodes that are twins into one node. Code inspired by
    https://github.com/kaustav-basu/IdentifyingCodes/blob/master/ilp.py
    Right now, it is implemented in a roundabout way because the big matrices
    in the original code require too much memory.
    :param G: Input graph
    :return:  (H, d), where H is a graph which is G but with twins removed, and
              d is a dictionary mapping nodes to their set of twins.
    """
    twins = dict()
    for node in G.nodes():
        node_neighbourhood = set(G.neighbors(node))
        node_neighbourhood.add(node)
        for neighbour in node_neighbourhood:
            if node < neighbour:
                neighbour_neighbourhood = set(G.neighbors(neighbour))
                neighbour_neighbourhood.add(neighbour)
                if node_neighbourhood == neighbour_neighbourhood:
                    if node in twins:
                        twins[node].add(neighbour)
                    else:
                        twins[node] = {node, neighbour}
                    if neighbour in twins:
                        twins[neighbour].add(node)
                    else:
                        twins[neighbour] = {node, neighbour}

    twin_nodes = sorted(list(twins.keys()))
    for node in twin_nodes:
        if node in twins:
            neighbours = twins[node]
            for neighbour in neighbours:
                if neighbour is not node:
                    if neighbour in twins:
                        del twins[neighbour]
                        G.remove_node(neighbour)
    return G, twins


def prepend_multiple_lines(file_name, list_of_lines):
    """Insert given list of strings as a new lines at the beginning of a file.
    Code from https://thispointer.com/python-how-to-insert-lines-at-the-top-of-a-file/
    """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open given original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Iterate over the given list of strings and write them to dummy file as lines
        for line in list_of_lines:
            write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)

class IdentifyingCodesInstance:
    def __init__(self):

        self._network_file = None
        self._k = None
        self._budget = None
        self._two_step = None

        self._G = None
        self._twins = dict()
        self._node_2_label = dict()
        self._label_2_node = dict()

        self._n_vars = None

    def build_from_file(self,
                        network_file,
                        budget=-1,
                        two_step=False):
        """

        :param network_file: edge list or mtx file describing a network
        :param budget:       maximum number of sensors to place
        :param k:            list of maximum identifiable set sizes
        :param two_step:     True if using two_step encoding
        :return:             None
        """

        self._network_file = network_file
        print("network file: ", self._network_file)
        self._two_step = two_step
        print("two_step?", self._two_step)

        if '.mtx' in network_file:
            print("Creating from mtx file")
            self._create_from_mtx_file()
        else:
            print("Creating from edge list")
            self._create_from_edge_list()
        self._preprocess_graph()
        self._n_vars = self._G.number_of_nodes()
        self._budget = budget

    def _create_from_edge_list(self):
        with open(self._network_file, 'r') as infile:
        # with gzip.open(self._network_file, 'rt', encoding='utf-8') as infile:
            edges = [tuple(line.split()[:2])
                     for line in infile.readlines()
                     if not (line.startswith('#') or line.startswith('%'))]
            self._G = nx.Graph()
            self._G.add_edges_from(edges)

    def _create_from_mtx_file(self):
        self._G = nx.Graph(mmread(self._network_file))

    def _preprocess_graph(self):
        # In de 1-step setting, we can only guarantee the existence of a
        # solution if there are no twins in the graph.
        if not self._two_step:
            self._G, self._twins = twin_removal(self._G)

        # Make sure that node names are consecutive indices, starting at 1
        # and ending at self._G.number_of_nodes(). Also create mapping
        # from original node names to new labels and vice versa:
        self._label_2_node = {label: idx + 1 for idx, label in enumerate(sorted(self._G.nodes()))}
        self._node_2_label = {idx: label for label, idx in self._label_2_node.items()}
        self._G = nx.relabel_nodes(self._G, self._label_2_node)

    def _get_header(self, encoding=None, k=1, remove_supersets=False, check_2_neighbourhood=False):
        """
        Generates a list of strings that form the header of the dimacs file,
        documenting some basic info about the input graph and its encoding into
        CNF/dimacs.
        :param encoding: Specifies if it's ILP, MaxSAT, SAT or Independent Support
        :return:         List of strings, each string a line in the header
        """
        header = [
            '',
            'NETWORK DATA',
            '------------',
            'Network file:     {f}'.format(f=self._network_file),
            'Twins removed?    {a}'.format(a="yes" if (not self._two_step and self._twins) else "no"),
            'Number of nodes (after preprocess): {n}'.format(n=self._G.number_of_nodes()),
            'Number of edges (after preprocess): {n}'.format(n=self._G.number_of_edges()),
            '', '',
            'PROBLEM PARAMETERS',
            '------------------',
            'Budget:            {a}'.format(a="N/A" if self._budget == -1 else self._budget),
            'k:                 {k}'.format(k="N/A" if self._budget > -1 else k),
            'Encoding:          {e}'.format(e=encoding),
            'Approach:          {a}'.format(a="two-step" if self._two_step else "one-step"),
        ]
        if encoding.lower() == 'ilp':
            header += [
            'Remove supersets:  {r}'.format(r=remove_supersets),
                'Check 2 neighbourhood: {c}'.format(c=check_2_neighbourhood)
                ]
        elif encoding.lower() == 'pb':
            if VERITAS_PBLIB_DIR is not None:
                res = subprocess.check_output('git --git-dir {VERITAS_PBLIB_DIR}/.git config --get remote.origin.url'.format(
                    VERITAS_PBLIB_DIR=VERITAS_PBLIB_DIR), stderr=subprocess.STDOUT, shell=True)
                repo = res.decode()[:-1]
                res = subprocess.check_output('git --git-dir {VERITAS_PBLIB_DIR}/.git branch'.format(
                    VERITAS_PBLIB_DIR=VERITAS_PBLIB_DIR), stderr=subprocess.STDOUT, shell=True)
                branch = res.decode()[2:-1]
                res = subprocess.check_output('git --git-dir {VERITAS_PBLIB_DIR}/.git log --format="%H" -n 1'.format(
                    VERITAS_PBLIB_DIR=VERITAS_PBLIB_DIR), stderr=subprocess.STDOUT, shell=True)
                commit = res.decode()[:-1]
                header += [
                    '',
                    '',
                    'ENCODING INFO',
                    '-------------',
                    'Encoding:          {e}'.format(e=encoding),
                    'Approach:          {a}'.format(a="two-step" if self._two_step else "one-step"),
                    'VeritasPBLib repo:   {r}'.format(r=repo),
                    'VeritasPBLib branch: {b}'.format(b=branch),
                    'VeritasPBLib commit: {c}'.format(c=commit),
                ]

        header += [
            '', '',
            'REPRODUCIBILITY INFO',
            '--------------------',
            'Generated with:    {s}'.format(s=os.path.basename(__file__)),
        ]
        # TODO: check for existence of .git
        if PROJECT_DIR is not None:
            res = subprocess.check_output('git --git-dir {PROJECT_DIR}/../.git config --get remote.origin.url'.format(
                PROJECT_DIR=PROJECT_DIR), stderr=subprocess.STDOUT, shell=True)
            repo = res.decode()[:-1]
            res = subprocess.check_output('git --git-dir {PROJECT_DIR}/../.git branch'.format(
                PROJECT_DIR=PROJECT_DIR), stderr=subprocess.STDOUT, shell=True)
            branch = res.decode()[2:-1]
            res = subprocess.check_output('git --git-dir {PROJECT_DIR}/../.git log --format="%H" -n 1'.format(
                PROJECT_DIR=PROJECT_DIR), stderr=subprocess.STDOUT, shell=True)
            commit = res.decode()[:-1]
            header += [
                'Repository:        {r}'.format(r=repo),
                'Branch:            {b}'.format(b=branch),
                'Commit:            {c}'.format(c=commit),
                'Machine:           {m}'.format(m=socket.gethostname()),
            ]
        header += [
            'Date (YYYY-MM-DD): {d}'.format(d=datetime.now().strftime("%Y-%m-%d")),
            ''
        ]
        header += self._get_label_map()
        return header

    def _get_label_map(self):
        width = max(len(max([str(label) for label in self._node_2_label.values()], key=len)), 14)
        label_map = ['',
                     'VARIABLE MAP',
                     '------------',
                     '',
                     '{var:>10} {label:>{width}}'.format(var='variable', width=width, label='original name'),
                     '-' * (11 + width)]
        for idx, label in self._node_2_label.items():
            label_map.append('{var:>10} {label:>{width}}'.format(var=str(idx), width=width, label=label))
        label_map.append('')

        twin_map = []
        if self._twins:
            width = max(len(max([str(label) for label in self._twins.keys()], key=len)), 17)
            twin_map = ['',
                        'TWIN MAP',
                        '--------',
                        '',
                        '{node:>{width}}  {twin:>{width}}'.format(node='node name', width=width, twin='replaced by twin'),
                        '-' * (2+2*width)]
            for node, twins in self._twins.items():
                for twin in twins:
                    if node is not twin:
                        twin_map.append('{twin:>{width}}  {node:>{width}}'.format(twin=twin, width=width, node=node))
            twin_map.append('')
        return label_map + twin_map





