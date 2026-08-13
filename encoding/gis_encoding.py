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
@time: 11 Oct 2022
@file: gis_encoding.py
@desc: Class for encoding an Identifying Codes instance into a CNF that can be
       used to solve the instance by finding a minimal GIS.
"""

from identifying_codes import IdentifyingCodesInstance, cardinality_constraint
import gzip
import networkx as nx
import os

class GISEncoding(IdentifyingCodesInstance):

    def __init__(self, two_step=False):
        """

        :param two_step:    True if using two-step encoding, which implies use
                            of Grouped Independent Support
        """
        IdentifyingCodesInstance.__init__(self)
        self._n_vars = 0
        self._n_clss = 0

        self._fire_vars = []
        self._detector_vars = []

        self._encoded_detection = False
        self._detection_clauses = []
        self._two_step = two_step

    def encode(self, dimacs_file, k):
        n = self._G.number_of_nodes()

        if not self._encoded_detection:
            assert (len(self._detection_clauses) == 0) and \
                   (len(self._fire_vars) == 0) and \
                   (len(self._detector_vars) == 0)
            # Define "fire" variables
            self._fire_vars = sorted(list(self._G.nodes()))
            # Define "detection" variables
            self._detector_vars = list(range(n + 1, 2 * n + 1))
            # Encode the detection constraints
            self._detection_clauses = self._detection_constraints()
            self._encoded_detection = True

        # (re)set the number of interesting variables, in anticipation of the
        # auxiliary variables created by the cardinality constraint
        self._n_vars = 2*n

        # Create cardinality constraint, possibly updating the number of
        # variables
        cardinality_clauses, self._n_vars = cardinality_constraint(
            self._fire_vars,
            ub=k,
            start_idx=self._n_vars+1,
            infix=os.path.basename(self._network_file))

        self._n_clss = len(cardinality_clauses) + len(self._detection_clauses)

        # Create DIMACS header
        header = self._get_header(encoding='independent support', k=k)

        # Define the sets of variables for computing independent support:
        #   - ind = the set from which to draw variables for the independent support
        #   - defined = the set of variables that must be defined by the independent support
        #   - groups = a set of sets of variables that are to be grouped together
        #              in the computation of the independent support
        ind = []
        defined = self._fire_vars
        groups = []
        if self._two_step:
            ind = self._fire_vars + self._detector_vars
            groups = list(zip(self._fire_vars, self._detector_vars))
        else:
            ind = self._detector_vars
            defined = self._fire_vars

        # Write to file
        self._write_2_dimacs(
            dimacs_file,
            clauses=cardinality_clauses + self._detection_clauses,
            ind=ind, defined=defined, groups=groups, header=header)

    def _detection_constraints(self):
        """
        For each detector variable, we encode the conditions in which it goes off,
        namely: iff there's a fire in its room or in one of its neighbouring
        rooms. Thus, for each node n, with fire variable X_n and detector
        variable Y_n, we have a clause
            -Y_n V X_n V X_1 V X_2 V ...,
        where X_1, X_2, ... are the fire variables for n's neighbours. We also
        have corresponding clauses
            Y_n V -X_n
            Y_n V -X_1
            Y_n V -X_2
            ...
        to indicate the other direction of the implication.
        :return: a list of strings, where each string is a clause
        """
        clauses = []
        # node ranges from 1 to number_of_nodes
        # the corresponding detector variable is always 2*node
        for node in self._G.nodes():
            # Get complete 1-neighbourhood of node (this includes node itself)
            neighbourhood = list(nx.ego_graph(self._G, node, radius=1, center=True).nodes())
            # Create the long clause
            clauses.append(str(-self._detector_vars[node-1]) + ' ' + ' '.join(
                [str(neighbour) for neighbour in neighbourhood]))
            # Create the binary clauses
            for neighbour in neighbourhood:
                clauses.append(str(self._detector_vars[node-1]) + ' ' + str(-neighbour))
        return clauses

    def _write_2_dimacs(self, dimacs_file, header=None, clauses=None,
                        ind=None, defined=None, groups=None):
        """
        Write CNF for independent support encoding to DIMACS format.
        :param dimacs_file: .cnf file to write the CNF formula to
        :param header:      header with basic info about the file
        :param clauses:     list of strings, each string a clause
        :param ind:         list of variables from which to draw independent support
        :param defined:     list of variables to be defined by independent support
        :param groups:      list of sets of variables for grouped independent support
        :return:            None
        """
        if header is None:
            header = []
        if clauses is None:
            clauses = []
        if ind is None:
            ind = []
        if defined is None:
            defined = []
        if groups is None:
            groups = []
        dimacs = ['c ' + line + '\n' for line in header]
        dimacs.append('p cnf {nvars} {nclss}\n'.format(nvars=self._n_vars, nclss=self._n_clss))
        dimacs.append('c def ' + ' '.join([str(var) for var in defined]) + ' 0\n')
        dimacs.append('c ind ' + ' '.join([str(var) for var in ind]) + ' 0\n')
        for group in groups:
            dimacs.append('c grp ' + ' '.join([str(var) for var in group]) + ' 0\n')
        dimacs.extend(['{cls} 0\n'.format(cls=cls) for cls in clauses])
        with open(dimacs_file, 'w') as d_file:
        # with gzip.open(dimacs_file, 'wb') as d_file:
            print("Writing dimacs to", dimacs_file)
            d_file.write(''.join(dimacs))


