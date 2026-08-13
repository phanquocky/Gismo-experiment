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
@file: ilp_encoding.py
@desc: Class for encoding an Identifying Codes instance as an ILP problem.
"""

from contextlib import suppress
import cplex
from identifying_codes import IdentifyingCodesInstance, \
    log_message, prepend_multiple_lines
from itertools import combinations
import networkx as nx
import sys

class ILPEncoding(IdentifyingCodesInstance):

    def __init__(self, two_step=False):
        IdentifyingCodesInstance.__init__(self)
        self._detection_vars = None
        self._fire_vars = None
        self._two_step = two_step

        self._n_vars = 0
        self._n_csts = 0

        # Initialise model
        self._ilp_enc = None
        self._node_vars = []

    def encode(self, lp_file, k, remove_supersets=False, check_2_neighbourhood=False):
        log_message("{classname}: Start encoding".format(classname=self.__class__.__name__))
        if self._two_step:
            self.encode_two_step(lp_file, k, remove_supersets=remove_supersets, check_2_neighbourhood=check_2_neighbourhood)
        else:
            self.encode_one_step(lp_file, k)

    def _one_step_detection_constraint(self):
        """
        Each fire should be detectable.
        :return:
        """
        rows = []
        for node in self._G.nodes():
            neighbourhood = list(self._G.neighbors(node)) + [node]
            bvars = ['x' + str(node) for node in sorted(set(neighbourhood))]
            coeff = [1] * len(bvars)
            rows.append([bvars, coeff])
        senses = 'G' * len(rows)
        rhs = [1] * len(rows)
        names = ['d' + str(i) for i in range(len(rows))]
        return rows, senses, rhs, names

    def _one_step_uniqueness_constraint(self):
        identity_constraints = set()
        rows = []
        for node in self._G.nodes():
            # The 2-neighbourhood contains all neighbours at a distance of
            # 1 or 2 from node.
            neighbourhood = nx.ego_graph(self._G, node, 2).nodes()

            for neighbour in neighbourhood:
                pair = tuple(sorted([node, neighbour]))

                # Check that this node pair doesn't already have a constraint.
                # If it doesn't, create it.
                if node is not neighbour and pair not in identity_constraints:
                    # Get the complete 1-neighbourhood of each node in the pair
                    # (such that the node itself is included)
                    N0 = set(self._G.neighbors(pair[0])).union({pair[0]})
                    N1 = set(self._G.neighbors(pair[1])).union({pair[1]})
                    # Get the difference between the neighbourhoods:
                    distinguishing_set = N0.symmetric_difference(N1)

                    # Encode that at least one node in the distinguishing set
                    # must have a colour injected / a sensor placed on it,
                    # in order for the node's and the neighbour's signatures to
                    # be different:
                    bvars = ['x' + str(n) for n in sorted(distinguishing_set)]
                    coeff = [1] * len(bvars)
                    rows.append([bvars, coeff])
                    identity_constraints.add(pair)
        senses = 'G' * len(rows)
        rhs = [1] * len(rows)
        names = ['i' + str(i) for i in range(len(rows))]
        return rows, senses, rhs, names

    def _objective_function(self):
        return self._ilp_enc.sum(self._node_vars)

    def encode_one_step(self, lp_file, k):
        self._k = k
        self._ilp_enc = cplex.Cplex()

        # Define variables and their bounds
        varnames = ['x' + str(node) for node in self._G.nodes()]
        vartypes = [self._ilp_enc.variables.type.binary] * len(varnames)
        obj_coeff = [1] * len(varnames)
        lbs = [0] * len(varnames)
        ubs = [1] * len(varnames)
        self._ilp_enc.variables.add(obj=obj_coeff, lb=lbs, ub=ubs, names=varnames, types=vartypes)
        self._node_vars = varnames
        self._n_vars = len(varnames)

        d_rows, d_senses, d_rhs, d_names = self._one_step_detection_constraint()
        i_rows, i_senses, i_rhs, i_names = self._one_step_uniqueness_constraint()

        rows = d_rows + i_rows
        senses = d_senses + i_senses
        rhs = d_rhs + i_rhs
        names = d_names + i_names

        self._ilp_enc.linear_constraints.add(lin_expr=rows, senses=senses,
                                             rhs=rhs, names=names)
        self._n_csts = len(rows)

        # Write model to file
        if lp_file.endswith('.lp.gz'):
            lp_file = lp_file[:-3]
        self._ilp_enc.write(lp_file)

        # Get header
        header = self._get_header(encoding="ILP", k=k)
        lines = ['\ ' + line for line in header]

        # Add header to the top of the model file
        prepend_multiple_lines(lp_file, lines)

    def _two_step_detection_constraint(self):
        """
        For each detection variable y_v, we must encode the following constraint:
            y_v = sum_{u in N_1^+(v)} x_u
        This can be translated in the following linear constraints:
            {y_v - sum_{u in N_1^+(v)} x_u == 0
        :return:
        """
        rows = []
        for node in self._G.nodes():
            neighbourhood = nx.ego_graph(self._G, node, radius=1, center=True).nodes()
            bvars = ['y' + str(node)] + ['x' + str(node) for node in sorted(set(neighbourhood))]
            coeff = [1] + [-1] * len(neighbourhood)
            rows.append([bvars, coeff])
        senses = 'E' * len(rows)
        rhs = [0] * len(rows)
        names = ['d' + str(i) for i in range(len(rows))]
        return rows, senses, rhs, names

    def _two_step_alo_constraint(self):
        """
        Each node must have a colour, so for each node v, we must encode the
        following constraint:
            y_v >= 1
        :return:
        """
        bvars_list = ['y' + str(node) for node in self._G.nodes()]
        coeff_list = [1] * self._G.number_of_nodes()
        rows = [[[bvars], [coeff]] for bvars, coeff in zip(bvars_list, coeff_list)]
        senses = 'G' * len(rows)
        rhs = [1] * len(rows)
        names = ['a' + str(i) for i in range(len(rows))]
        return rows, senses, rhs, names

    def _get_set_neighbourhood(self, the_set, distance, closed=False):
        """

        :param the_set:
        :param distance:
        :param closed:
        :return:
        """
        the_neighbourhood = set()
        if distance == 1:
            for node in the_set:
                the_neighbourhood = the_neighbourhood.union(self._1_neighbourhoods[node])
            if closed:
                the_neighbourhood = the_neighbourhood.union(the_set)
        elif distance == 2:
            assert len(self._2_neighbourhoods) > 0
            for node in the_set:
                the_neighbourhood = the_neighbourhood.union(self._2_neighbourhoods[node])
            if closed:
                the_neighbourhood = the_neighbourhood.union(the_set)
        else:
            for node in the_set:
                neighbourhood = nx.ego_graph(self._G, node, radius=distance, center=closed).nodes()
                the_neighbourhood = the_neighbourhood.union(neighbourhood)
        return the_neighbourhood

    def _two_step_uniqueness_constraint(self,
                                        k=1,
                                        remove_supersets=True,
                                        check_2_neighbourhood=True):
        """ For all subsets U and W of the nodes in the graph, with |U| <= k
        and |W| <= k, create the following constraint, which guarantees the
        uniqueness of the signature:

            sum_{z \in U \cup W \cup (N_1^+(U) \triangle N_1^+(W))} x_z >= 1

        Here, \triangle means the symmetric difference between sets.
        The symmetric difference U \triangle W can make the first element of
        the signature unique.
        The symmetric difference N_1^+(U) \triangle N_1^+(W) can make the second
        element of the signature unique. Here, N_1^+(U) is the complete
        1-neighbourhood of the set U.

        Note that many of the U \cup W \cup (N_1^+(U) \triangle N_1^+(W)
        sets may overlap, so we remove duplicates.
        NOTE: 15 oct 2022: I think the above set may be wrong. I think it should be (U \triangle W) \cup ((N_1^+(U) \triangle N_1^+(W)), and I'm changing the code accordingly

        Let's say that Z = U \cup W \cup (N_1^+(U) \triangle N_1^+(W).
        Note that if sum_{z in Z} x_z >= 1 holds, then sum_{z in Z'} x_z >= 1
        also holds, with Z' any superset of Z. We can therefore reduce the
        number of constraints by also removing all supersets.

        :param k:   Maximum identifiable set size (maximum number of
                    simultaneous events).
        :param remove_supersets:    Optimisation to eliminate larger constraints
                                    that must be satisfied if smaller
                                    constraints are satisfied.
        :param check_2_neighbourhood:   Optimisation to limit number of
                                        constraints.
        :return:
        """
        ds_sigs = set()
        n_nodes = self._G.number_of_nodes()
        # Do a bit of preprocessing
        self._1_neighbourhoods = {
            node: frozenset(nx.ego_graph(self._G, node, radius=1, center=False).nodes())
            for node in self._G.nodes()
        }
        self._2_neighbourhoods = dict()
        if check_2_neighbourhood:
            self._2_neighbourhoods = {
                node: frozenset(nx.ego_graph(self._G, node, radius=2, center=False).nodes())
                for node in self._G.nodes()
            }
        # Iterate over all possible cardinalities of set U
        for U_size in range(1, k + 1):
            # Generate all sets U of cardinality U_size
            for U in combinations(range(1, n_nodes + 1), U_size):
                # 1-neighbourhood of set U (works for both closed and not closed)
                # NOTE: 15 oct 2022: I can't remember why it should work for both, changing to closed.
                # NOTE: 15 oct 2022: I think I made a mistake earlier, we need
                # (U triangle W) cup (N_1(U) triangle N_1(W))
                # The neighbourhood indeed can be either closed or not, doesn't matter.
                N1_U = self._get_set_neighbourhood(U, 1, closed=False)
                if check_2_neighbourhood:
                    N2_U = self._get_set_neighbourhood(U, 2, closed=True)    # closed 2-neighbourhood of set U

                # Iterate over all possible cardinalities of set W:
                for W_size in range(U_size, k + 1):
                    # Generate all sets W of cardinality W_size
                    for W in combinations(range(1, n_nodes + 1), W_size):
                        if U == W:
                            continue

                        # Determine the distinguishing set for the first element
                        # of the signature:
                        # ds_sig0 = set(U).union(set(W)) # NOTE: 15 oct 202: I think this line is wrong. I think it should be symmetric difference. Changing it now
                        ds_sig0 = set(U).symmetric_difference(set(W))

                        # Get the neighbourhoods for set W
                        # 1-neighbourhood of set W (works for both closed and not closed)
                        N1_W = self._get_set_neighbourhood(W, 1, closed=False)
                        if check_2_neighbourhood:
                            N2_W = self._get_set_neighbourhood(W, 2, closed=True)  # closed 2-neighbourhood of set W

                        # Avoid adding unnecessary constraints
                        if check_2_neighbourhood:
                            intersection_U_and_W = N2_U.intersection(N2_W)
                            if len(intersection_U_and_W) == 0:
                                continue

                        # Determine the distinguishing set of the second element
                        # of the signature:
                        ds_sig1 = N1_U.symmetric_difference(N1_W)

                        # Determine the entire distinguishing set for this
                        # (U, W) pair
                        ds_full_sig = frozenset(ds_sig0.union(ds_sig1))

                        # Avoid adding a distinguishing set that is a super set
                        # of a distinguishing set that we already have
                        if remove_supersets:
                            # Check if the new signature is a superset of an
                            # existing signature
                            is_superset = False
                            for ds_sig in ds_sigs:
                                if ds_full_sig >= ds_sig:
                                    is_superset = True
                                    break

                            # If it is not, then check which existing signatures
                            # are supersets of the new one, create a set of
                            # existing supersets to remove, and then remove
                            # them.
                            if not is_superset:
                                to_remove = set()
                                for ds_sig in ds_sigs:
                                    if ds_sig >= ds_full_sig:
                                        to_remove.add(ds_sig)
                                for removable_sig in to_remove:
                                    with suppress(ValueError, AttributeError):
                                        ds_sigs.remove(removable_sig)

                        # Add the distinguishing set to the set of
                        # distinguishing sets that will be constraints:
                        if not remove_supersets \
                                or (remove_supersets and not is_superset) \
                                and len(ds_full_sig) > 0:
                            ds_sigs.add(ds_full_sig)


        bvars_list = [tuple(['x' + str(node) for node in ds_sig]) for ds_sig in ds_sigs]
        rows = [[bvars, [1] * len(bvars)] for bvars in bvars_list]
        senses = 'G' * len(rows)
        rhs = [1] * len(rows)
        names = ['u' + str(i) for i in range(len(rows))]
        return rows, senses, rhs, names

    def encode_two_step(self, lp_file, k, remove_supersets=False, check_2_neighbourhood=False):
        log_message("{classname}: Start two-step encoding".format(classname=self.__class__.__name__))

        self._ilp_enc = cplex.Cplex()
        log_message("{classname}: Initialised CPLEX".format(classname=self.__class__.__name__))

        # Define variables and their bounds

        # Define fire variables (first element of signature):
        varnames_X = ['x' + str(node) for node in range(1, self._G.number_of_nodes() + 1)]
        vartypes_X = [self._ilp_enc.variables.type.binary] * len(varnames_X)
        obj_coeff_X = [1] * len(varnames_X)     # these variables are part of the objective function
        lbs_X = [0] * len(varnames_X)
        ubs_X = [1] * len(varnames_X)
        self._ilp_enc.variables.add(
            obj=obj_coeff_X, lb=lbs_X, ub=ubs_X, names=varnames_X, types=vartypes_X)
        self._fire_vars = varnames_X

        # Define detection variables (second element of signature)
        varnames_Y = ['y' + str(node) for node in range(1, self._G.number_of_nodes() + 1)]
        vartypes_Y = [self._ilp_enc.variables.type.integer] * len(varnames_Y)
        obj_coeff_Y = [0] * len(varnames_Y)         # these variables are not part of the objective function
        lbs_Y = [0] * len(varnames_Y)
        ubs_Y = [len(nx.ego_graph(self._G, node, radius=1, center=True).nodes())
                 for node in sorted(self._G.nodes())]
        self._ilp_enc.variables.add(
            obj=obj_coeff_Y, lb=lbs_Y, ub=ubs_Y, names=varnames_Y, types=vartypes_Y)
        self._detection_vars = varnames_Y

        self._n_vars = len(varnames_X) + len(varnames_Y)
        log_message("{classname}: Initialised variables and domains.".format(classname=self.__class__.__name__))

        a_rows, a_senses, a_rhs, a_names = self._two_step_alo_constraint()
        log_message("{classname}: Generated alo constraints.".format(classname=self.__class__.__name__))
        d_rows, d_senses, d_rhs, d_names = self._two_step_detection_constraint()
        log_message("{classname}: Generated two-step detection constraints.".format(classname=self.__class__.__name__))
        u_rows, u_senses, u_rhs, u_names = self._two_step_uniqueness_constraint(k=k, remove_supersets=remove_supersets, check_2_neighbourhood=check_2_neighbourhood)
        log_message("{classname}: Generated two-step uniqueness constraints.".format(classname=self.__class__.__name__))

        self._n_csts = len(a_rows) + len(d_rows) + len(u_rows)

        log_message("{classname}: number of constraints generated:.".format(classname=self.__class__.__name__))
        log_message("{classname}: {n_a} alo constraints.".format(classname=self.__class__.__name__, n_a=len(a_rows)))
        log_message("{classname}: {n_d} two-step detection constraints.".format(classname=self.__class__.__name__, n_d=len(d_rows)))
        log_message("{classname}: {n_u} two-step uniqueness constraints.".format(classname=self.__class__.__name__, n_u=len(u_rows)))

        self._ilp_enc.linear_constraints.add(
            lin_expr=a_rows + d_rows + u_rows,
            senses=a_senses + d_senses + u_senses,
            rhs=a_rhs + d_rhs + u_rhs,
            names=a_names + d_names + u_names)
        log_message("{classname}: Added constraints to model.".format(classname=self.__class__.__name__))



        # Write model to file
        if lp_file.endswith('.lp.gz'):
            lp_file = lp_file[:-3]
        self._ilp_enc.write(lp_file)
        log_message("{classname}: Wrote model to file {lp_file}".format(
            classname=self.__class__.__name__, lp_file=lp_file))


        # Get header
        header = self._get_header(encoding="ILP",
                                  k=k,
                                  remove_supersets=remove_supersets,
                                  check_2_neighbourhood=check_2_neighbourhood)
        lines = ['\ ' + line for line in header]
        log_message("{classname}: Generated header.".format(classname=self.__class__.__name__))

        # Add header to the top of the model file
        prepend_multiple_lines(lp_file, lines)
        log_message("{classname}: Prepended header to model file.".format(classname=self.__class__.__name__))