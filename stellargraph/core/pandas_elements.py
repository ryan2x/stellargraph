# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sps

from .validation import require_dataframe_has_columns
from ..globalvar import TYPE_ATTR_NAME, SOURCE, TARGET, WEIGHT


class SingleTypeData:
    def __init__(self, shared, features):
        if not isinstance(features, (np.ndarray, sps.spmatrix)):
            raise TypeError(
                f"features: expected numpy or scipy array, found {type(features)}"
            )
        if not isinstance(shared, pd.DataFrame):
            raise TypeError(f"shared: expected pandas DataFrame, found {type(shared)}")

        if len(features.shape) != 2:
            raise ValueError(
                f"expected features to be 2 dimensional, found {len(features.shape)}"
            )

        rows, _columns = features.shape
        if len(shared) != rows:
            raise ValueError(
                f"expected one ID per feature row, found {len(shared)} IDs and {rows} feature rows"
            )

        self.shared = shared
        self.features = features


def _index(single_type_data):
    type_start_index = {}
    rows_so_far = 0
    type_dfs = []
    for type_name, type_data in single_type_data.items():
        type_start_index[type_name] = rows_so_far
        rows_so_far += len(type_data.shared)

        type_dfs.append(type_data.shared.assign(**{TYPE_ATTR_NAME: type_name}))

    id_to_type = pd.concat(type_dfs)

    idx = id_to_type.index
    if not idx.is_unique:
        # had some duplicated IDs, which is an error
        duplicated = idx[idx.duplicated()].unique()

        count = len(duplicated)
        assert count > 0
        # in a large graph, printing all duplicated IDs might be rather too many
        limit = 20

        rendered = ", ".join(x for x in duplicated[:limit])
        continuation = f", ... ({count - limit} more)" if count > limit else ""

        raise ValueError(
            f"expected IDs to appear once, found some that appeared more: {rendered}{continuation}"
        )

    return type_start_index, id_to_type


class ElementData:
    def __init__(self, features):
        if not isinstance(features, dict):
            raise TypeError(f"features: expected dict, found {type(features)}")

        for key, value in features.items():
            if not isinstance(value, SingleTypeData):
                raise TypeError(
                    f"features[{key!r}]: expected 'SingleTypeData', found {type(value)}"
                )

        self._features = {
            type_name: type_data.features for type_name, type_data in features.items()
        }
        self._type_start_indices, self._id_to_type = _index(features)

    def __len__(self):
        return len(self._id_to_type)

    def __contains__(self, item):
        return item in self._id_to_type.index

    def ids(self):
        """
        Returns:
             All of the IDs of these elements.
        """
        return self._id_to_type.index

    def types(self):
        """
        Returns:
             All of the types of these elements.
        """
        return self._features.keys()

    def type(self, ids):
        """
        Return the types of the ID(s)

        Args:
            ids (Any or Iterable): a single ID of an element, or an iterable of IDs of eleeents

        Returns:
             A sequence of types, corresponding to each of the ID(s)
        """
        return self._id_to_type.loc[ids, TYPE_ATTR_NAME]

    def features(self, type_name, ids):
        """
        Return features for a set of IDs within a given type.

        Args:
            type_name (hashable): the name of the type for all of the IDs
            ids (iterable of IDs): a sequence of IDs of elements of type type_name

        Returns:
            A 2D numpy array, where the rows correspond to the ids
        """
        indices = self._id_to_type.index.get_indexer(ids)
        start = self._type_start_indices[type_name]
        indices -= start

        # FIXME: better error messages
        if (indices < 0).any():
            # ids were < start, e.g. from an earlier type, or unknown (-1)
            raise ValueError("unknown IDs")

        try:
            return self._features[type_name][indices, :]
        except IndexError:
            # some of the indices were too large (from a later type)
            raise ValueError("unknown IDs")

    def feature_sizes(self):
        """
        Returns:
             A dictionary of type_name to an integer representing the size of the features of
             that type.
        """
        return {
            type_name: type_features.shape[1]
            for type_name, type_features in self._features.items()
        }


class NodeData(ElementData):
    pass


class EdgeData(ElementData):
    def __init__(self, features):
        super().__init__(features)

        for key, value in features.items():
            require_dataframe_has_columns(
                f"features[{key!r}].shared", value.shared, {SOURCE, TARGET, WEIGHT}
            )

        self._edges_in = self._id_to_type.groupby(TARGET)
        self._edges_out = self._id_to_type.groupby(SOURCE)
        # return an empty dataframe in the same format as the grouped ones, for vertices that
        # have no edges in a particular direction
        self._no_edges_df = self._id_to_type.iloc[0:0, :]

    def _degree_single(self, previous, col):
        series = self._id_to_type.groupby(col).size()
        if previous is None:
            return series
        return previous.add(series, fill_value=0)

    def degrees(self, ins=True, outs=True):
        """
        Compute the degrees of every non-isolated node.

        Args:
            ins (bool): count the in-degree
            outs (bool): count the out-degree

        Returns:
            The in-, out- or total (summed) degree of all non-isolated nodes.
        """
        series = None
        if ins:
            series = self._degree_single(series, TARGET)
        if outs:
            series = self._degree_single(series, SOURCE)

        if series is None:
            raise ValueError("expected at least one of `ins` and `outs` to be True")

        return series

    def all(self, triple):
        """
        Return all edges as a pandas DataFrame.

        Args:
            triple (bool): include the types as well as the source and target

        Returns:
            A pandas DataFrame containing columns for each source and target and (if triple) the
            type.
        """
        columns = [SOURCE, TARGET]
        if triple:
            columns.append(TYPE_ATTR_NAME)
        return self._id_to_type[columns]

    def ins(self, target_id):
        """
        Return the incoming edges for the node represented by target_id.

        Args:
            target_id: the ID of the node

        Returns:
            A pandas DataFrame containing all the information the edges entering the node.
        """
        try:
            return self._edges_in.get_group(target_id)
        except KeyError:
            # This cannot tell the difference between no edges and a vertex not existing,
            # so it has to assume it's the former
            return self._no_edges_df

    def outs(self, source_id):
        """
        Return the outgoing edges for the node represented by source_id.

        Args:
            source_id: the ID of the node

        Returns:
            A pandas DataFrame containing all the information the edges leaving the node.
        """
        try:
            return self._edges_in.get_group(source_id)
        except KeyError:
            return self._no_edges_df
