# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    MultiLabelDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task




logger = logging.getLogger(__name__)


def load_multilabel_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    dataset_impl,
    left_pad_source,
    left_pad_target,
    shuffle=True,
    pad_to_multiple=1,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    # infer langcode
    if split_exists(split, src, tgt, src, data_path):
        prefix = os.path.join(data_path, "{}.{}-{}.".format(split, src, tgt))
    elif split_exists(split, tgt, src, src, data_path):
        prefix = os.path.join(data_path, "{}.{}-{}.".format(split, tgt, src))
    else:
        raise FileNotFoundError(
            "Dataset not found: {} ({})".format(split, data_path)
        )

    src_dataset = data_utils.load_indexed_dataset(
        prefix + src, src_dict, dataset_impl
    )
    # TODO: Maybe add support for truncating dataset

    tgt_dataset = data_utils.load_indexed_dataset(
        prefix + tgt, tgt_dict, dataset_impl
    )

    # Make sure that the dataset is in raw format when  reading here
    # Now tgt dataset will have .lines variable that will store raw lines
    # Iterate over it and use dictionary to convert them
    # Then make sure it is used correctly for the task
    # input("\n\n\nLoading dataset\n\n\n")
    # from IPython import embed
    # embed()

    print("Encoding multi-label dataset target")
    try:
        from tqdm import tqdm
        new_tokens_list = []
        new_sizes = []
        for curr_label_list in tqdm(tgt_dataset.lines):
            curr_tokens_list = []
            curr_size = 0
            for curr_label in json.loads(curr_label_list):
                tokens = tgt_dict.encode_line(
                    line=curr_label,
                    add_if_not_exist=False,
                    append_eos=True,
                    reverse_order=False,
                ).long()
                curr_tokens_list += [tokens]
                curr_size += len(tokens)
            new_tokens_list += [curr_tokens_list]
            new_sizes += [curr_size]

        tgt_dataset.tokens_list = new_tokens_list
        tgt_dataset.sizes = np.array(new_sizes)
    except Exception as e:
        from IPython import embed
        embed()
        raise e

    logger.info("{} {} {}-{} {} examples".format(data_path, split, src, tgt, len(src_dataset)))

    return MultiLabelDataset(
        src=src_dataset,
        src_sizes=src_dataset.sizes,
        src_dict=src_dict,
        tgt=tgt_dataset,
        tgt_sizes=tgt_dataset.sizes,
        tgt_dict=tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@dataclass
class MultiLabelConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")


@register_task("multilabel", dataclass=MultiLabelConfig)
class MultiLabelTask(FairseqTask):
    """
    Given a input text generates text of related labels for multi-label classification

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::
        # FIXME: Verify this
        # The translation task is compatible with :mod:`fairseq-train`,
        # :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: MultiLabelConfig

    def __init__(self, cfg: MultiLabelConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: MultiLabelConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (MultiLabelConfig)
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_multilabel_dataset(
            data_path=data_path,
            split=split,
            src=src,
            src_dict=self.src_dict,
            tgt=tgt,
            tgt_dict=self.tgt_dict,
            dataset_impl=self.cfg.dataset_impl,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        # FIXME:
        assert constraints is None
        return MultiLabelDataset(
            src=src_tokens,
            src_sizes=src_lengths,
            src_dict=self.source_dictionary,
            tgt_dict=self.target_dictionary,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict


