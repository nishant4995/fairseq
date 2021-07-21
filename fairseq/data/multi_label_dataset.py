# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    # raise NotImplementedError
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    try:
        id = torch.LongTensor([s["id"] for s in samples])
        src_tokens = merge(
            key="source",
            left_pad=left_pad_source,
            pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
        )
        # sort by descending source length
        src_lengths = torch.LongTensor(
            [s["source"].ne(pad_idx).long().sum() for s in samples]
        )
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        prev_output_tokens = None
        target_tensor = None
        target_per_src = None
        if samples[0].get("target", None) is not None:
            unsorted_target = [s["target"] for s in samples]

            # unsorted_prev_output_tokens = None
            # if input_feeding:
            #     unsorted_prev_output_tokens = []
            #     prev_output_tokens = [None] * len(samples)
            #     for s in samples:
            #         curr_prev_outs = []
            #         for label in s["target"]:
            #             temp = label.new(label.shape).fill_(pad_idx)
            #             temp[0] = label[-1]
            #             temp[1:] = label[:-1]
            #             curr_prev_outs += [temp]
            #
            #         unsorted_prev_output_tokens += [curr_prev_outs]

            target_list_of_lists    = [[] for _ in range(len(samples))]
            target_per_src          = [] # List of number of labels per src input
            # Sort target according to sort_order TODO: Verify if this is correct way to sort
            for new_idx, orig_idx in enumerate(sort_order):
                target_list_of_lists[new_idx] = unsorted_target[orig_idx]
                target_per_src.append(len(target_list_of_lists[new_idx]))
                # if input_feeding:
                #     prev_output_tokens[i] = unsorted_prev_output_tokens[i]

            target_list = [temp_tgt for curr_target in target_list_of_lists for temp_tgt in curr_target] # Convert list of list to a single list
            target_tensor = data_utils.collate_tokens(
                    values=target_list, #[List of 1-D tensors],
                    pad_idx=pad_idx,
                    eos_idx=eos_idx,
                    left_pad=left_pad_target,
                    move_eos_to_beginning=False,
                    pad_to_length=pad_to_length["target"] if pad_to_length is not None else None,
                    pad_to_multiple=pad_to_multiple,
                    pad_to_bsz=None)

            new_src_tokens = []
            for i, n_tgt in enumerate(target_per_src):
                new_src_tokens += [torch.clone(src_tokens[i]) for _ in range(n_tgt)] # Create n_tgt copies of src_tokens[i]
            src_tokens = torch.vstack(new_src_tokens)

            # TODO: Fix this. Do we need ntokens and tgt_lengths at training time somewhere? If yes, then fix calc below
            tgt_lengths = torch.LongTensor(
                [temp_tgt.ne(pad_idx).long().sum() for temp_tgt in target_list]
            )
            ntokens = tgt_lengths.sum().item()

            if input_feeding:
                prev_output_tokens = data_utils.collate_tokens(
                                    values=target_list,  # [List of 1-D tensors],
                                    pad_idx=pad_idx,
                                    eos_idx=eos_idx,
                                    left_pad=left_pad_target,
                                    move_eos_to_beginning=True, # Move eos token to beginning to create prev output tokens for given target
                                    pad_to_length=pad_to_length["target"] if pad_to_length is not None else None,
                                    pad_to_multiple=pad_to_multiple,
                                    pad_to_bsz=None)



            # target = merge(
            #     key="target",
            #     left_pad=left_pad_target,
            #     pad_to_length=pad_to_length["target"]
            #     if pad_to_length is not None
            #     else None,
            # )
            # target = target.index_select(0, sort_order)
            # tgt_lengths = torch.LongTensor(
            #     [s["target"].ne(pad_idx).long().sum() for s in samples]
            # ).index_select(0, sort_order)
            # ntokens = tgt_lengths.sum().item()

            # if samples[0].get("prev_output_tokens", None) is not None:
            #     prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
            # elif input_feeding:
            #     # we create a shifted version of targets for feeding the
            #     # previous output token(s) into the next decoder step
            #     prev_output_tokens = merge(
            #         "target",
            #         left_pad=left_pad_target,
            #         move_eos_to_beginning=True,
            #         pad_to_length=pad_to_length["target"]
            #         if pad_to_length is not None
            #         else None,
            #     )
        else:
            ntokens = src_lengths.sum().item()

    except Exception as e:
        from IPython import embed
        embed()
        raise e

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths,},
        "target": target_tensor,
        "target_per_src": target_per_src,
    }
    if prev_output_tokens is not None:
        # batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(0, sort_order)
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch


class MultiLabelDataset(FairseqDataset):
    """
    Dataset wrapper for MultiLabel dataset storing input and outputs separately

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
    """
    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        pad_to_multiple=1,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(tgt), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.eos = src_dict.eos()

        self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        # TODO: Make sure target contain all labels for the corresponding input
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }

        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # TODO: Verify that this is correct
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)

    def get_batch_shapes(self):
        """
        Return a list of valid batch shapes, for example::

            [(8, 512), (16, 256), (32, 128)]

        The first dimension of each tuple is the batch size and can be ``None``
        to automatically infer the max batch size based on ``--max-tokens``.
        The second dimension of each tuple is the max supported length as given
        by :func:`fairseq.data.FairseqDataset.num_tokens`.

        This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`
        to restrict batch shapes. This is useful on TPUs to avoid too many
        dynamic shapes (and recompilations).
        """
        return None


    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        # TODO: Verify that this is correct
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes, self.tgt_sizes, indices, max_sizes,
        )

