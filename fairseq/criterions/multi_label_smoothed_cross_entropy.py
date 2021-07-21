# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import warnings
import numpy as np
from collections import defaultdict
from IPython import embed

@dataclass
class MultiLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    # label_smoothing: float = field(
    #     default=0.0,
    #     metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    # )
    multi_label_loss_type: str = field(
        default="cross_ent",
        metadata={"help": "loss function to compute token level multi-label loss"},
    )


def old_multi_label_smoothed_cross_ent_loss(scores, target, n_tgt_per_src, epsilon, reduce=True):
    try:
        if target.dim() == scores.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -scores.gather(dim=-1, index=target)
        smooth_loss = -scores.sum(dim=-1, keepdim=True)

        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / (scores.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss
    except Exception as e:
        embed()
        raise e


def multi_label_smoothed_cross_ent_loss_v0(scores, target, use_sigmoid, n_tgt_per_src, epsilon, reduce=True):
    try:
        if target.dim() == scores.dim() - 1:
            target = target.unsqueeze(-1)

        if use_sigmoid:
            scores = torch.sigmoid(scores)

        pos_scores = scores.gather(dim=-1, index=target)
        scores_sum = scores.sum(dim=-1, keepdim=True)

        # print("Checkpoint 1")
        # embed()
        pos_scores = pos_scores.squeeze(-1)
        scores_sum = scores_sum.squeeze(-1)

        ce_loss = -1*(2*pos_scores - scores_sum)
        # print("Checkpoint 2")
        # embed()
        if reduce:
            ce_loss = ce_loss.mean()
            scores_sum = scores_sum.mean()

        # eps_i = epsilon / (scores.size(-1) - 1)
        # loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
        return ce_loss, ce_loss
    except Exception as e:
        embed()
        raise e


def multi_label_smoothed_cross_ent_loss(scores, target, n_tgt_per_src, **loss_opts):
    """
    sco
    """


    try:
        # TODO: Compute mean of loss instead of sum
        loss_per_dtpoint = []
        total_n_labels = 0
        for n_labels in n_tgt_per_src:
            # print(f"\n\nComputing loss for datapoint for {n_labels} labels")
            curr_scores = scores[total_n_labels: total_n_labels+n_labels]
            curr_target = target[total_n_labels: total_n_labels+n_labels]

            loss, _ = multi_label_smoothed_cross_ent_loss_per_data_point(scores=curr_scores,
                                                                         target=curr_target,
                                                                         **loss_opts)
            loss_per_dtpoint += [loss]
            total_n_labels = total_n_labels + n_labels
            # print(f"--------------------------------------------------------------------------------------------------------\n\n")

        batch_loss = torch.mean(torch.vstack(loss_per_dtpoint))
        return batch_loss, batch_loss
    except Exception as e:
        embed()
        raise e


def multi_label_smoothed_cross_ent_loss_per_data_point(scores, target, **loss_opts):
    try:
        # target : shape : (number of labels ) x max_seq_len
        # scores : shape : (number of labels ) x max_seq_len x vocab_size
        # max_seq_len corresponds to number of time steps for generation

        # print(f"Computing loss for datapoint for scores with {scores.shape} and target with {target.shape}")
        # print(f"Target : {target}")

        # TODO: Fix issue of computing loss for pad tokens which are present in ground-truth sequence - Semi fixed: If we have eos or pad token as token at step t, then we don't compute loss but still iteratate over a for-loop. There can be a faster way of doing this.

        assert target.shape[1] > 0
        loss_per_time_step = []
        for time_step in range(target.shape[-1]):
            # print("Time step", time_step)
            target_upto_t = target[:, :time_step]  # Get target sequence upto current time step
            # print("target_upto_t\n", target_upto_t)

            # Step 1: Find unique sequences in target_upto_t
            # inv_indices : shape target_upto_t.shape[0].
            # This maps final index of target_upto_t's tensors in uniq_target_upto_t tensor.
            # inv_indices[0] gives location of target_upto_t[0] in uniq_target_upto_t tensor
            if time_step > 0:
                uniq_target_upto_t, inv_indices = torch.unique(target_upto_t, dim=0, return_inverse=True)
                inv_indices                     = inv_indices.cpu().numpy()
            else:
                uniq_target_upto_t  = target_upto_t
                inv_indices         = np.zeros(target_upto_t.shape[0], np.int64)

            # Step 2: Gather remaining portion of target sequences following each unique tensor in uniq_target_upto_t
            # uniq_idx_to_orig_idxs: Maps idx to a list of indices where uniq_target_upto_t[idx] occurs in target_upto_t
            uniq_idx_to_orig_idxs = defaultdict(list)
            for orig_idx, uniq_idx in enumerate(inv_indices):
                uniq_idx_to_orig_idxs[uniq_idx].append(orig_idx)

            # TODO: Optimize
            # if len(uniq_idx_to_orig_idxs) == target.shape[0]:
            #     print("Now we have all unique labels from this point onward. These might be some optimization that we can do here")
            #     pass

            # Break loop and avoid iterating over uniq prefs if last token at current time_step is pad token for all targets
            # TODO: Use pad_idx instead of hardcoded value of 1 for matching pad tokens
            if time_step > 0 and all(torch.logical_or(target_upto_t[:,-1] == 1, target_upto_t[:,-1] == 2)):
                # print("Breaking because all sequences have either reached a pad token or eos token")
                break

            # print(f"Inv indices - {inv_indices}")
            # print(f"uniq_idx_to_orig_idxs -> {uniq_idx_to_orig_idxs}")
            # Iterate over each uniq prefix tensor in target_upto_t up to this time_step and compute multi-label style loss for tokens following each prefix.
            loss_per_uniq_pref = []
            for uniq_idx, curr_orig_idxs in uniq_idx_to_orig_idxs.items():

                # TODO: Use eos_idx and pad_idx here
                if time_step > 0 and (uniq_target_upto_t[uniq_idx][-1] == 2 or uniq_target_upto_t[uniq_idx][-1] == 1):
                    # print(f"Avoiding computing loss for pad tokens at idx {uniq_idx} with tensor {uniq_target_upto_t[uniq_idx]} originally present at {curr_orig_idxs}")
                    # print("Next tokens for which loss would be computed", target[curr_orig_idxs, time_step])  # Shape : (num occurrences of this prefix))
                    continue

                # print(f"Computing score for uniq seq at idx {uniq_idx} with tensor {uniq_target_upto_t[uniq_idx]} originally present at {curr_orig_idxs} indices")
                curr_target = target[curr_orig_idxs, time_step]  # Shape : (num occurrences of this prefix)
                curr_scores = scores[curr_orig_idxs, time_step, :] # Shape : (num occurrences of this prefix) x vocab_size

                assert curr_target.shape == (len(curr_orig_idxs),)
                assert curr_scores.shape == (len(curr_orig_idxs), scores.shape[-1])
                # All rows in curr_scores should have the same score values as prefix used to generated the score is the same.
                # But due to drop out at training time, these scores differ even for the same input
                # curr_scores = torch.unique(curr_scores, dim=0) Do not compute unique here because of dropout issue
                #
                # if curr_scores.shape[0] != 1: # Raise a warning
                #     # print(f"\n\n curr_scores contain more than one unique array , curr_scores.shape = {curr_scores.shape} \n\n")
                #     warnings.warn(f"\n\n curr_scores contain more than one unique array , curr_scores.shape = {curr_scores.shape} {curr_scores}\n\n")
                #     print("Time step", time_step)
                #     print("target_upto_t", target_upto_t)
                #     print(f"Inv indices - {inv_indices}")
                #     print(f"uniq_idx_to_orig_idxs -> {uniq_idx_to_orig_idxs}")
                #     print(f"Computing score for uniq seq at idx {uniq_idx} with tensor {uniq_target_upto_t[uniq_idx]} originally present at {curr_orig_idxs}")
                #     embed()
                #
                #     raise Exception(f"\n\n curr_scores contain more than one unique array , curr_scores.shape = {curr_scores.shape} {curr_scores}\n\n")
                # curr_scores = curr_scores[0].view(-1) # Shape : vocab_size

                loss = ml_ce_helper(scores=curr_scores, target=curr_target, **loss_opts)
                loss_per_uniq_pref += [loss]

            if len(loss_per_uniq_pref) > 0:
                loss_per_time_step += [torch.mean(torch.vstack(loss_per_uniq_pref))]

        loss_per_datapoint = torch.mean(torch.vstack(loss_per_time_step))
        return loss_per_datapoint, loss_per_datapoint
    except Exception as e:
        embed()
        raise e


def ml_ce_helper(scores, target, **loss_opts):
    try:
        # TODO: See if this can be batched
        # TODO: Implement options for squared hinge loss, hinge loss, cross entropy loss and ce loss without sigmoid, ce with logsumexp on neg scores
        # TODO: Implement some label smoothening

        # scores shape : (num_pos_tokens, vocab_size)
        # target shape : (num_pos_tokens)
        assert target.dim() == scores.dim() - 1 == 1

        pos_one_hot = torch.nn.functional.one_hot(input=target, num_classes=scores.shape[-1]).sum(dim=0, dtype=torch.bool).type_as(target)
        neg_one_hot = 1 - pos_one_hot # (1, vocab_size)

        # target = target.unsqueeze(-1)  # shape : (num_pos_tokens, 1). add another dim so that we can use gather operation on scores with target
        # pos_scores = scores.gather(dim=-1, index=target)  # Pos indices are treated as multi-set
        pos_scores  = torch.multiply(scores, pos_one_hot) # Get scores of tokens that are part of pos_tokens
        neg_scores  = torch.multiply(scores, neg_one_hot) # Get scores of tokens that are not part of pos_tokens

        # TODO: Design decision: Give equal weight to pos and negative examples? Or give equal weight to each example

        multi_label_loss_type = loss_opts.get("multi_label_loss_type", "cross_ent")
        if multi_label_loss_type == "cross_ent":
            loss_func = torch.nn.BCEWithLogitsLoss()
            pos_loss    = loss_func.forward(input=pos_scores, target=torch.ones(pos_scores.shape).to(pos_scores))
            neg_loss    = loss_func.forward(input=neg_scores, target=-torch.ones(neg_scores.shape).to(neg_scores))
            loss        = (pos_loss + neg_loss)
        elif multi_label_loss_type == "scores":
            loss        = -pos_scores.mean() + neg_scores.mean()
        elif multi_label_loss_type == "hinge":
            raise NotImplementedError
        elif multi_label_loss_type == "sq_hinge":
            raise NotImplementedError
        else:
            raise NotImplementedError

        # print("Checkpoint 1")
        # print("scores.shape", scores.shape, scores)
        # print("target.shape", target.shape, target)
        # print("pos_scores", pos_scores.shape, pos_scores)
        # print("neg_scores", neg_scores.shape, neg_scores)
        # print("neg_one_hot", neg_one_hot)
        # print("loss", loss, pos_scores.sum(), neg_scores.sum())
        # print("....")
        # if target.shape[0] > 1:
        #     embed()
        return loss
    except Exception as e:
        embed()
        raise e



@register_criterion(
    "multi_label_smoothed_cross_entropy", dataclass=MultiLabelSmoothedCrossEntropyCriterionConfig
)
class MultiLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        multi_label_loss_type
    ):
        super().__init__(task)
        self.multi_label_loss_type = multi_label_loss_type

    @classmethod
    def add_args(cls, parser):
        super(MultiLabelSmoothedCrossEntropyCriterion, cls).add_args(parser)
        parser.add_argument(
            "--multi_label_loss_type",
            default="cross_ent",
            type=str,
            choices=["cross_ent", "scores", "hinge", "sq_hinge"],
            help="loss function to compute token level multi-label loss",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        try:
            # print("Passing sample to model")
            # embed()
            # input("Press something to continue")
            net_output = model(**sample["net_input"])
            loss, ce_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            # print("Done computing loss")
            # embed()
            # input("Press something to continue")
            # sample_size = (
            #     sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            # )
            sample_size = 1 # TODO: This is used to scale gradients in L796 in fairreq/trainer.py
            # print("Loss.data", loss.data)
            logging_output = {
                "loss": loss.data.cpu().numpy(),
                "ce_loss": ce_loss.data.cpu().numpy(),
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            return loss, sample_size, logging_output
        except Exception as e:
            embed()
            input("")
            raise e

    def get_scores(self, net_output):
        scores = net_output[0] # TODO: Is there a way to get this from model.?? like we do for logprobs using model.get_normalized_probs?
        return scores

    def get_target(self, sample):
        target = sample["target"]
        return target

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        # if self.ignore_prefix_size > 0:
        #     if getattr(lprobs, "batch_first", False):
        #         lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
        #         target = target[:, self.ignore_prefix_size :].contiguous()
        #     else:
        #         lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
        #         target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        # lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        scores = self.get_scores(net_output=net_output)
        target = sample["target"]
        n_tgt_per_src = sample["target_per_src"]

        assert reduce
        loss, base_loss = multi_label_smoothed_cross_ent_loss(
            scores=scores,
            target=target,
            n_tgt_per_src=n_tgt_per_src,
            multi_label_loss_type=self.multi_label_loss_type
        )
        return loss, base_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ce_loss", ce_loss_sum / ntokens / math.log(2), ntokens, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
