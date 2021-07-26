# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import time
import torch.nn.functional as F
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
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    multi_label_loss_type: str = field(
        default="cross_ent",
        metadata={"help": "loss function to compute token level multi-label loss"},
    )
    ml_loss_timestep: int = field(
        default=1,
        metadata={"help": "Use multi-label loss for time step < ml_loss_timestep and nll loss thereafter"},
    )


def label_smoothed_cross_ent_loss(lprobs, target, epsilon):
    try:
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

        # Reduce loss to a single number
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

        eps_i = epsilon / (lprobs.size(-1) - 1)
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


def multi_label_smoothed_cross_ent_loss_per_data_point_t_0(scores, target, **loss_opts):
    # This is optimized to compute loss for first tokens of target sequences
    try:
        # target : shape : (number of labels ) x 1
        # scores : shape : (number of labels ) x 1 x vocab_size
        assert target.shape[1] == 1 and target.dim() == 2
        assert scores.shape[1] == 1 and scores.dim() == 3

        curr_target = target.squeeze(-1)    # Shape : (num occurrences of this prefix)
        curr_scores = scores.squeeze(1)     # Shape : (num occurrences of this prefix) x vocab_size

        # assert curr_target.shape == (target.shape[0],)
        # assert curr_scores.shape == (scores.shape[0], scores.shape[-1])

        loss = ml_ce_helper(scores=curr_scores, target=curr_target, **loss_opts)

        return loss, loss
    except Exception as e:
        embed()
        raise e


def ml_ce_helper(scores, target, **loss_opts):
    try:
        # TODO: See if this can be batched
        # TODO: Implement some label smoothening

        # scores shape : (num_pos_tokens, vocab_size)
        # target shape : (num_pos_tokens)
        assert target.dim() == scores.dim() - 1 == 1

        pos_one_hot = torch.nn.functional.one_hot(input=target, num_classes=scores.shape[-1]).sum(dim=0, dtype=torch.bool).type_as(target)
        neg_one_hot = 1 - pos_one_hot # (1, vocab_size)

        # Compute scores for target tokens while accounting for count of each pos token in target token
        target = target.unsqueeze(-1)  # shape : (num_pos_tokens, 1). add another dim so that we can use gather operation on scores with target
        pos_scores = scores.gather(dim=-1, index=target)  # Pos indices are treated as multi-set

        # pos_scores  = torch.multiply(scores, pos_one_hot) # Get scores of tokens that are part of pos_tokens. This ignores count of each pos token in target tokens
        neg_scores  = torch.multiply(scores, neg_one_hot) # Get scores of tokens that are not part of pos_tokens

        # TODO: Design decision: Give equal weight to pos and negative examples? Or give equal weight to each example

        multi_label_loss_type = loss_opts.get("multi_label_loss_type", "cross_ent")
        if multi_label_loss_type == "cross_ent":
            loss_func = torch.nn.BCEWithLogitsLoss()
            pos_loss    = loss_func.forward(input=pos_scores, target=torch.ones(pos_scores.shape).to(pos_scores))
            neg_loss    = loss_func.forward(input=neg_scores, target=torch.zeros(neg_scores.shape).to(neg_scores))
            loss        = (pos_loss + neg_loss)
        elif multi_label_loss_type == "scores":
            pos_loss    = -pos_scores.mean()
            neg_loss    = neg_scores.mean()
            loss        = pos_loss + neg_loss
        elif multi_label_loss_type == "hinge":
            pos_scores[pos_scores > 1] = 0
            neg_scores[neg_scores < -1] = 0
            pos_loss    = -pos_scores.mean()
            neg_loss    = neg_scores.mean()
            loss        = (pos_loss + neg_loss)
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


def hybrid_multi_label_loss_func(scores, target, n_tgt_per_src, **loss_opts):

    # scores.shape : num_src x max_timestep x vocab_size
    # target.shape : num_src x max_timestep
    try:
        # For time_step < ml_loss_timestep, calculate multilabel loss, and then calculate nll loss

        max_t = target.shape[1]
        ml_loss_timestep = loss_opts.get("ml_loss_timestep", 1)
        ml_scores   = scores[:, :ml_loss_timestep, :]
        ml_target   = target[:, :ml_loss_timestep]

        if ml_loss_timestep == 0:
            ml_loss = 0.
        elif ml_loss_timestep == 1:
            ml_loss, _ = multi_label_smoothed_cross_ent_loss_per_data_point_t_0(scores=ml_scores, target=ml_target,
                                                                                n_tgt_per_src=n_tgt_per_src,
                                                                                **loss_opts)
            # print("Multi-label loss = {} computed in time {:.4f}".format(ml_loss, time.time() - t1))
        else:
            ml_loss, _ = multi_label_smoothed_cross_ent_loss_per_data_point(scores=ml_scores, target=ml_target,
                                                                            n_tgt_per_src=n_tgt_per_src,
                                                                            **loss_opts)
            # print("Multi-label loss = {} computed in time {:.4f}".format(ml_loss, time.time() - t1))


        nll_lprobs = F.log_softmax(scores[:, ml_loss_timestep:, :], dim=-1)
        nll_target = target[:, ml_loss_timestep:]

        epsilon = loss_opts.get("epsilon", 0.0)
        smooth_loss, nll_loss = label_smoothed_cross_ent_loss(lprobs=nll_lprobs, target=nll_target,
                                                              epsilon=epsilon)

        final_smooth_loss = (ml_loss * ml_loss_timestep + smooth_loss * (max_t - ml_loss_timestep))/max_t
        final_loss = (ml_loss * ml_loss_timestep + nll_loss * (max_t - ml_loss_timestep))/max_t

        # print(f"Computed loss {ml_loss} and {smooth_loss} ({nll_loss}). Final loss is {final_smooth_loss} ({final_loss})")
        # embed()
        return final_smooth_loss, final_loss, ml_loss, nll_loss
    except Exception as e:
        embed()
        raise e


######################### Loss func w/ padding of target tensors ##############

def label_smoothed_cross_ent_loss_w_multi_targets_v1(scores, target, epsilon, parallel_targets):
    # target shape : num_tgt x max_seq_len (x 1) : Last dim is added after target = target.unsqueeze(-1)
    try:
        if target.dim() == scores.dim() - 1:
            target = target.unsqueeze(-1)

        pos_loss    = -scores.gather(dim=-1, index=target)

        assert scores.dim() == 3
        max_tgt_seq_len = scores.shape[1]
        vocab_size = scores.shape[2]

        prll_tgt_pos_one_hot_list = []
        for curr_prll_tgts in parallel_targets: # Iterate over parallel tokens list for each target

            curr_tgt_pos_one_hot_list = []
            for curr_prll_tokens in curr_prll_tgts: # For each time_step, look at parallel tokens for current target
                temp_pos_one_hot = torch.nn.functional.one_hot(input=torch.LongTensor(curr_prll_tokens),
                                                               num_classes=scores.shape[-1])
                temp_pos_one_hot = temp_pos_one_hot.sum(dim=0) # Shape: vocab_size
                temp_pos_one_hot[temp_pos_one_hot > 1] = 1
                curr_tgt_pos_one_hot_list += [temp_pos_one_hot]

            # Shape: tgt_seq_len x vocab_size
            curr_tgt_pos_one_hot = torch.vstack(curr_tgt_pos_one_hot_list)

            tgt_seq_len = len(curr_prll_tgts)

            # Shape: max_tgt_seq_len x vocab_size
            padded_curr_tgt_pos_one_hot = torch.zeros(max_tgt_seq_len, vocab_size)
            padded_curr_tgt_pos_one_hot[:tgt_seq_len] = curr_tgt_pos_one_hot
            prll_tgt_pos_one_hot_list += [padded_curr_tgt_pos_one_hot]

        # Shape: num_tgt x tgt_seq_len x vocab_size
        prll_tgt_pos_one_hot = torch.stack(prll_tgt_pos_one_hot_list, dim=0)

        # print("\npos_one_hot.shape", prll_tgt_pos_one_hot.shape)
        # print("target.shape", target.shape)
        # print("scores.shape", scores.shape)

        pos_one_hot = prll_tgt_pos_one_hot.to(scores)

        neg_one_hot = 1 - pos_one_hot  # (1, vocab_size)

        neg_scores  = torch.multiply(scores, neg_one_hot)
        neg_loss    = neg_scores.logsumexp(dim=-1, keepdim=True)

        nll_loss    = pos_loss + neg_loss

        lprobs      = F.log_softmax(scores, dim=-1)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

        # Reduce loss to a single number
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

        eps_i = epsilon / (lprobs.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss
    except Exception as e:
        embed()
        raise e


def label_smoothed_cross_ent_loss_w_multi_targets(scores, target, epsilon, parallel_targets):
    # target shape : num_tgt x max_seq_len (x 1) : Last dim is added after target = target.unsqueeze(-1)
    try:
        if target.dim() == scores.dim() - 1:
            target = target.unsqueeze(-1)

        pos_loss    = -scores.gather(dim=-1, index=target)


        prll_tgt_pos_one_hot    = torch.nn.functional.one_hot(input=torch.LongTensor(parallel_targets).to(scores.device),
                                                              num_classes=scores.shape[-1]).sum(dim=-2)
        # print(prll_tgt_pos_one_hot.shape)
        # prll_tgt_pos_one_hot    = prll_tgt_pos_one_hot.sum(dim=-2)
        # print(prll_tgt_pos_one_hot.shape)
        prll_tgt_pos_one_hot[prll_tgt_pos_one_hot > 1] = 1
        # print()

        assert prll_tgt_pos_one_hot.shape == scores.shape

        pos_one_hot = prll_tgt_pos_one_hot.to(scores)

        neg_one_hot = 1 - pos_one_hot  # (1, vocab_size)

        neg_scores  = torch.multiply(scores, neg_one_hot)
        neg_loss    = neg_scores.logsumexp(dim=-1, keepdim=True)

        nll_loss    = pos_loss + neg_loss

        lprobs      = F.log_softmax(scores, dim=-1)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

        # Reduce loss to a single number
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

        eps_i = epsilon / (lprobs.size(-1) - 1)
        loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss
    except Exception as e:
        embed()
        raise e


def hybrid_multi_label_loss_func_w_multi_targets(scores, target, parallel_targets, ml_loss_timestep, **loss_opts):

    # scores.shape : num_src x max_timestep x vocab_size
    # target.shape : num_src x max_timestep
    try:
        # For time_step < ml_loss_timestep, calculate multilabel loss, and then calculate nll loss

        max_t = target.shape[1]
        # ml_loss_timestep = loss_opts.get("ml_loss_timestep", 1)


        if ml_loss_timestep == 0:
            ml_loss = 0.
        else:
            # ml_scores = scores[:, :ml_loss_timestep, :]
            # ml_target = target[:, :ml_loss_timestep]
            ml_loss, _ = label_smoothed_cross_ent_loss_w_multi_targets(scores=scores[:, :ml_loss_timestep, :],
                                                                       target=target[:, :ml_loss_timestep],
                                                                       parallel_targets=parallel_targets,
                                                                       **loss_opts)


        nll_lprobs = F.log_softmax(scores[:, ml_loss_timestep:, :], dim=-1)
        nll_target = target[:, ml_loss_timestep:]

        epsilon = loss_opts.get("epsilon", 0.0)
        smooth_loss, nll_loss = label_smoothed_cross_ent_loss(lprobs=nll_lprobs, target=nll_target,
                                                              epsilon=epsilon)

        final_smooth_loss = (ml_loss * ml_loss_timestep + smooth_loss * (max_t - ml_loss_timestep))/max_t
        final_loss = (ml_loss * ml_loss_timestep + nll_loss * (max_t - ml_loss_timestep))/max_t

        # print(f"Computed loss {ml_loss} and {smooth_loss} ({nll_loss}). Final loss is {final_smooth_loss} ({final_loss})")
        # embed()
        return final_smooth_loss, final_loss, ml_loss, nll_loss
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
        multi_label_loss_type,
        label_smoothing,
        ml_loss_timestep

    ):
        super().__init__(task)
        self.multi_label_loss_type  = multi_label_loss_type
        self.epsilon                = label_smoothing
        self.ml_loss_timestep       = ml_loss_timestep

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
        parser.add_argument(
            "--label_smoothing",
            default=0.0,
            type=float,
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument(
            "--ml_loss_timestep",
            default=1,
            type=int,
            help="Use multi-label loss for time step < ml_loss_timestep and nll loss thereafter",
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
            loss, loss_wo_smooth, ml_loss, nll_loss  = self.compute_loss(model, net_output, sample, reduce=reduce)
            # print("Done computing loss")
            # embed()
            # input("Press something to continue")
            sample_size = 1 # TODO: This is used to scale gradients in Line:796 in fairreq/trainer.py
            logging_output = {
                "loss": loss.data,
                "loss_wo_smooth": loss_wo_smooth.data,
                "ml_loss": ml_loss if isinstance(ml_loss, float) else ml_loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            return loss, sample_size, logging_output
        except Exception as e:
            print(sample["net_input"])
            embed()
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
        # parallel_tgt_tokens_list_of_lists_wo_pad = sample["parallel_target_tokens_wo_pad"]
        # parallel_tgt_tokens_list_of_lists = sample["parallel_target_tokens"]
        parallel_tgt_tokens_list_of_lists_2 = sample["parallel_target_tokens_2"]
        assert reduce
        # start1 = time.time()
        # loss, loss_wo_smooth, ml_loss, nll_loss = hybrid_multi_label_loss_func(
        #     scores=scores,
        #     target=target,
        #     n_tgt_per_src=n_tgt_per_src,
        #     multi_label_loss_type=self.multi_label_loss_type,
        #     epsilon=self.epsilon,
        #     ml_loss_timestep=self.ml_loss_timestep
        # )
        # end1 = time.time()

        # start2 = time.time()
        # loss2, loss_wo_smooth2 = label_smoothed_cross_ent_loss_w_multi_targets_v1(scores=scores,
        #                                                                        target=target,
        #                                                                        parallel_targets=parallel_tgt_tokens_list_of_lists_wo_pad,
        #                                                                        epsilon=self.epsilon)
        # end2 = time.time()

        # start3 = time.time()
        # loss3, loss_wo_smooth3 = label_smoothed_cross_ent_loss_w_multi_targets(scores=scores,
        #                                                                        target=target,
        #                                                                        parallel_targets=parallel_tgt_tokens_list_of_lists,
        #                                                                        epsilon=self.epsilon)
        # end3 = time.time()

        # start4 = time.time()
        # loss4, base_loss4 = multi_label_smoothed_cross_ent_loss(
        #     scores=scores,
        #     target=target,
        #     n_tgt_per_src=n_tgt_per_src,
        #     multi_label_loss_type=self.multi_label_loss_type
        # )
        # end4 = time.time()


        # start5 = time.time()
        loss, loss_wo_smooth, ml_loss, nll_loss = hybrid_multi_label_loss_func_w_multi_targets(scores=scores,
                                                                              target=target,
                                                                              parallel_targets=parallel_tgt_tokens_list_of_lists_2,
                                                                              ml_loss_timestep=2,
                                                                              epsilon=self.epsilon)
        # end5 = time.time()


        # start6 = time.time()
        # nll_lprobs = F.log_softmax(scores, dim=-1)
        # loss6, base_loss6 = label_smoothed_cross_ent_loss(
        #     lprobs=nll_lprobs,
        #     target=target,
        #     epsilon=0.0
        # )
        # end6 = time.time()
        #
        # start2 = time.time()
        # loss2, base_loss2 = hybrid_multi_label_loss_func(
        #     scores=scores,
        #     target=target,
        #     n_tgt_per_src=n_tgt_per_src,
        #     multi_label_loss_type=self.multi_label_loss_type
        # )
        # end2 = time.time()

        #
        # print("Hybrid Loss                   {} calc time :{:.4f}".format(loss, end1 - start1))
        # print("Multi-Label Loss (new)        {} calc time :{:.4f}".format(loss2,  end2 - start2))
        # # print("Multi-Label Loss (new pad)    {} calc time :{:.4f}".format(loss3,  end3 - start3))
        # print("Multi-Label Loss              {} calc time :{:.4f}".format(loss4,  end4 - start4))
        # print("Hybrid Multi-Label(new)       {} calc time :{:.4f}".format(loss5,  end5 - start5))
        # print("NLL Loss                      {} calc time :{:.4f}".format(loss6, end6 - start6))
        # print()


        return loss, loss_wo_smooth, ml_loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        for loss_type in ["loss", "loss_wo_smooth", "ml_loss", "nll_loss"]:
            temp_sum = sum(log.get(loss_type, 0) for log in logging_outputs)
            metrics.log_scalar(
                loss_type, temp_sum / sample_size / math.log(2), sample_size, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
