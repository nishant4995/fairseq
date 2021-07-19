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

from IPython import embed

@dataclass
class MultiLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    # use_sigmoid: int = field(
    #     default=0,
    #     metadata={"help": "use sigmoid to convert scores into probabilities"},
    # )
    sentence_avg: bool = II("optimization.sentence_avg")


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


def multi_label_smoothed_cross_ent_loss(scores, target, use_sigmoid, n_tgt_per_src, epsilon, reduce=True):
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


@register_criterion(
    "multi_label_smoothed_cross_entropy", dataclass=MultiLabelSmoothedCrossEntropyCriterionConfig
)
class MultiLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        use_sigmoid = 0
        self.use_sigmoid = bool(use_sigmoid)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        """
        We have two options
        1) Either compute a multi-label loss at each step where we need to?
        2) 
        
        Steps:
        1) First, compute output for each label. We need to unpack prev_output_tokens etc for each output
        2) Then in compute loss, we could either add up loss from all labels -> that would again be MLE like stuff
        3) 
        """
        # print("Passing sample to model")
        # embed()
        # input("Press something to continue")
        net_output = model(**sample["net_input"])
        loss, ce_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # print("Done computing loss")
        # embed()
        # input("Press something to continue")
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ce_loss": ce_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def get_scores(self, net_output):
        scores = net_output[0] # TODO: Is there a way to get this from model.?? like we do for logprobs using model.get_normalized_probs?
        # print("Getting scores")
        # embed()
        ans = scores.view(-1, scores.size(-1))
        return ans

    def get_target(self, sample):
        target = sample["target"]
        return target.view(-1)

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
        target = sample["target"].view(-1)
        n_tgt_per_src = sample["target_per_src"]
        # print("Now computing loss")
        # embed()
        loss, base_loss = multi_label_smoothed_cross_ent_loss(
            scores=scores,
            target=target,
            use_sigmoid=self.use_sigmoid,
            n_tgt_per_src=n_tgt_per_src,
            epsilon=self.eps,
            reduce=reduce,
        )
        return loss, base_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        # raise NotImplementedError
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
        # metrics.log_derived(
        #     "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        # )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
