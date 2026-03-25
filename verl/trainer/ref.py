# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
You need to manually specify the score normalization function in fsdp_workers.py if using rm
You need to manually write the multi-turn prompt if using multi-turn algorithm in _generate_multi_turn_batch in ray_trainer.py
You can set the other settings in the configuration
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from collections import defaultdict, Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Tuple
from rouge_score import rouge_scorer
import json
import re


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return math.compute_score
    elif data_source == 'DigitalLearningGmbH/MATH-lighteval':
        return math.compute_score
    else:
        raise NotImplementedError

def gaussian_kernel_matrix_normalized(x, sigma=1.0):
    """
    Compute Gaussian (RBF) kernel matrix for normalized vectors.

    Args:
        x: Tensor of shape [batch_size, dim], rows should be unit norm
        sigma: Bandwidth parameter

    Returns:
        kernel_matrix: [batch_size, batch_size]
    """
    # Cosine similarity (since normalized, dot product == cosine sim)
    sim = x @ x.T  # [bs, bs]
    
    # Squared distance from similarity
    dist_sq = 2 * (1 - sim)

    # Gaussian kernel
    gamma = 1.0 / (2 * sigma ** 2)
    kernel_matrix = torch.exp(-gamma * dist_sq)

    return kernel_matrix

def json_format_reward(output: str) -> float:
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            city = data["city"]
            return 1.0
        except:
            return 0.5
    else:
        return 0.0


class RewardManager():
    """
    The reward manager.
    """

    def __init__(self, tokenizer, num_examine, num_responses, dqo_config, is_train=True) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.num_responses = num_responses
        self.is_train = is_train
        self.dqo_cfg = dqo_config
        self.qd_coeff = dqo_config.alpha

        if self.is_train:
            self.embedding_model = SentenceTransformer(dqo_config.embedding_model)

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            reward_tensor = data.batch['rm_scores']
            rule_based_r = False
        else:
            data_source = data[0].non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            rule_based_r = True

        valid_response_length_list = [] # to locate the reward  
        responses_str_list = [] # to calculate det
        set_index = defaultdict(list) # to figure out prompt-response pair
        print_answers = defaultdict(list) # print out responses to check

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            responses_str = self.tokenizer.decode(valid_response_ids)
            
            # store necessary values
            responses_str_list.append(responses_str)
            set_index[data_item.non_tensor_batch['uid']].append(i)
            valid_response_length_list.append(valid_response_length)

            # if no rm, calculate rule-based reward
            if rule_based_r:
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                score = compute_score_fn(solution_str=responses_str, ground_truth=ground_truth)
                reward_tensor[i, valid_response_length - 1] = score
            
            # store examples to print out
            if len(print_answers) < self.num_examine or data_item.non_tensor_batch['uid'] in print_answers:
                print_answers[data_item.non_tensor_batch['uid']].append((prompt_str, responses_str))
        
        if self.is_train:
            response_embeddings = self.embedding_model.encode(responses_str_list)
            response_embeddings = torch.from_numpy(response_embeddings)
            
            # incorporate reward to embedding; optional
            #reward_tensor = torch.exp(reward_tensor) ** 2.0 # hyperparameter to balance the reward and diversity
            #response_embeddings = response_embeddings * reward_tensor.unsqueeze(1) # multiplying reward to embedding

            # calculate determinant
            real_det_list = []
            training_det_list = []
            if self.dqo_cfg.is_distance:
                dist_list = []
            
            for i in set_index: # for each prompt
                X = response_embeddings[set_index[i]]
                X_norm = F.normalize(X, p=2, dim=1)
                
                if self.dqo_cfg.kernels=="cosine":
                    kernel_matrix = X_norm @ X_norm.T
                elif self.dqo_cfg.kernels=="gaussian":
                    kernel_matrix = gaussian_kernel_matrix_normalized(X_norm)
                else:
                    raise ValueError(
                            f"Unknown kernel type: {self.dqo_cfg.kernels}. "
                            "Expected one of {'cosine', 'gaussian'}."
                        )

                real_det = torch.linalg.det(kernel_matrix)
                real_det_list.append(real_det)
                training_det = torch.linalg.det(kernel_matrix + torch.eye(self.num_responses))
                training_det_list.append(training_det)

                # pair-wise distance
                if self.dqo_cfg.is_distance:
                    dist_matrix = torch.cdist(X, X, p=2)  # Euclidean distance
                    avg_dist = dist_matrix.sum(dim=1) / (self.num_responses - 1)
                    avg_group_dist = dist_matrix.sum() / (self.num_responses * (self.num_responses - 1))
                    dist_list.append(avg_group_dist)

                if self.qd_coeff > 0:
                    sub_dets = []
                    for j in range(len(set_index[i])):
                        idx = torch.tensor([k for k in range(len(set_index[i])) if k != j])
                        sub_matrix = kernel_matrix[idx][:, idx]
                        sub_dets.append(torch.linalg.det(sub_matrix + torch.eye(self.num_responses-1)))

                    for idx, j in enumerate(set_index[i]):
                        if not self.dqo_cfg.is_distance:
                            reward_tensor[j, valid_response_length_list[j] - 1] += self.qd_coeff * (torch.log(training_det) - torch.log(sub_dets[idx]))
                        else:
                            reward_tensor[j, valid_response_length_list[j] - 1] += self.qd_coeff * avg_dist[idx]

            return reward_tensor, sum(real_det_list)/len(real_det_list), sum(training_det_list)/len(training_det_list)
        else:
            # for validation, calculate diversity metrics
            metrics_name_list = ['self-bleu', 'self-rouge', 'distinct-3', 'distinct-2', 'distinct-1']
            valid_metrics_dict = {k: [] for k in metrics_name_list}
            best_of_n_scores = []

            for i in set_index:
                response_group = [responses_str_list[j] for j in set_index[i]]
                # self bleu score
                valid_metrics_dict['self-bleu'].append(self._calculate_self_bleu(responses=response_group))
                # self rouge score
                valid_metrics_dict['self-rouge'].append(self._calculate_self_rouge(responses=response_group))
                # ngram score
                valid_metrics_dict['distinct-3'].append(self._ngram_statistics(response_group, n=3))
                valid_metrics_dict['distinct-2'].append(self._ngram_statistics(response_group, n=2))
                valid_metrics_dict['distinct-1'].append(self._ngram_statistics(response_group, n=1))
                # best of n scores
                rewards_per_q = [reward_tensor[j, valid_response_length_list[j] - 1] for j in set_index[i]]
                best_of_n_scores.append(max(rewards_per_q))
            
            # calculate the mean value for each metric
            mean_metrics_dict = {k: sum(v)/len(v) if v else 0 for k, v in valid_metrics_dict.items()}

            # print out examples to check
            for i in print_answers:
                for seq in print_answers[i]:
                    prompt_str, response_str = seq
                    print(f"Prompt:\n{prompt_str}\nResponse:\n{response_str}")

            return reward_tensor, sum(best_of_n_scores)/len(best_of_n_scores), mean_metrics_dict

    def _calculate_self_bleu(self, responses, ngram=4):
        """
        Computes Self-BLEU score for a list of responses.
        Lower score = higher diversity.
        """
        weights_map = {
            1: (1.0, 0, 0, 0),
            2: (0.5, 0.5, 0, 0),
            3: (0.33, 0.33, 0.33, 0),
            4: (0.25, 0.25, 0.25, 0.25),
        }
        weights = weights_map.get(ngram, (0.25, 0.25, 0.25, 0.25))

        scores = []
        for i in range(len(responses)):
            candidate = nltk.word_tokenize(responses[i])
            references = [nltk.word_tokenize(r) for j, r in enumerate(responses) if j != i]
            score = sentence_bleu(references, candidate, weights=weights, smoothing_function=SmoothingFunction().method1)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_self_rouge(self, responses):
        """
        Calculate the average self-ROUGE-L score for a list of strings.
        High score = more similar (less diverse).
        """
        if len(responses) < 2:
            return 0.0
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = []

        for i, hyp in enumerate(responses):
            # references are all other responses
            refs = responses[:i] + responses[i+1:]
            # Compute ROUGE-L against each reference
            for ref in refs:
                score = scorer.score(hyp, ref)['rougeL'].fmeasure
                scores.append(score)

        return sum(scores)/len(scores)

    def _get_ngrams(self, text: str, n: int):
        tokens = text.split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def _ngram_statistics(self, responses, n=4):
        all_ngrams = []
        ngram_counts_per_response = []

        for response in responses:
            ngrams = self._get_ngrams(response, n)
            all_ngrams.extend(ngrams)
            ngram_counts_per_response.append(len(set(ngrams)))

        total_ngrams = len(all_ngrams)
        unique_ngrams = len(set(all_ngrams))
        diversity = unique_ngrams / total_ngrams if total_ngrams > 0 else 0

        return diversity


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer,
                              num_examine=0,
                              num_responses=config.actor_rollout_ref.rollout.n,
                              dqo_config=config.dqo,
                              is_train=True)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer,
                                  num_examine=1,
                                  num_responses=config.actor_rollout_ref.rollout.n,
                                  dqo_config=config.dqo,
                                  is_train=False)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()