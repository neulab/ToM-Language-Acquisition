# flake8: noqa: E128
from asyncore import write


if True:
    import argparse
    import os
    import random
    import time
    from distutils.util import strtobool
    import spacy
    nlp = spacy.load("en_core_web_sm")

    import gym
    import wandb
    import numpy as np
    import transformers
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.categorical import Categorical
    from torch.utils.tensorboard import SummaryWriter
    from referential_game_env import ReferentialGameEnv
    from tom_speaker import TOMSpeaker
    from coco_speaker import COCOSpeaker
    from metrics import Fluency, SemanticSimilarity, sentence_length, num_nouns


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="ReferentialGame-v0",
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="ToM-language-acquisition-train",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--render-html', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="whether to save HTML images")
    parser.add_argument('--run-name', type=str, default="test",
        help="run name to save HTML files under")
    parser.add_argument('--render-every-N', type=int, default=50000,
        help="render an HTML file every N updates")
    parser.add_argument('--captions-file', type=str, default="train_org",
        help="file to get auxiliary captions from")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--log-nouns', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will keep track of how many nouns it is generating (significantly slows down code)')
    parser.add_argument('--less-logging', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='logs every 1000 timesteps instead of every timestep (recommended for optimization)')

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=4,
        help='the number of parallel game environments')
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--exp-decay', type=float, default=0.994)
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=1.0,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent-coef', type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=None,
        help='the target KL divergence threshold')

    parser.add_argument('--supervised-coef', type=float, default=0.01, help='the ratio of supervised loss')
    parser.add_argument('--length-pen', type=float, default=0.0, help='length penalty')

    # tom training arguments
    parser.add_argument('--use-coco', type=lambda x:bool(strtobool(x)), default = False, nargs='?', 
        const = True, help = 'toggle usage of COCOSpeaker')
    parser.add_argument('--use-tom', type=lambda x:bool(strtobool(x)), default = False, nargs='?', 
        const = True, help = 'toggle usage of theory of mind')
    parser.add_argument('--sigma', type=float, default = 0.5, help = "exploration sigma value for ToM speaker")
    parser.add_argument('--tom-weight', type=float, default=1.0, 
        help = "If using a ToM speaker, what weight to give to ToM listener ranking")
    parser.add_argument('--tom-losscoef', type=float, default=1, help = "coef for tom loss")
    parser.add_argument('--separate-training', type=lambda x:bool(strtobool(x)), default = False, nargs='?',
        const = True, help = "Separate ToM Listener training from rest of network")
    parser.add_argument('--beam-size', type=int, default=25,
        help = "number of candidates to generate for ToM listener")
    parser.add_argument('--beam-search', type=lambda x:bool(strtobool(x)), default = False, nargs = '?',
        const = True, help = 'use beam search instead of sampling')
    parser.add_argument('--tom-anneal', type=lambda x:bool(strtobool(x)), default = False, nargs='?',
        const = True, help = 'toggle anneal of ToM listener influence')
    parser.add_argument('--tom-anneal-start', type=float, default=0.2, 
        help = "fraction of updates that must pass to start using ToM listener")
    parser.add_argument('--sigma-decay', type=lambda x:bool(strtobool(x)), default = False, nargs='?',
        const = True, help = 'toggle anneal of ToM listener influence')
    parser.add_argument('--sigma-decay-end', type=float, default=1.0, 
        help = "fraction of updates that must pass to converge to final sigma value")
    parser.add_argument('--sigma-low', type=float, default=0.1, 
        help = "final sigma value to converge to")
    parser.add_argument('--gold-standard', type=lambda x:bool(strtobool(x)), default = False, nargs='?',
        const = True, help = 'RSA (give ToM speaker access to actual listener)')
    
    # Environment specific arguments
    parser.add_argument('--vocabulary-size', type=int, 
        default=200,
        help='vocabulary size of speaker')
    parser.add_argument('--max-len', type=int,
        default=20,
        help='maximum utterance length')
    parser.add_argument('--game-file-path', type=str)
    parser.add_argument('--dev-game-file-path', type=str, default="data/game_file_dev.pt")

    parser.add_argument('--theta-1', type=float, default=.4, help='theta 1')
    parser.add_argument('--theta-2', type=float, default=.9, help='theta 2')
    parser.add_argument('--model-path', type=str, default=None, help='the path of the model')
    parser.add_argument('--n-distr', type=int, default=2)
    parser.add_argument('--distribution', type=str, default='uniform', help='uniform or zipf')

    parser.add_argument('--sup-coef-decay', action='store_true', help='decay supervised coeff')
    parser.add_argument('--D_img', type=int, default=2048,)
    parser.add_argument('--pretrained-path', type=str, default=None,
        help='load in the wandb path for a pretrained model if you want to run in evaluation mode')

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    
    fluency = Fluency(device="cpu")
    semantic_similarity = SemanticSimilarity()
    ################################################################################
    # Setup Experiment and Logger                                                  #
    ################################################################################
    if True:
        run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=args.exp_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    ################################################################################
    # Seeding                                                                      #
    ################################################################################
    if True:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
    ################################################################################
    # Device                                                                       #
    ################################################################################
    if True:
        # device = torch.device("cpu")
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    ################################################################################
    # Referential Game Environments                                                #
    ################################################################################
    envs = ReferentialGameEnv(max_len=args.max_len,
                 eos_id=3,
                 noop_penalty=0.5,
                 length_penalty=args.length_pen,
                 batch_size=4,
                 n_distr=args.n_distr,
                 game_file_path=args.game_file_path,
                 theta_1=args.theta_1,
                 theta_2=args.theta_2,
                 distribution=args.distribution,
                 model_path = args.model_path,
                 captions_file = args.captions_file)
    dev_envs = ReferentialGameEnv(max_len=args.max_len,
                eos_id=3,
                noop_penalty=0.5,
                length_penalty=args.length_pen,
                batch_size=256,
                n_distr=args.n_distr,
                game_file_path=args.dev_game_file_path,
                theta_1=args.theta_1,
                theta_2=args.theta_2,
                distribution=args.distribution,
                model_path = args.model_path,
                captions_file = args.captions_file)
    i2w = torch.load("i2w")
    ################################################################################
    # Agent                                                                        #
    ################################################################################
    if args.pretrained_path is not None:
        args.learning_rate = 0.0
        speaker_path = "wandb/" + args.pretrained_path + "/files/speaker_model.pt"
        if args.use_tom:
            listener_path = "wandb/" + args.pretrained_path + "/files/tom_listener.pt"
            # speaker = torch.load(speaker_path)
            # tom_listener = torch.load(listener_path)
            agent = TOMSpeaker(maxlen=args.max_len, vocabsize=args.vocabulary_size, 
                    sigma=args.sigma, beam_size=args.beam_size, tom_weight = args.tom_weight,
                    use_pretrained=args.gold_standard, beam_search = args.beam_search,
                    loaded_model_paths=(speaker_path,listener_path)).to(device)
        else:
            agent = Speaker(max_len=args.max_len, vocabulary_size=args.vocabulary_size).to(device)
            agent.load_state_dict(torch.load(speaker_path))
    else:
        freq_words = list(range(200))
        tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
        if args.use_tom:
            agent = TOMSpeaker(maxlen=args.max_len, vocabsize=tokenizer.vocab_size, sigma=args.sigma, 
            beam_size=args.beam_size, tom_weight = args.tom_weight, use_coco = True, word_list=freq_words, 
            use_pretrained=args.gold_standard, beam_search = args.beam_search).to(device)
        else:
            agent = COCOSpeaker(
                max_len=args.max_len,
                vocabulary_size=tokenizer.vocab_size,
                D_img=args.D_img,
                word_list=freq_words  # manually put in <pad>
            ).to(device)
            agent = torch.jit.script(agent)
        
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    ################################################################################
    # Rollout Buffer                                                               #
    ################################################################################
    if True:
        images = torch.zeros((args.num_steps, args.num_envs, 1+args.n_distr) + envs.image_size).to(device)
        images_original = torch.zeros((args.num_steps, args.num_envs) + envs.image_size).to(device)
        targets = torch.zeros(args.num_steps, args.num_envs).long().to(device)
        choices = torch.zeros(args.num_steps, args.num_envs).long().to(device)
        controls = torch.zeros(args.num_steps, args.num_envs).long().to(device)
        actions = torch.zeros(args.num_steps, args.num_envs, args.max_len).long().to(device)
        logprobs = torch.zeros(args.num_steps, args.num_envs, args.max_len).to(device)
        rewards = torch.zeros(args.num_steps, args.num_envs, args.max_len).to(device)
        values = torch.zeros(args.num_steps, args.num_envs, args.max_len).to(device)
        feedback = torch.zeros(args.num_steps, args.num_envs, args.max_len).to(device)
        feedback_mask = torch.zeros(args.num_steps, args.num_envs).to(device)
        tom_mask = torch.zeros(args.num_steps, args.num_envs).to(device)
    ################################################################################
    # Start Game                                                                   #
    ################################################################################
    if True:
        global_step = 0
        start_time = time.time()
        obs = envs.reset()
        B = obs["images"].shape[0]
        next_images = torch.Tensor(
            obs["images"][range(B), :]
        ).to(device)
        next_images_original = torch.Tensor(
            obs["images"][range(B), obs["goal"]]
        ).to(device)
        next_target = torch.Tensor(obs["goal"]).long().to(device)
        num_updates = args.total_timesteps // args.batch_size
        tom_anneal_update = num_updates*args.tom_anneal_start
        sigma_decay_update = num_updates*args.sigma_decay_end

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            if args.exp_decay == 1.0:
                frac = 1.0 - (update - 1.0) / num_updates
            else:
                frac = args.exp_decay ** (update/100)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        if args.sup_coef_decay:
            sup_coef = (1.0 - (update - 1.0) / num_updates) * args.supervised_coef
        else:
            sup_coef = args.supervised_coef

        if args.tom_anneal:
            tom_weight = args.tom_weight*max(update - tom_anneal_update, 0)/(num_updates - tom_anneal_update)
            agent.update_tom_weight(tom_weight)
        
        if args.sigma_decay:
            new_sigma = max(sigma_decay_update - update, 0)/(num_updates) *(args.sigma - args.sigma_low) + args.sigma_low
            agent.update_sigma(new_sigma)
            
        ################################################################################
        # Rollout                                                                      #
        ################################################################################
        average_reward = []
        average_accuracy = []
        length_list = []
        noun_list = []

        with torch.no_grad(): # no need to track gradient in rollouts
            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                # Act and Store
                if args.use_tom:
                    sentence, logprob, _, value = agent.sample(next_images, next_target)
                    values[step] = value.view(args.num_envs, args.max_len) # remove flatten here. see what will happen
                    actions[step] = sentence
                    logprobs[step] = logprob
                    images[step] = next_images
                    targets[step] = next_target
                elif args.use_coco:
                    sentence, logprob, _, value = agent.get_action_and_value(images=next_images_original)
                    values[step] = value.view(args.num_envs, args.max_len)
                    actions[step] = sentence
                    logprobs[step] = logprob
                    images[step] = next_images
                    targets[step] = next_target
                else:
                    sentence, logprob, _, value = agent.get_action_and_value(next_images_original)
                    values[step] = value.view(args.num_envs, args.max_len) # remove flatten here. see what will happen
                    actions[step] = sentence
                    logprobs[step] = logprob
                    images_original[step] = next_images_original
                # Step and Store
                if args.render_html and (global_step % args.render_every_N == 0):
                    obs, reward = envs.step(sentence.cpu().numpy(), render=True, name=args.exp_name + "_" + str(global_step))
                else:
                    obs, reward = envs.step(sentence.cpu().numpy())

                rewards[step] = torch.tensor(reward).to(device)
                next_images = torch.Tensor(
                    obs["images"][range(B), :]
                ).to(device)
                next_target = torch.Tensor(obs["goal"]).long().to(device)
                next_images_original = torch.Tensor(
                    obs["images"][range(B), obs["goal"]]
                ).to(device)
                feedback[step] = torch.tensor(obs["feedback"]).to(device)
                choices[step] = obs["choices"].clone().detach().to(device)
                controls[step] = obs["controls"].clone().detach().to(device)
                feedback_mask[step] = ((controls[step] == 1).float()).clone().detach().to(device)
                tom_mask[step] = ((controls[step] <= 1).float()).clone().detach().to(device)                
                # Logging
                average_reward.append(float(rewards.sum(dim=-1).mean()))
                average_accuracy.append(obs["accuracy"])
                length_list.extend([sentence_length(' '.join(map(lambda x: i2w[x], sent.cpu().tolist()))) for sent in sentence])
                if args.log_nouns:
                    noun_list.extend([num_nouns(nlp, ' '.join(map(lambda x: i2w[x], sent.cpu().tolist()))) for sent in sentence])

                avg_return = sum(average_reward)/len(average_reward)
                avg_accuracy = sum(average_accuracy)/len(average_accuracy)
                if not args.less_logging or global_step % 1000 == 0:
                    writer.add_scalar("charts/episodic_return", avg_return, global_step)
                    writer.add_scalar("charts/episode_accuracy", avg_accuracy, global_step)
                    average_reward = []
                    average_accuracy = []

                if global_step % 1000 == 0:
                    print(f"global_step={global_step}, episodic_return={avg_return}, episode_acc={avg_accuracy}")
                    fluency_list = [fluency(' '.join(map(lambda x: i2w[x], sent.cpu().tolist()))) for sent in sentence]
                    average_fluency = sum(fluency_list)/len(fluency_list)
                    semantic_similarity_list \
                        = [semantic_similarity(
                            ' '.join(map(lambda x: i2w[x], sent1.cpu().tolist())),
                            ' '.join(map(lambda x: i2w[x], sent2))
                        ) for sent1, sent2 in zip(sentence, obs["ground_truth"])]
                    average_semantic_similarity = sum(semantic_similarity_list)/len(semantic_similarity_list)
                    average_utterance_length = sum(length_list)/len(length_list)
                    if args.log_nouns:
                        noun_proportion = sum(noun_list)/sum(length_list)
                        noun_list = []
                    length_list = []
                    
                    writer.add_scalar("charts/average_fluency", average_fluency, global_step)
                    writer.add_scalar("charts/average_semantic_similarity", average_semantic_similarity, global_step)
                    writer.add_scalar("charts/average_utterance_length", average_utterance_length, global_step)
                    if args.log_nouns:
                        writer.add_scalar("charts/fraction_of_nouns", noun_proportion, global_step)

                    print(f"average_fluency={average_fluency}, average_semantic_similarity={average_semantic_similarity}, average_utterance_length={average_utterance_length}")
        ################################################################################
        # Dev Performance                                                              #
        ################################################################################
        with torch.no_grad():
            obs = envs.reset()
            B = obs["images"].shape[0]
            next_images = torch.Tensor(
                    obs["images"][range(B), :]
            ).to(device)
            next_target = torch.Tensor(obs["goal"]).long().to(device)
            next_images_original = torch.Tensor(
                obs["images"][range(B), obs["goal"]]
            ).to(device)
            if args.use_tom:
                sentence, logprob, _, value = agent.sample(next_images, next_target)
            elif args.use_coco:
                sentence, logprob, _, value = agent.get_action_and_value(next_images_original)
            else:
                sentence, logprob, _, value = agent.get_action_and_value(next_images_original)
            obs, reward = envs.step(sentence.cpu().numpy())
            dev_reward = rewards.sum(dim=-1).mean()
            dev_accuracy = obs["accuracy"]
            writer.add_scalar("charts/dev_return", dev_reward, global_step)
            writer.add_scalar("charts/dev_accuracy", dev_accuracy, global_step)
        ################################################################################
        # Advantage Estimation                                                         #
        ################################################################################
        with torch.no_grad():
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.max_len)):
                    if t == args.max_len - 1:
                        nextvalues = 0
                    else:
                        nextvalues = values[:, :, t + 1]  # TODO: put length in front
                    delta = rewards[:, :, t] + args.gamma * nextvalues - values[:, :, t]
                    advantages[:, :, t] = lastgaelam = delta + args.gamma * \
                        args.gae_lambda * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.max_len)):
                    if t == args.num_steps - 1:
                        next_return = 0
                    else:
                        next_return = returns[:, :, t + 1]
                    returns[:, :, t] = rewards[:, :, t] + args.gamma * next_return
                advantages = returns - values
        ################################################################################
        # Flatten Batch                                                                #
        ################################################################################
        if True:
            b_images = images.reshape((-1,1+args.n_distr) + envs.image_size)
            b_images_original = images_original.reshape((-1,)+envs.image_size)
            b_targets = targets.reshape(-1)
            b_choices = choices.reshape(-1)
            b_controls = controls.reshape(-1)
            b_logprobs = logprobs.reshape(-1, args.max_len)
            b_actions = actions.reshape(-1, args.max_len)
            b_advantages = advantages.reshape(-1, args.max_len)
            b_returns = returns.reshape(-1, args.max_len)
            b_values = values.reshape(-1, args.max_len)
            b_feedback = feedback.reshape(-1, args.max_len)
            b_feedback_mask = feedback_mask.reshape(-1)
            b_tom_mask = tom_mask.reshape(-1)
        ################################################################################
        # Optimizing the policy and value network                                     #
        ################################################################################
        if True:
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    if args.use_tom:
                        _, newlogprob, entropy, newvalue = agent.sample(b_images[mb_inds], b_targets[mb_inds], actions=b_actions.long()[mb_inds], beam_size = 1)
                    elif args.use_coco:
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_images_original[mb_inds], actions=b_actions.long()[mb_inds])
                    else:
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_images_original[mb_inds], b_actions.long()[mb_inds])
                    newvalue = newvalue.view(args.minibatch_size, args.max_len)
                    logratio = (newlogprob - b_logprobs[mb_inds])
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        # old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() >
                                    args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (
                            mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * \
                        torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * \
                            ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    # supervised loss
                    if args.use_tom:
                        supervised_loss = agent.supervised_loss(b_images[mb_inds], b_feedback[mb_inds].long(), b_targets[mb_inds], b_feedback_mask[mb_inds])
                        if not args.gold_standard:
                            if args.separate_training:
                                tom_loss = agent.tom_listener.train_step(b_images[mb_inds], b_choices[mb_inds], b_actions[mb_inds], b_tom_mask[mb_inds])
                                loss = (1-sup_coef) * loss + sup_coef * supervised_loss
                            else:
                                tom_loss = agent.tom_listener.supervised_loss(b_images[mb_inds], b_choices[mb_inds], b_actions[mb_inds], b_tom_mask[mb_inds])
                                loss = (1-sup_coef) * loss + sup_coef * supervised_loss + args.tom_losscoef*tom_loss
                        else:
                            loss = (1-sup_coef) * loss + sup_coef * supervised_loss
                    elif args.use_coco:
                        supervised_loss = agent.supervised_loss(
                        b_images_original[mb_inds],
                        b_feedback[mb_inds].long(),
                        b_feedback_mask[mb_inds])
                        loss = (1 - sup_coef) * loss + sup_coef * supervised_loss
                    else:
                        supervised_loss = agent.supervised_loss(
                            b_images_original[mb_inds], b_feedback[mb_inds].long(), b_feedback_mask[mb_inds])
                        loss = (1-sup_coef) * loss + sup_coef * supervised_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        agent.parameters(), args.max_grad_norm)
                    optimizer.step()


                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break
        ################################################################################
        # Logging                                                                      #
        ################################################################################
        if True:
            sample_actions = actions[0][0].cpu().tolist()
            sentence = list(map(lambda x: i2w[x], sample_actions))
            print(' '.join(sentence))
            sample_feedback = feedback[0][0].cpu().tolist()
            sample_feedback = list(map(lambda x: i2w[x], sample_feedback))
            print(' '.join(sample_feedback))

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - \
                np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_text("sampled_sentence", ' '.join(sentence), global_step)
            writer.add_text("sampled_feedback", ' '.join(sample_feedback), global_step)
            writer.add_scalar("charts/learning_rate",
                            optimizer.param_groups[0]["lr"], global_step)
            if args.use_tom and not args.gold_standard:
                writer.add_scalar("losses/tom_loss", tom_loss, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance",
                            explained_var, global_step)
            writer.add_scalar("losses/supervised_loss", supervised_loss.item(), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step /
                            (time.time() - start_time)), global_step)

    envs.close()
    dev_envs.close()
    if args.use_tom:
        torch.save(agent.speaker.state_dict(), os.path.join(wandb.run.dir, "speaker_model.pt"))
        torch.save(agent.tom_listener.state_dict(), os.path.join(wandb.run.dir, "tom_listener.pt"))
        if args.use_coco:
            agent.speaker.save(os.path.join(wandb.run.dir, "jit_speaker_model.pt"))
    else:
        torch.save(agent.state_dict(), os.path.join(wandb.run.dir, "speaker_model.pt"))
        if args.use_coco:
            agent.save(os.path.join(wandb.run.dir, "jit_speaker_model.pt"))
    writer.close()
    
