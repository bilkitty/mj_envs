from mj_envs_vision.algos.baselines import Planet


def train(env, experience, policy, optimiser):
  # sample from experience

  # update models

  # generate action

  # forward sim

  pass


def evaluate(cfg, policy):
  # instantiate test env


  # close env
  pass


if __name__ == "__main__":
  # TODO: sanity check
  #       compare metrics of this training loop with those of PlaNet/main.py

  # load config (json)

  # TODO: create worker setup and parallelise
  # initialise models
  # instantiate env
  # initialise experience buffer

  # for N episodes
  #   train on env
  #   (opt) evalute on test env
  #   collect experience

  # save performance metrics
  # visualise performance
  pass
