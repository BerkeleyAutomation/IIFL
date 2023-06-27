
class ParallelAgent:
    """
    An abstract class containing an API for all parallel agents to implement.
    """

    def __init__(self, envs, exp_cfg, logdir):
        raise NotImplementedError

    def add_transitions(self, transitions):
        raise NotImplementedError

    def train(self, t):
        raise NotImplementedError

    def get_actions(self, states, t):
        raise NotImplementedError

    def get_allocation_metrics(self, states, t): 
        # should be called after get_actions() in a given timestep
        raise NotImplementedError

    def save(self):
        # save model weights or other key info
        raise NotImplementedError

        # https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        self.remove_unpicklable(state)
        return state

    def remove_unpicklable(self, state):
        raise NotImplementedError

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)

    @classmethod
    def load(cls, resume_logdir, envs, exp_cfg, logdir):
        raise NotImplementedError
